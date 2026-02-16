#!/usr/bin/env python3
"""Q*K GEMM kernel aligned with PA_A16W16 PagedAttention ASM kernel's GEMM1 logic.

Computes: Score[16x16, FP32] = K[16x128, BF16] * Q[16x128, BF16]^T

This implements the GEMM1 (Q*K^T) phase of the paged attention kernel using
V_MFMA_F32_16X16X16_BF16, with the same register class assignment and MFMA
decomposition as the pure ASM kernel (pa_a16w16.s).

Key correspondences with pa_a16w16.s GEMM1:
  - K data: loaded into AccVGPR via buffer_load_dwordx2 (ACC=1)
  - Q data: loaded into VGPR via standard buffer_load (ACC=0)
  - MFMA: v_mfma_f32_16x16x16_bf16 v[D], a[A], v[B], {0|v[D]}
    ACC_CD=0 (D in ArchVGPR), ACC[0]=1 (A from AccVGPR), ACC[1]=0 (B from ArchVGPR)
  - 8 MFMA ops accumulate along dim=128 (128 dims / 16 per MFMA = 8 iterations)

K data layout (vLLM paged KV cache format):
  Physical: [head_dim/8, block_size, 8] = [16, 16, 8]
  K_phys[dg, t, de] = K_logical[token=t, dim=dg*8+de]
  This layout makes buffer_load_dwordx2 at v_offset=lane_id*16 directly
  produce the correct data for MFMA A operand.

Q data layout: row-major [16, 128]
  Q[head_n, dim_k] at flat element index = head_n * 128 + dim_k

MFMA A operand (K): lane l provides A[m=l%16, k=(l/16)*4+j]
  = K[token=l%16, dim=(dim_group)*8+j]
MFMA B operand (Q): lane l provides B[k=(l/16)*4+j, n=l%16]
  = Q[head=l%16, dim=(dim_group)*8+j]
MFMA D output:      lane l provides D[n=l%16, m=(l/16)*4+reg_idx]
  = Score[token=(l/16)*4+reg_idx, head=l%16]
"""

import functools
import torch
import flydsl
from flydsl.dialects.ext import flir, arith, gpu, buffer_ops, rocdl, vector
from flydsl.lang.ir.types import T, memref
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from _mlir import ir
from _mlir.dialects import llvm

# Problem dimensions
Q_ROWS = 16       # number of Q heads (fills full 16x16 MFMA)
K_ROWS = 16       # number of K tokens (= block_size)
DIM = 128          # head dimension
NUM_MFMA = 8       # DIM / 16 = 128 / 16 = 8 MFMA iterations
WAVESIZE = 64
MODULE_NAME = "qk_gemm_pa_test"
KERNEL_NAME = "qk_gemm_kernel"


def _unwrap(value):
    """Unwrap ArithValue wrappers to get raw MLIR ir.Value."""
    max_depth = 10
    depth = 0
    while depth < max_depth and not isinstance(value, ir.Value):
        if hasattr(value, '_value'):
            value = value._value
        elif hasattr(value, 'value'):
            value = value.value
        else:
            break
        depth += 1
    return value


# ─────────────────────────────────────────────────────────────────────────────
# Inline-asm helpers (matching pa_a16w16.s patterns)
# ─────────────────────────────────────────────────────────────────────────────

def accvgpr_buffer_load_dwordx2(rsrc, voffset_bytes, offset=0):
    """buffer_load_dwordx2 -> AccVGPR (ACC=1).

    Loads 8 bytes (= 4 x bf16 = 2 x i32) directly into AccVGPR.
    Returns: i64 residing in a[N:N+1].

    Matches pa_a16w16.s pattern:
      buffer_load_dwordx4 acc[0:3], v22, s[16:19], 0 offen
    We use dwordx2 (returning i64) to avoid sub-register extraction issues.
    """
    i64 = ir.IntegerType.get_signless(64)
    if offset > 0:
        asm_str = f"buffer_load_dwordx2 $0, $1, $2, 0 offen offset:{offset}"
    else:
        asm_str = "buffer_load_dwordx2 $0, $1, $2, 0 offen"
    return llvm.InlineAsmOp(
        res=i64,
        operands_=[_unwrap(voffset_bytes), _unwrap(rsrc)],
        asm_string=asm_str,
        constraints="=a,v,s",
        has_side_effects=True,
        is_align_stack=False,
    ).result


def _pack_vec4i16_to_i64(b_vec):
    """Pack vector<4xi16> (in VGPR) into i64 for inline asm operand."""
    i32 = ir.IntegerType.get_signless(32)
    i64 = ir.IntegerType.get_signless(64)
    vec2_i32 = ir.VectorType.get([2], i32)
    vec1_i64 = ir.VectorType.get([1], i64)
    b_i32x2 = vector.bitcast(vec2_i32, _unwrap(b_vec))
    b_i64v = vector.bitcast(vec1_i64, b_i32x2)
    return vector.extract(b_i64v, static_position=[int(0)], dynamic_position=[])


def mfma_bf16_16x16x16_first(a_i64_accvgpr, b_i64_vgpr):
    """First MFMA: D = A * B + 0 (C=0, initialize accumulator).

    v_mfma_f32_16x16x16_bf16 v[D], a[A], v[B], 0
    """
    f32 = ir.F32Type.get()
    vec4_f32 = ir.VectorType.get([4], f32)
    return llvm.InlineAsmOp(
        res=vec4_f32,
        operands_=[_unwrap(a_i64_accvgpr), _unwrap(b_i64_vgpr)],
        asm_string="v_mfma_f32_16x16x16_bf16 $0, $1, $2, 0",
        constraints="=&v,a,v",
        has_side_effects=True,
        is_align_stack=False,
    ).result


def mfma_bf16_16x16x16_accum(a_i64_accvgpr, b_i64_vgpr, c_vec4f32):
    """Accumulating MFMA: D = A * B + C (C=previous D).

    v_mfma_f32_16x16x16_bf16 v[D], a[A], v[B], v[D]

    Constraint "0" ties $3 (C input) to $0 (D output) so they share the
    same VGPR. Early-clobber "&" prevents D from overlapping B.
    """
    f32 = ir.F32Type.get()
    vec4_f32 = ir.VectorType.get([4], f32)
    return llvm.InlineAsmOp(
        res=vec4_f32,
        operands_=[_unwrap(a_i64_accvgpr), _unwrap(b_i64_vgpr), _unwrap(c_vec4f32)],
        asm_string="v_mfma_f32_16x16x16_bf16 $0, $1, $2, $3",
        constraints="=&v,a,v,0",
        has_side_effects=True,
        is_align_stack=False,
    ).result


def mfma_bf16_16x16x16_accum_last(a_i64_accvgpr, b_i64_vgpr, c_vec4f32):
    """Last MFMA with trailing s_nop for hazard protection.

    MFMA is multi-pass (4 passes). Without s_nop, LLVM may schedule
    VGPR reads of the result before the MFMA fully completes.
    18 cycles of nop covers the 4-pass latency.
    """
    f32 = ir.F32Type.get()
    vec4_f32 = ir.VectorType.get([4], f32)
    return llvm.InlineAsmOp(
        res=vec4_f32,
        operands_=[_unwrap(a_i64_accvgpr), _unwrap(b_i64_vgpr), _unwrap(c_vec4f32)],
        asm_string=(
            "v_mfma_f32_16x16x16_bf16 $0, $1, $2, $3\n"
            "s_nop 7\n"
            "s_nop 7\n"
            "s_nop 1"
        ),
        constraints="=&v,a,v,0",
        has_side_effects=True,
        is_align_stack=False,
    ).result


# ─────────────────────────────────────────────────────────────────────────────
# Kernel compilation
# ─────────────────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=32)
def compile_kernel():
    gpu_arch = get_hip_arch()

    class _Mod(flir.MlirModule):
        GPU_MODULE_NAME = MODULE_NAME
        GPU_MODULE_TARGETS = [
            f'#rocdl.target<chip = "{gpu_arch}", abi = "500", features = "+sramecc,+xnack">'
        ]

        @flir.kernel
        def qk_gemm_kernel(
            self: flir.T.i64,
            Q:     lambda: memref(Q_ROWS * DIM, T.bf16),
            K_buf: lambda: memref(DIM // 8 * K_ROWS * 8, T.bf16),
            S:     lambda: memref(K_ROWS * Q_ROWS, T.f32),
        ):
            i32 = T.i32
            i16 = ir.IntegerType.get_signless(16)

            tid = gpu.thread_id("x")
            lane_id = buffer_ops.index_cast_to_i32(tid)

            c1_i32 = arith.constant(1, type=i32)
            c2_i32 = arith.constant(2, type=i32)
            c3_i32 = arith.constant(3, type=i32)
            c4_i32 = arith.constant(4, type=i32)
            c8_i32 = arith.constant(8, type=i32)
            c16_i32 = arith.constant(16, type=i32)
            c128_i32 = arith.constant(128, type=i32)

            c16 = arith.constant(16, index=True)
            lane_mod_16 = tid % c16
            lane_div_16 = tid / c16
            lm16 = buffer_ops.index_cast_to_i32(lane_mod_16)
            ld16 = buffer_ops.index_cast_to_i32(lane_div_16)

            # ═══════════════════════════════════════════════════════════════
            # Create buffer resource descriptors
            # ═══════════════════════════════════════════════════════════════
            q_rsrc = buffer_ops.create_buffer_resource(
                Q, num_records_bytes=Q_ROWS * DIM * 2
            )
            k_rsrc = buffer_ops.create_buffer_resource(
                K_buf, num_records_bytes=(DIM // 8) * K_ROWS * 8 * 2
            )
            s_rsrc = buffer_ops.create_buffer_resource(
                S, num_records_bytes=K_ROWS * Q_ROWS * 4
            )

            # ═══════════════════════════════════════════════════════════════
            # Load K into AccVGPR via buffer_load_dwordx2
            # ═══════════════════════════════════════════════════════════════
            # K is in vLLM paged format: [head_dim/8, block_size, 8]
            # K_phys[dg, t, de] = K[token=t, dim=dg*8+de]
            # Flat byte offset: dg*256 + t*16 + de*2
            #
            # Per lane: v_k_offset = lane_id * 16 (byte offset)
            #   lane l: dg = l/16, t = l%16
            #   loads K[dg=l/16, t=l%16, de=0..7] = 8 BF16 = 16 bytes
            #
            # With buffer_load_dwordx2 (8 bytes each), we split each 16-byte
            # chunk into two halves:
            #   offset+0: de=0..3 (for MFMA #2i)
            #   offset+8: de=4..7 (for MFMA #2i+1)
            #
            # 4 dim chunks (offset 0/1024/2048/3072) x 2 halves = 8 loads
            k_byte_off = buffer_ops.i32_mul(lane_id, c16_i32)

            # 8 loads: 4 dim chunks x 2 halves (de=0..3, de=4..7)
            k_acc_0 = accvgpr_buffer_load_dwordx2(k_rsrc, k_byte_off, offset=0)
            k_acc_1 = accvgpr_buffer_load_dwordx2(k_rsrc, k_byte_off, offset=8)
            k_acc_2 = accvgpr_buffer_load_dwordx2(k_rsrc, k_byte_off, offset=1024)
            k_acc_3 = accvgpr_buffer_load_dwordx2(k_rsrc, k_byte_off, offset=1032)
            k_acc_4 = accvgpr_buffer_load_dwordx2(k_rsrc, k_byte_off, offset=2048)
            k_acc_5 = accvgpr_buffer_load_dwordx2(k_rsrc, k_byte_off, offset=2056)
            k_acc_6 = accvgpr_buffer_load_dwordx2(k_rsrc, k_byte_off, offset=3072)
            k_acc_7 = accvgpr_buffer_load_dwordx2(k_rsrc, k_byte_off, offset=3080)
            k_acc = [k_acc_0, k_acc_1, k_acc_2, k_acc_3,
                     k_acc_4, k_acc_5, k_acc_6, k_acc_7]

            # ═══════════════════════════════════════════════════════════════
            # Load Q into VGPR via standard buffer_load
            # ═══════════════════════════════════════════════════════════════
            # Q is row-major [16, 128]: Q[head, dim] at element head*128+dim
            #
            # For MFMA B operand at iteration i, lane l needs:
            #   Q[l%16, (dim_group)*8 + j] for j=0..3
            # where dim_group depends on the MFMA iteration and the lane group.
            #
            # For MFMA #(2*c + h) where c=dim_chunk(0..3), h=half(0..1):
            #   dim_start = (l/16)*8 + c*32 + h*4
            #   Q element offset = (l%16)*128 + dim_start
            #
            # Each load: 4 BF16 = vec_width=4, dtype=i16
            q_base = buffer_ops.i32_add(
                buffer_ops.i32_mul(lm16, c128_i32),
                buffer_ops.i32_mul(ld16, c8_i32),
            )

            # Load Q for each of 8 MFMA iterations
            # Dim offsets: chunk0(de0-3), chunk0(de4-7), chunk1, ..., chunk3
            c0_i32 = arith.constant(0, type=i32)
            c4v_i32 = arith.constant(4, type=i32)
            c32_i32 = arith.constant(32, type=i32)
            c36_i32 = arith.constant(36, type=i32)
            c64_i32 = arith.constant(64, type=i32)
            c68_i32 = arith.constant(68, type=i32)
            c96_i32 = arith.constant(96, type=i32)
            c100_i32 = arith.constant(100, type=i32)

            q_off_list = [c0_i32, c4v_i32, c32_i32, c36_i32,
                          c64_i32, c68_i32, c96_i32, c100_i32]

            q_vgpr = []
            for dim_off_c in q_off_list:
                q_elem_off = buffer_ops.i32_add(q_base, dim_off_c)
                q_vec = buffer_ops.buffer_load(
                    q_rsrc, q_elem_off, vec_width=4, dtype=i16
                )
                q_vgpr.append(_pack_vec4i16_to_i64(q_vec))

            # ═══════════════════════════════════════════════════════════════
            # Wait for all loads to complete
            # ═══════════════════════════════════════════════════════════════
            rocdl.s_waitcnt(0)

            # ═══════════════════════════════════════════════════════════════
            # 8x MFMA: Score = K * Q^T along dim=128
            # ═══════════════════════════════════════════════════════════════
            # Each MFMA processes a 16-element dim chunk:
            #   D[16x16] = A_K[16x16] * B_Q[16x16] + C
            #
            # Matches pa_a16w16.s GEMM1 pattern:
            #   v_mfma_f32_16x16x16_bf16 v[96:99], acc[0:1],  v[80:81], 0
            #   v_mfma_f32_16x16x16_bf16 v[96:99], acc[2:3],  v[82:83], v[96:99]
            #   ... (6 more)
            result = mfma_bf16_16x16x16_first(k_acc[0], q_vgpr[0])
            result = mfma_bf16_16x16x16_accum(k_acc[1], q_vgpr[1], result)
            result = mfma_bf16_16x16x16_accum(k_acc[2], q_vgpr[2], result)
            result = mfma_bf16_16x16x16_accum(k_acc[3], q_vgpr[3], result)
            result = mfma_bf16_16x16x16_accum(k_acc[4], q_vgpr[4], result)
            result = mfma_bf16_16x16x16_accum(k_acc[5], q_vgpr[5], result)
            result = mfma_bf16_16x16x16_accum(k_acc[6], q_vgpr[6], result)
            result = mfma_bf16_16x16x16_accum_last(k_acc[7], q_vgpr[7], result)

            # ═══════════════════════════════════════════════════════════════
            # Store Score from VGPR to global memory
            # ═══════════════════════════════════════════════════════════════
            # MFMA result layout: lane l holds
            #   Score[m=(l/16)*4+reg_idx, n=l%16] for reg_idx=0..3
            # Score is [K_ROWS, Q_ROWS] = [16, 16] row-major
            #   flat index = m * 16 + n
            n_out = lm16
            m_base = buffer_ops.i32_mul(ld16, c4_i32)

            v0 = vector.extract(result, static_position=[int(0)], dynamic_position=[])
            d0_off = buffer_ops.i32_add(buffer_ops.i32_mul(m_base, c16_i32), n_out)
            buffer_ops.buffer_store(v0, s_rsrc, d0_off)

            v1 = vector.extract(result, static_position=[int(1)], dynamic_position=[])
            m1 = buffer_ops.i32_add(m_base, c1_i32)
            d1_off = buffer_ops.i32_add(buffer_ops.i32_mul(m1, c16_i32), n_out)
            buffer_ops.buffer_store(v1, s_rsrc, d1_off)

            v2 = vector.extract(result, static_position=[int(2)], dynamic_position=[])
            m2 = buffer_ops.i32_add(m_base, c2_i32)
            d2_off = buffer_ops.i32_add(buffer_ops.i32_mul(m2, c16_i32), n_out)
            buffer_ops.buffer_store(v2, s_rsrc, d2_off)

            v3 = vector.extract(result, static_position=[int(3)], dynamic_position=[])
            m3 = buffer_ops.i32_add(m_base, c3_i32)
            d3_off = buffer_ops.i32_add(buffer_ops.i32_mul(m3, c16_i32), n_out)
            buffer_ops.buffer_store(v3, s_rsrc, d3_off)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            Q:     lambda: memref(Q_ROWS * DIM, T.bf16),
            K_buf: lambda: memref(DIM // 8 * K_ROWS * 8, T.bf16),
            S:     lambda: memref(K_ROWS * Q_ROWS, T.f32),
        ):
            c1 = arith.constant(1, index=True)
            bdx = arith.constant(WAVESIZE, index=True)
            flir.gpu_ext.LaunchFuncOp(
                [MODULE_NAME, KERNEL_NAME],
                grid_size=(c1, c1, c1),
                block_size=(bdx, c1, c1),
                kernel_operands=[Q, K_buf, S],
            )

    return flydsl.compile(_Mod())


# ─────────────────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Q*K GEMM — aligned with PA_A16W16 GEMM1 logic")
    print("=" * 70)
    print(f"GPU arch: {get_hip_arch()}")
    print(f"Score[{K_ROWS},{Q_ROWS}] = K[{K_ROWS},{DIM}] @ Q[{Q_ROWS},{DIM}]^T  (bf16->f32)")
    print(f"MFMA iterations: {NUM_MFMA} (dim={DIM}, 16 per MFMA)")
    print()
    print("Register class assignment (matching pa_a16w16.s):")
    print("  A (K data):  AccVGPR  (buffer_load_dwordx2, inline asm '=a')")
    print("  B (Q data):  ArchVGPR (standard buffer_load)")
    print("  D (Score):   ArchVGPR (inline asm '=&v', ACC_CD=0)")
    print()

    compiled = compile_kernel()

    torch.manual_seed(42)
    Q_ref = torch.randn(Q_ROWS, DIM, dtype=torch.bfloat16, device='cuda')
    K_ref = torch.randn(K_ROWS, DIM, dtype=torch.bfloat16, device='cuda')
    S_out = torch.zeros(K_ROWS, Q_ROWS, dtype=torch.float32, device='cuda')

    # Expected: Score = K @ Q^T
    expected = torch.matmul(K_ref.float(), Q_ref.float().T)

    # Convert K to vLLM paged format: [head_dim/8, block_size, 8]
    # K_ref is [16, 128] -> reshape to [16, 16, 8] (token, dim_group, elem)
    # Then permute to [16, 16, 8] (dim_group, token, elem) = vLLM format
    K_paged = K_ref.reshape(K_ROWS, DIM // 8, 8).permute(1, 0, 2).contiguous()

    print(f"Q shape:       {Q_ref.shape} (row-major)")
    print(f"K_ref shape:   {K_ref.shape} (logical)")
    print(f"K_paged shape: {K_paged.shape} (vLLM: [dim/8, block_size, 8])")
    print(f"Score shape:   {S_out.shape}")

    Q_flat = Q_ref.contiguous().view(-1)
    K_flat = K_paged.contiguous().view(-1)
    S_flat = S_out.view(-1)

    print("\nRunning kernel...")
    compiled(Q_flat, K_flat, S_flat)
    torch.cuda.synchronize()

    print(f"\nOutput Score[:4,:4]:")
    for i in range(4):
        print(f"  {[round(x, 4) for x in S_out[i, :4].tolist()]}")
    print(f"Expected[:4,:4]:")
    for i in range(4):
        print(f"  {[round(x, 4) for x in expected[i, :4].tolist()]}")

    max_diff = (S_out - expected).abs().max().item()
    print(f"\nMax abs diff: {max_diff:.6f}")

    if torch.allclose(S_out, expected, atol=0.5, rtol=0.1):
        print("PASS — Q*K GEMM matches PA_A16W16 GEMM1 computation!")
    else:
        print(f"FAIL: max diff = {max_diff}")
        print(f"S_out[0] = {S_out[0].tolist()}")
        print(f"expected[0] = {expected[0].tolist()}")


if __name__ == "__main__":
    main()
