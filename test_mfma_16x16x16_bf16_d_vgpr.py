#!/usr/bin/env python3
"""Test kernel: v_mfma_f32_16x16x16_bf16 with D output in VGPR (not AccVGPR)

Demonstrates controlling MFMA register classes via inline asm constraints:
  - A (left matrix):  loaded to AccVGPR via buffer_load ACC=1
  - B (right matrix): loaded to VGPR via standard buffer_load
  - C (accumulator):  zero (immediate 0 in asm)
  - D (output):       forced to VGPR via "=&v" constraint (ACC_CD=0)

Generated ISA target:
  buffer_load_dwordx2 a[...], ...         ; A → AccVGPR
  buffer_load_dwordx2 v[...], ...         ; B → VGPR
  v_mfma_f32_16x16x16_bf16 v[D], a[A], v[B], 0  ; D in VGPR!
  buffer_store_dword v[...], ...          ; store from VGPR

VOP3P-MAI encoding bits achieved:
  ACC_CD (bit 15) = 0  →  D in ArchVGPR
  ACC[0] (bit 59) = 1  →  A from AccVGPR
  ACC[1] (bit 60) = 0  →  B from ArchVGPR

Key inline-asm lessons for MFMA:
  1. buffer_load to AccVGPR must return i64 (not vector<2xi32>).
     Bitcast/extract on AccVGPR values triggers v_accvgpr_read_b32 moves.
  2. Early-clobber "=&v" is required: MFMA is multi-pass (4 passes),
     partial writes are observable. Without "&", LLVM may overlap D and B.
  3. s_nop delay is required after MFMA inside inline asm: LLVM doesn't
     know the asm contains a multi-cycle MFMA, so it won't insert the
     hazard nops automatically. 18 cycles (s_nop 7 + s_nop 7 + s_nop 1)
     covers the 4-pass latency.

Register layout (gfx942, wave64, V_MFMA_F32_16X16X16_BF16):
  A input  (vector<4xi16>): lane l → row m = l%16,  k = (l/16)*4 + elem_idx
  B input  (vector<4xi16>): lane l → col n = l%16,  k = (l/16)*4 + elem_idx
  D output (vector<4xf32>): lane l → col n = l%16,  row m = (l/16)*4 + reg_idx
"""

import functools
import torch
import flydsl
from flydsl.dialects.ext import flir, arith, gpu, buffer_ops, rocdl, vector
from flydsl.lang.ir.types import T, memref
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from _mlir import ir
from _mlir.dialects import llvm

M, N, K = 16, 16, 16
WAVESIZE = 64
MODULE_NAME = "mfma_16x16x16_bf16_d_vgpr_test"
KERNEL_NAME = "gemm_kernel_16x16x16_bf16_d_vgpr"


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
# Inline-asm helpers
# ─────────────────────────────────────────────────────────────────────────────

def accvgpr_buffer_load_dwordx2(rsrc, voffset_bytes):
    """buffer_load_dwordx2 → AccVGPR (ACC=1).

    Loads 8 bytes (= 4 × bf16 = 2 × i32) directly into AccVGPR.
    Constraint "=a" tells LLVM register allocator → AccVGPR destination.

    Returns: i64 residing in a[N:N+1].
    Using i64 (not vector<2xi32>) avoids LLVM inserting v_accvgpr_read
    when downstream code does bitcast/extract on the result.
    """
    i64 = ir.IntegerType.get_signless(64)
    return llvm.InlineAsmOp(
        res=i64,
        operands_=[_unwrap(voffset_bytes), _unwrap(rsrc)],
        asm_string="buffer_load_dwordx2 $0, $1, $2, 0 offen",
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


def mfma_bf16_a_accvgpr_d_vgpr(a_i64_accvgpr, b_i64_vgpr):
    """MFMA with A in AccVGPR and D in VGPR (ACC_CD=0).

    Generates:
      v_mfma_f32_16x16x16_bf16 v[D], a[A], v[B], 0

    VOP3P-MAI encoding bits:
      ACC_CD (bit 15) = 0  →  D in ArchVGPR   (from "=v" constraint)
      ACC[0] (bit 59) = 1  →  A from AccVGPR  (from "a" constraint)
      ACC[1] (bit 60) = 0  →  B from ArchVGPR (from "v" constraint)

    CRITICAL: a_i64_accvgpr must be an i64 directly from inline asm with
    "=a" constraint. Do NOT bitcast/extract AccVGPR values — LLVM will
    insert v_accvgpr_read_b32 moves that corrupt the register mapping.

    The early-clobber "&" on "=&v" is essential: MFMA is a multi-pass
    instruction (4 passes for 16x16x16) whose partial writes are observable.
    Without "&", LLVM may allocate D overlapping B (both in VGPR), and
    early passes writing D would corrupt B data needed by later passes.
    """
    f32 = ir.F32Type.get()
    vec4_f32 = ir.VectorType.get([4], f32)
    return llvm.InlineAsmOp(
        res=vec4_f32,
        operands_=[_unwrap(a_i64_accvgpr), _unwrap(b_i64_vgpr)],
        asm_string=(
            "v_mfma_f32_16x16x16_bf16 $0, $1, $2, 0\n"
            "s_nop 7\n"
            "s_nop 7\n"
            "s_nop 1"
        ),
        constraints="=&v,a,v",
        has_side_effects=True,
        is_align_stack=False,
    ).result


@functools.lru_cache(maxsize=32)
def compile_kernel():
    gpu_arch = get_hip_arch()

    class _Mod(flir.MlirModule):
        GPU_MODULE_NAME = MODULE_NAME
        GPU_MODULE_TARGETS = [
            f'#rocdl.target<chip = "{gpu_arch}", abi = "500", features = "+sramecc,+xnack">'
        ]

        @flir.kernel
        def gemm_kernel_16x16x16_bf16_d_vgpr(
            self: flir.T.i64,
            A:   lambda: memref(M * K, T.bf16),   # A[m, k] row-major, K contiguous
            B_T: lambda: memref(N * K, T.bf16),   # B^T[n, k] transposed, K contiguous
            D:   lambda: memref(M * N, T.f32),     # D[m, n] row-major output
        ):
            i32 = T.i32

            tid = gpu.thread_id("x")
            c2_i32 = arith.constant(2, type=i32)
            c4_i32 = arith.constant(4, type=i32)
            c16_i32 = arith.constant(16, type=i32)
            c16 = arith.constant(16, index=True)

            lane_mod_16 = tid % c16
            lane_div_16 = tid / c16
            lm16 = buffer_ops.index_cast_to_i32(lane_mod_16)
            ld16 = buffer_ops.index_cast_to_i32(lane_div_16)

            a_rsrc  = buffer_ops.create_buffer_resource(A,   num_records_bytes=M * K * 2)
            bt_rsrc = buffer_ops.create_buffer_resource(B_T, num_records_bytes=N * K * 2)
            d_rsrc  = buffer_ops.create_buffer_resource(D,   num_records_bytes=M * N * 4)

            k_base_i32 = buffer_ops.i32_mul(ld16, c4_i32)   # (lane/16)*4

            # ═══════════════════════════════════════════════════════════════
            # Load A → AccVGPR via buffer_load_dwordx2 with ACC=1
            # ═══════════════════════════════════════════════════════════════
            # A[m, k] row-major: elem_offset = m*16 + k_base
            # byte_offset = elem_offset * 2 (bf16 = 2 bytes)
            a_elem_off = buffer_ops.i32_add(
                buffer_ops.i32_mul(lm16, c16_i32),
                k_base_i32,
            )
            a_byte_off = buffer_ops.i32_mul(a_elem_off, c2_i32)

            # Load 8 bytes into AccVGPR as i64: a[N:N+1] ← global memory
            a_i64 = accvgpr_buffer_load_dwordx2(a_rsrc, a_byte_off)

            # ═══════════════════════════════════════════════════════════════
            # Load B → VGPR via standard buffer_load (ACC=0, default)
            # ═══════════════════════════════════════════════════════════════
            b_elem_off = buffer_ops.i32_add(
                buffer_ops.i32_mul(lm16, c16_i32),
                k_base_i32,
            )
            b_vec = buffer_ops.buffer_load(bt_rsrc, b_elem_off, vec_width=4,
                                           dtype=ir.IntegerType.get_signless(16))
            b_i64 = _pack_vec4i16_to_i64(b_vec)

            # ═══════════════════════════════════════════════════════════════
            # MFMA: D = A * B + 0
            #   ACC_CD=0 → D in ArchVGPR
            #   ACC[0]=1 → A from AccVGPR
            #   ACC[1]=0 → B from ArchVGPR
            # ═══════════════════════════════════════════════════════════════
            rocdl.s_waitcnt(0)
            result = mfma_bf16_a_accvgpr_d_vgpr(a_i64, b_i64)

            # ═══════════════════════════════════════════════════════════════
            # Store D from VGPR to global memory
            # ═══════════════════════════════════════════════════════════════
            # MFMA result is explicitly constrained to VGPR in inline asm.
            # Result layout: lane → col n = lane%16, row m = (lane/16)*4 + reg_idx
            # Each lane holds 4 consecutive rows at the same column.
            #
            # BUT: the 4 f32 values are for rows m_base+0..3 at column n,
            # which are NOT contiguous in row-major memory (stride = 16 elements).
            # So we must store element-by-element with stride.

            n_out = lm16
            m_base = buffer_ops.i32_mul(ld16, c4_i32)
            c1_i32 = arith.constant(1, type=i32)
            c3_i32 = arith.constant(3, type=i32)

            v0 = vector.extract(result, static_position=[int(0)], dynamic_position=[])
            d0_off = buffer_ops.i32_add(buffer_ops.i32_mul(m_base, c16_i32), n_out)
            buffer_ops.buffer_store(v0, d_rsrc, d0_off)

            v1 = vector.extract(result, static_position=[int(1)], dynamic_position=[])
            m1 = buffer_ops.i32_add(m_base, c1_i32)
            d1_off = buffer_ops.i32_add(buffer_ops.i32_mul(m1, c16_i32), n_out)
            buffer_ops.buffer_store(v1, d_rsrc, d1_off)

            v2 = vector.extract(result, static_position=[int(2)], dynamic_position=[])
            m2 = buffer_ops.i32_add(m_base, c2_i32)
            d2_off = buffer_ops.i32_add(buffer_ops.i32_mul(m2, c16_i32), n_out)
            buffer_ops.buffer_store(v2, d_rsrc, d2_off)

            v3 = vector.extract(result, static_position=[int(3)], dynamic_position=[])
            m3 = buffer_ops.i32_add(m_base, c3_i32)
            d3_off = buffer_ops.i32_add(buffer_ops.i32_mul(m3, c16_i32), n_out)
            buffer_ops.buffer_store(v3, d_rsrc, d3_off)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            A:   lambda: memref(M * K, T.bf16),
            B_T: lambda: memref(N * K, T.bf16),
            D:   lambda: memref(M * N, T.f32),
        ):
            c1 = arith.constant(1, index=True)
            bdx = arith.constant(WAVESIZE, index=True)
            flir.gpu_ext.LaunchFuncOp(
                [MODULE_NAME, KERNEL_NAME],
                grid_size=(c1, c1, c1),
                block_size=(bdx, c1, c1),
                kernel_operands=[A, B_T, D],
            )

    return flydsl.compile(_Mod())


def main():
    print("=" * 65)
    print("MFMA 16x16x16 BF16 — D output in VGPR (ACC_CD=0)")
    print("=" * 65)
    print(f"GPU arch: {get_hip_arch()}")
    print(f"D[{M},{N}] = A[{M},{K}] @ B[{K},{N}]  (bf16→f32)")
    print()
    print("Register class assignment:")
    print("  A: AccVGPR  (buffer_load ACC=1, inline asm '=a')")
    print("  B: ArchVGPR (standard buffer_load)")
    print("  D: ArchVGPR (inline asm '=&v', ACC_CD=0)")
    print()

    compiled = compile_kernel()

    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    B = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')
    D = torch.zeros(M, N, dtype=torch.float32, device='cuda')

    expected = torch.matmul(A.float(), B.float())

    B_T = B.t().contiguous()

    print(f"A   shape: {A.shape}  (row-major, K contiguous)")
    print(f"B_T shape: {B_T.shape} (transposed, K contiguous)")

    compiled(A.contiguous().view(-1), B_T.view(-1), D.view(-1))
    torch.cuda.synchronize()

    print(f"\nOutput D[:4,:4]:")
    for i in range(4):
        print(f"  {[round(x, 4) for x in D[i,:4].tolist()]}")
    print(f"Expected[:4,:4]:")
    for i in range(4):
        print(f"  {[round(x, 4) for x in expected[i,:4].tolist()]}")

    max_diff = (D - expected).abs().max().item()
    print(f"\nMax abs diff: {max_diff:.6f}")

    if torch.allclose(D, expected, atol=0.1, rtol=0.05):
        print("PASS — MFMA D-in-VGPR (ACC_CD=0) works correctly!")
    else:
        print(f"FAIL: max diff = {max_diff}")
        print(f"D[0] = {D[0].tolist()}")
        print(f"E[0] = {expected[0].tolist()}")


if __name__ == "__main__":
    main()
