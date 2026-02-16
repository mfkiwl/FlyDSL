#!/usr/bin/env python3
"""Q*K GEMM kernel aligned with PA_A16W16 PagedAttention ASM kernel's GEMM1 logic.

Uses native ROCDL intrinsics (NO inline asm) with explicit register class control:
  - K data: forced into AccVGPR via rocdl.to_agpr_v4i16 intrinsic
  - MFMA D/C: forced into ArchVGPR via rocdl.mfma_f32_16x16x16bf16_1k_vcd intrinsic
  - Q data: standard VGPR via buffer_load

Target ISA (matching pa_a16w16.s GEMM1):
  buffer_load_dwordx2 acc[0:1], ...   // K -> AccVGPR (via to_agpr)
  buffer_load_dwordx2 v[80:81], ...   // Q -> VGPR
  v_mfma_f32_16x16x16_bf16 v[96:99], acc[0:1], v[80:81], 0  // A=AccVGPR, D=VGPR

Computes: Score[16x16, FP32] = K[16x128, BF16] * Q[16x128, BF16]^T

K data layout (vLLM paged KV cache format):
  Physical: [head_dim/8, block_size, 8] = [16, 16, 8]
  K_phys[dg, t, de] = K_logical[token=t, dim=dg*8+de]

Q data layout: row-major [16, 128]
"""

import functools
import torch
import flydsl
from flydsl.dialects.ext import flir, arith, gpu, buffer_ops, rocdl, vector
from flydsl.lang.ir.types import T, memref
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from _mlir import ir
from _mlir.dialects import arith as std_arith

# Problem dimensions
Q_ROWS = 16
K_ROWS = 16
DIM = 128
NUM_MFMA = 8
WAVESIZE = 64
MODULE_NAME = "qk_gemm_pa_test"
KERNEL_NAME = "qk_gemm_kernel"


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
            i16 = ir.IntegerType.get_signless(16)
            f32 = ir.F32Type.get()
            v4f32_type = ir.VectorType.get([4], f32)

            i32 = T.i32
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
            # Buffer resource descriptors
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
            # Load K as v4i16 (dwordx2), force into AccVGPR
            # ═══════════════════════════════════════════════════════════════
            # K flat element index = dg*128 + t*8 + de
            # For lane l: base = l*8 (bf16 elements)
            # Each load: 4 bf16 = 8 bytes = dwordx2
            # 8 loads per lane: dim offsets 0,4,512,516,1024,1028,1536,1540
            k_elem_base = buffer_ops.i32_mul(lane_id, c8_i32)

            k_offsets = [0, 4, 512, 516, 1024, 1028, 1536, 1540]
            k_vecs = []
            for off in k_offsets:
                off_c = arith.constant(off, type=i32)
                k_off = buffer_ops.i32_add(k_elem_base, off_c)
                k_vec = buffer_ops.buffer_load(
                    k_rsrc, k_off, vec_width=4, dtype=i16
                )
                k_agpr = rocdl.to_agpr_v4i16(k_vec)
                k_vecs.append(k_agpr)

            # ═══════════════════════════════════════════════════════════════
            # Load Q as v4i16 (dwordx2) into VGPR
            # ═══════════════════════════════════════════════════════════════
            q_base = buffer_ops.i32_add(
                buffer_ops.i32_mul(lm16, c128_i32),
                buffer_ops.i32_mul(ld16, c8_i32),
            )

            q_offsets = [0, 4, 32, 36, 64, 68, 96, 100]
            q_vecs = []
            for off in q_offsets:
                off_c = arith.constant(off, type=i32)
                q_off = buffer_ops.i32_add(q_base, off_c)
                q_vec = buffer_ops.buffer_load(
                    q_rsrc, q_off, vec_width=4, dtype=i16
                )
                q_vecs.append(q_vec)

            # ═══════════════════════════════════════════════════════════════
            # Wait for all loads
            # ═══════════════════════════════════════════════════════════════
            rocdl.s_waitcnt(0)

            # ═══════════════════════════════════════════════════════════════
            # 8x MFMA: A=AccVGPR(K), B=VGPR(Q), D=VGPR (ACC_CD=0)
            # ═══════════════════════════════════════════════════════════════
            c_zero_attr = ir.DenseElementsAttr.get_splat(
                v4f32_type, ir.FloatAttr.get(f32, 0.0)
            )
            c_zero_val = std_arith.ConstantOp(v4f32_type, c_zero_attr).result

            result = rocdl.mfma_f32_16x16x16bf16_1k_vcd(
                v4f32_type, [k_vecs[0], q_vecs[0], c_zero_val, 0, 0, 0]
            )
            result = rocdl.mfma_f32_16x16x16bf16_1k_vcd(
                v4f32_type, [k_vecs[1], q_vecs[1], result, 0, 0, 0]
            )
            result = rocdl.mfma_f32_16x16x16bf16_1k_vcd(
                v4f32_type, [k_vecs[2], q_vecs[2], result, 0, 0, 0]
            )
            result = rocdl.mfma_f32_16x16x16bf16_1k_vcd(
                v4f32_type, [k_vecs[3], q_vecs[3], result, 0, 0, 0]
            )
            result = rocdl.mfma_f32_16x16x16bf16_1k_vcd(
                v4f32_type, [k_vecs[4], q_vecs[4], result, 0, 0, 0]
            )
            result = rocdl.mfma_f32_16x16x16bf16_1k_vcd(
                v4f32_type, [k_vecs[5], q_vecs[5], result, 0, 0, 0]
            )
            result = rocdl.mfma_f32_16x16x16bf16_1k_vcd(
                v4f32_type, [k_vecs[6], q_vecs[6], result, 0, 0, 0]
            )
            result = rocdl.mfma_f32_16x16x16bf16_1k_vcd(
                v4f32_type, [k_vecs[7], q_vecs[7], result, 0, 0, 0]
            )

            # ═══════════════════════════════════════════════════════════════
            # Store Score
            # ═══════════════════════════════════════════════════════════════
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


def main():
    print("=" * 70)
    print("Q*K GEMM — K in AccVGPR, D in VGPR (no inline asm)")
    print("=" * 70)
    print(f"GPU arch: {get_hip_arch()}")
    print(f"Score[{K_ROWS},{Q_ROWS}] = K[{K_ROWS},{DIM}] @ Q[{Q_ROWS},{DIM}]^T")
    print(f"MFMA iterations: {NUM_MFMA}")
    print()
    print("Register class control (matching pa_a16w16.s):")
    print("  K: AccVGPR (rocdl.to_agpr_v4i16)")
    print("  Q: ArchVGPR (buffer_load)")
    print("  D: ArchVGPR (mfma_vcd, ACC_CD=0)")
    print()

    compiled = compile_kernel()

    torch.manual_seed(42)
    Q_ref = torch.randn(Q_ROWS, DIM, dtype=torch.bfloat16, device='cuda')
    K_ref = torch.randn(K_ROWS, DIM, dtype=torch.bfloat16, device='cuda')
    S_out = torch.zeros(K_ROWS, Q_ROWS, dtype=torch.float32, device='cuda')

    expected = torch.matmul(K_ref.float(), Q_ref.float().T)
    K_paged = K_ref.reshape(K_ROWS, DIM // 8, 8).permute(1, 0, 2).contiguous()

    compiled(Q_ref.contiguous().view(-1), K_paged.contiguous().view(-1), S_out.view(-1))
    torch.cuda.synchronize()

    print(f"Output Score[:4,:4]:")
    for i in range(4):
        print(f"  {[round(x, 4) for x in S_out[i, :4].tolist()]}")
    print(f"Expected[:4,:4]:")
    for i in range(4):
        print(f"  {[round(x, 4) for x in expected[i, :4].tolist()]}")

    max_diff = (S_out - expected).abs().max().item()
    print(f"\nMax abs diff: {max_diff:.6f}")

    if torch.allclose(S_out, expected, atol=0.5, rtol=0.1):
        print("PASS — Q*K GEMM (K=AccVGPR, D=VGPR, no inline asm) correct!")
    else:
        print(f"FAIL: max diff = {max_diff}")
        print(f"S_out[0] = {S_out[0].tolist()}")
        print(f"expected[0] = {expected[0].tolist()}")


if __name__ == "__main__":
    main()
