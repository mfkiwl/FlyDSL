#!/usr/bin/env python3
"""Test kernel: v_mfma_f32_16x16x16_bf16 — single MFMA 16x16x16 matmul

ISA reference: V_MFMA_F32_16X16X16_BF16 (opcode 97)
  D = A(16x16) * B(16x16) + C(16x16)
  A, B: bf16; C, D: f32.  4 passes, 16 cycles.

Register layout (gfx942, wave64):
  A input  (vector<4xi16>): lane l → row m = l%16,  k = (l/16)*4 + elem_idx
  B input  (vector<4xi16>): lane l → col n = l%16,  k = (l/16)*4 + elem_idx
  D output (vector<4xf32>): lane l → col n = l%16,  row m = (l/16)*4 + reg_idx

Memory layout (both K-contiguous for vector loads):
  A stored as A[m, k] row-major:    A_flat[m*16 + k]  → K is inner dim, contiguous
  B stored as B_T[n, k] transposed: B_T_flat[n*16 + k] → K is inner dim, contiguous
  Both lanes load 4 consecutive bf16 along K → single buffer_load_dwordx2 each.

Kernel: 1 wave (64 threads), D[16,16] = A[16,16] @ B[16,16].
"""

import functools
import torch
import flydsl
from flydsl.dialects.ext import flir, arith, gpu, buffer_ops, rocdl, vector
from flydsl.lang.ir.types import T, memref
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from _mlir import ir

M, N, K = 16, 16, 16
WAVESIZE = 64
MODULE_NAME = "mfma_16x16x16_bf16_test"
KERNEL_NAME = "kernel_main"


@functools.lru_cache(maxsize=32)
def compile_kernel():
    gpu_arch = get_hip_arch()

    class _Mod(flir.MlirModule):
        GPU_MODULE_NAME = MODULE_NAME
        GPU_MODULE_TARGETS = [
            f'#rocdl.target<chip = "{gpu_arch}", abi = "500", features = "+sramecc,+xnack">'
        ]

        @flir.kernel
        def kernel_main(
            self: flir.T.i64,
            A:   lambda: memref(M * K, T.bf16),   # A[m, k] row-major, K contiguous
            B_T: lambda: memref(N * K, T.bf16),   # B^T[n, k] transposed, K contiguous
            D:   lambda: memref(M * N, T.f32),     # D[m, n] row-major output
        ):
            i32 = T.i32
            vec4_i16 = T.i16x4
            vec4_f32 = T.f32x4

            tid = gpu.thread_id("x")
            tid_i32 = buffer_ops.index_cast_to_i32(tid)

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

            k_base_i32 = buffer_ops.i32_mul(ld16, c4_i32)   # k_base = (lane/16)*4

            # ── Load A: A[m, k] row-major, A_flat[m*16 + k] ──
            # m = lane%16, k = k_base+[0..3], offset = m*16 + k_base
            # 4 contiguous bf16 along K → vector load
            a_off = buffer_ops.i32_add(
                buffer_ops.i32_mul(lm16, c16_i32),   # m * 16
                k_base_i32,                            # + k_base
            )
            a_vec = buffer_ops.buffer_load(a_rsrc, a_off, vec_width=4,
                                           dtype=ir.IntegerType.get_signless(16))

            # ── Load B: B_T[n, k] row-major, B_T_flat[n*16 + k] ──
            # n = lane%16, k = k_base+[0..3], offset = n*16 + k_base
            # 4 contiguous bf16 along K → vector load
            b_off = buffer_ops.i32_add(
                buffer_ops.i32_mul(lm16, c16_i32),   # n * 16
                k_base_i32,                            # + k_base
            )
            b_vec = buffer_ops.buffer_load(bt_rsrc, b_off, vec_width=4,
                                           dtype=ir.IntegerType.get_signless(16))

            # ── Zero accumulator ──
            acc = arith.constant_vector(0.0, vec4_f32)

            # ── MFMA: D = A * B + 0 ──
            result = rocdl.mfma_f32_16x16x16bf16_1k(
                vec4_f32, [a_vec, b_vec, acc, 0, 0, 0])

            # ── Store D (row-major) ──
            # Result: col n = lane%16, row m = (lane/16)*4 + reg_idx
            n_out = lm16
            m_base = buffer_ops.i32_mul(ld16, c4_i32)

            c1_i32 = arith.constant(1, type=i32)
            c2_i32 = arith.constant(2, type=i32)
            c3_i32 = arith.constant(3, type=i32)

            v0 = vector.extract(result, static_position=[int(0)], dynamic_position=[])
            buffer_ops.buffer_store(v0, d_rsrc,
                buffer_ops.i32_add(buffer_ops.i32_mul(m_base, c16_i32), n_out))

            v1 = vector.extract(result, static_position=[int(1)], dynamic_position=[])
            m1 = buffer_ops.i32_add(m_base, c1_i32)
            buffer_ops.buffer_store(v1, d_rsrc,
                buffer_ops.i32_add(buffer_ops.i32_mul(m1, c16_i32), n_out))

            v2 = vector.extract(result, static_position=[int(2)], dynamic_position=[])
            m2 = buffer_ops.i32_add(m_base, c2_i32)
            buffer_ops.buffer_store(v2, d_rsrc,
                buffer_ops.i32_add(buffer_ops.i32_mul(m2, c16_i32), n_out))

            v3 = vector.extract(result, static_position=[int(3)], dynamic_position=[])
            m3 = buffer_ops.i32_add(m_base, c3_i32)
            buffer_ops.buffer_store(v3, d_rsrc,
                buffer_ops.i32_add(buffer_ops.i32_mul(m3, c16_i32), n_out))

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
    print(f"=== MFMA 16x16x16 BF16 Matmul Test (K-contiguous layout) ===")
    print(f"GPU arch: {get_hip_arch()}")
    print(f"D[{M},{N}] = A[{M},{K}] @ B[{K},{N}]  (bf16 input, f32 accumulation)")
    print(f"A stored as A[m,k] row-major (K contiguous)")
    print(f"B stored as B^T[n,k] transposed (K contiguous)")
    compiled = compile_kernel()

    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')   # A[m, k]
    B = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')   # B[k, n]
    D = torch.zeros(M, N, dtype=torch.float32, device='cuda')

    expected = torch.matmul(A.float(), B.float())

    # B^T[n, k] = B[k, n].T → shape [N, K], K is inner contiguous dim
    B_T = B.t().contiguous()   # [N=16, K=16]

    print(f"A   shape: {A.shape}   (row-major A[m,k], K contiguous)")
    print(f"B_T shape: {B_T.shape} (transposed B^T[n,k], K contiguous)")

    compiled(A.contiguous().view(-1), B_T.view(-1), D.view(-1))
    torch.cuda.synchronize()

    print(f"\nOutput D[:4,:4] =")
    for i in range(4):
        print(f"  {[round(x, 4) for x in D[i,:4].tolist()]}")
    print(f"Expected[:4,:4] =")
    for i in range(4):
        print(f"  {[round(x, 4) for x in expected[i,:4].tolist()]}")

    max_diff = (D - expected).abs().max().item()
    print(f"Max abs diff: {max_diff:.6f}")

    if torch.allclose(D, expected, atol=0.1, rtol=0.05):
        print("PASS - mfma_f32_16x16x16_bf16 (K-contiguous) works correctly!")
    else:
        print(f"FAIL: max diff = {max_diff}")
        print(f"D[0] = {D[0].tolist()}")
        print(f"E[0] = {expected[0].tolist()}")


if __name__ == "__main__":
    main()
