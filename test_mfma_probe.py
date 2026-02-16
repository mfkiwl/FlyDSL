#!/usr/bin/env python3
"""Probe MFMA 16x16x16 bf16 register layout by feeding identity matrices.

Strategy: Set A=I (identity), B=I (identity), then D = I*I = I.
Each output element D[m,n] should be 1.0 if m==n, else 0.0.
By checking which output positions are 1.0, we can verify the store mapping.

Then: Set A=I, B=arbitrary, D should = B (in f32).
This confirms the B loading is correct.

Then: Set A=arbitrary, B=I, D should = A (in f32).
This confirms the A loading is correct.
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
MODULE_NAME = "mfma_probe"
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
            A: lambda: memref(M * K, T.bf16),
            B: lambda: memref(K * N, T.bf16),
            D: lambda: memref(M * N, T.f32),
        ):
            i32 = T.i32
            vec4_i16 = T.i16x4
            vec4_f32 = T.f32x4

            tid = gpu.thread_id("x")
            c4_i32 = arith.constant(4, type=i32)
            c16_i32 = arith.constant(16, type=i32)
            c16 = arith.constant(16, index=True)

            tid_i32 = buffer_ops.index_cast_to_i32(tid)
            lane_mod_16 = tid % c16
            lane_div_16 = tid / c16
            lm16 = buffer_ops.index_cast_to_i32(lane_mod_16)
            ld16 = buffer_ops.index_cast_to_i32(lane_div_16)

            a_rsrc = buffer_ops.create_buffer_resource(A, num_records_bytes=M * K * 2)
            b_rsrc = buffer_ops.create_buffer_resource(B, num_records_bytes=K * N * 2)
            d_rsrc = buffer_ops.create_buffer_resource(D, num_records_bytes=M * N * 4)

            # ── Load A: try mapping row=lm16, k_base=ld16*4, load 4 contiguous along K ──
            # A[row, k] = A_flat[row*16 + k], load A[lm16, ld16*4 : ld16*4+4]
            a_off = buffer_ops.i32_add(
                buffer_ops.i32_mul(lm16, c16_i32),
                buffer_ops.i32_mul(ld16, c4_i32),
            )
            a_vec = buffer_ops.buffer_load(a_rsrc, a_off, vec_width=4,
                                           dtype=ir.IntegerType.get_signless(16))

            # ── Load B: try same mapping col=lm16, k_base=ld16*4, strided along K ──
            # B[k, col] = B_flat[k*16 + col]
            n_i32 = lm16
            kb = buffer_ops.i32_mul(ld16, c4_i32)

            def lb(ko):
                k = buffer_ops.i32_add(kb, arith.constant(ko, type=i32))
                return buffer_ops.buffer_load(b_rsrc,
                    buffer_ops.i32_add(buffer_ops.i32_mul(k, c16_i32), n_i32),
                    vec_width=1, dtype=ir.BF16Type.get())

            bv = vector.from_elements(
                ir.VectorType.get([4], ir.BF16Type.get()),
                [lb(0), lb(1), lb(2), lb(3)])
            b_vec = vector.bitcast(vec4_i16, bv)

            acc = arith.constant_vector(0.0, vec4_f32)

            result = rocdl.mfma_f32_16x16x16bf16_1k(
                vec4_f32, [a_vec, b_vec, acc, 0, 0, 0])

            # ── Store: just dump the raw result per lane ──
            # Write result[lane][reg] to D_flat[lane*4 + reg]
            # This way we can see the exact output per lane/register.
            base = buffer_ops.i32_mul(tid_i32, c4_i32)
            c1_i32 = arith.constant(1, type=i32)
            c2_i32 = arith.constant(2, type=i32)
            c3_i32 = arith.constant(3, type=i32)

            v0 = vector.extract(result, static_position=[int(0)], dynamic_position=[])
            buffer_ops.buffer_store(v0, d_rsrc, base)
            v1 = vector.extract(result, static_position=[int(1)], dynamic_position=[])
            buffer_ops.buffer_store(v1, d_rsrc, buffer_ops.i32_add(base, c1_i32))
            v2 = vector.extract(result, static_position=[int(2)], dynamic_position=[])
            buffer_ops.buffer_store(v2, d_rsrc, buffer_ops.i32_add(base, c2_i32))
            v3 = vector.extract(result, static_position=[int(3)], dynamic_position=[])
            buffer_ops.buffer_store(v3, d_rsrc, buffer_ops.i32_add(base, c3_i32))

        @flir.jit
        def __call__(
            self: flir.T.i64,
            A: lambda: memref(M * K, T.bf16),
            B: lambda: memref(K * N, T.bf16),
            D: lambda: memref(M * N, T.f32),
        ):
            c1 = arith.constant(1, index=True)
            bdx = arith.constant(WAVESIZE, index=True)
            flir.gpu_ext.LaunchFuncOp(
                [MODULE_NAME, KERNEL_NAME],
                grid_size=(c1, c1, c1),
                block_size=(bdx, c1, c1),
                kernel_operands=[A, B, D],
            )

    return flydsl.compile(_Mod())


def main():
    print(f"=== MFMA Register Layout Probe ===")
    print(f"GPU: {get_hip_arch()}")
    compiled = compile_kernel()

    # Test 1: A=I, B=I → D should be I
    print("\n--- Test 1: A=I, B=I → D should be I ---")
    A = torch.eye(16, dtype=torch.bfloat16, device='cuda').view(-1)
    B = torch.eye(16, dtype=torch.bfloat16, device='cuda').view(-1)
    D = torch.zeros(M * N, dtype=torch.float32, device='cuda')

    compiled(A, B, D)
    torch.cuda.synchronize()

    # D is stored as D_flat[lane*4 + reg] = result[lane][reg]
    # Print per-lane results
    print("Lane -> [reg0, reg1, reg2, reg3]")
    for lane in range(64):
        vals = D[lane*4 : lane*4+4].tolist()
        nonzero = any(abs(v) > 0.001 for v in vals)
        if nonzero or lane < 4 or lane in [16, 32, 48]:
            print(f"  lane {lane:2d} (mod16={lane%16:2d}, div16={lane//16}): {[round(v,3) for v in vals]}")

    # Find the 1.0 positions to determine the mapping
    print("\nPositions where result == 1.0:")
    for lane in range(64):
        for reg in range(4):
            val = D[lane*4 + reg].item()
            if abs(val - 1.0) < 0.01:
                print(f"  lane={lane} (mod16={lane%16}, div16={lane//16}), reg={reg} → 1.0")

    # Test 2: A=I, B=sequential → D should = B
    print("\n--- Test 2: A=I, B=arange → D should = B ---")
    B2 = torch.arange(256, dtype=torch.bfloat16, device='cuda')  # B[k,n] = k*16+n
    D2 = torch.zeros(M * N, dtype=torch.float32, device='cuda')
    compiled(A, B2, D2)
    torch.cuda.synchronize()

    print("Lane -> [reg0, reg1, reg2, reg3]  (should match B[?, lane%16])")
    for lane in [0, 1, 15, 16, 17, 31, 32, 48]:
        vals = D2[lane*4 : lane*4+4].tolist()
        print(f"  lane {lane:2d} (mod16={lane%16:2d}, div16={lane//16}): {[round(v,1) for v in vals]}")


if __name__ == "__main__":
    main()
