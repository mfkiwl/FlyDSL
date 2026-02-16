#!/usr/bin/env python3
"""Test kernel: buffer_load to AccVGPR + v_mfma_f32_16x16x16_bf16

Demonstrates loading matrix A directly from global memory into AccVGPR
using buffer_load with ACC=1 (a[] register notation), then using MFMA
with A in AccVGPR and B in ArchVGPR.

On gfx942 (CDNA3), AccVGPR and ArchVGPR share a unified physical register
file, so a[] and v[] are aliases. The ACC bit in the instruction encoding
is set automatically by the assembler based on a[]/v[] notation.

Memory layout (K-contiguous):
  A: A[m, k] row-major → K contiguous → buffer_load_dwordx2 to a[]
  B: B^T[n, k] transposed → K contiguous → buffer_load_dwordx2 to v[]
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
MODULE_NAME = "mfma_accvgpr_test"
KERNEL_NAME = "kernel_main"


def _unwrap(v):
    return buffer_ops._unwrap_value(v)


def _asm(asm_str, constraints, operands, result_types, has_side_effects=True):
    """Emit LLVM inline asm op."""
    return llvm.InlineAsmOp(
        result_types,
        operands,
        asm_str,
        constraints,
        has_side_effects=has_side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


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
            A:   lambda: memref(M * K, T.bf16),
            B_T: lambda: memref(N * K, T.bf16),
            D:   lambda: memref(M * N, T.f32),
        ):
            i32 = T.i32
            vec4_f32 = T.f32x4

            tid = gpu.thread_id("x")
            c4_i32 = arith.constant(4, type=i32)
            c16_i32 = arith.constant(16, type=i32)
            c16 = arith.constant(16, index=True)

            lm16 = buffer_ops.index_cast_to_i32(tid % c16)
            ld16 = buffer_ops.index_cast_to_i32(tid / c16)

            a_rsrc  = buffer_ops.create_buffer_resource(A,   num_records_bytes=M*K*2)
            bt_rsrc = buffer_ops.create_buffer_resource(B_T, num_records_bytes=N*K*2)
            d_rsrc  = buffer_ops.create_buffer_resource(D,   num_records_bytes=M*N*4)

            k_base = buffer_ops.i32_mul(ld16, c4_i32)
            elem_off = buffer_ops.i32_add(
                buffer_ops.i32_mul(lm16, c16_i32), k_base)

            # Byte offset for bf16 elements
            c2_i32 = arith.constant(2, type=i32)
            byte_off = buffer_ops.i32_mul(elem_off, c2_i32)

            # Scalar offset 0
            soffset_0 = buffer_ops._create_i32_constant(0)
            aux_0 = buffer_ops._create_i32_constant(0)

            # ────────────────────────────────────────────
            # Load A into AccVGPR via inline asm
            # buffer_load_dwordx2 a[0:1], voffset, rsrc, 0 offen
            # The "a" constraint tells LLVM to allocate AccVGPR
            # ────────────────────────────────────────────
            # Use LLVM type for inline asm result
            llvm_i64 = ir.IntegerType.get_signless(64)

            a_load_op = _asm(
                "buffer_load_dwordx2 $0, $2, $1, 0 offen",
                "=a,s,v",
                [_unwrap(a_rsrc), _unwrap(byte_off)],
                [llvm_i64],
            )
            a_i64 = a_load_op.results[0]

            # ────────────────────────────────────────────
            # Load B into ArchVGPR (normal buffer_load)
            # ────────────────────────────────────────────
            b_vec = buffer_ops.buffer_load(bt_rsrc, elem_off, vec_width=4,
                                           dtype=ir.IntegerType.get_signless(16))

            # Convert b_vec (vector<4xi16>) to i64 for asm operand
            i32_ty = ir.IntegerType.get_signless(32)
            vec2_i32 = ir.VectorType.get([2], i32_ty)
            b_as_i32x2 = vector.bitcast(vec2_i32, b_vec)
            vec1_i64 = ir.VectorType.get([1], llvm_i64)
            b_as_i64v = vector.bitcast(vec1_i64, b_as_i32x2)
            b_i64 = vector.extract(b_as_i64v, static_position=[int(0)], dynamic_position=[])

            # ────────────────────────────────────────────
            # MFMA via inline asm:
            #   v_mfma_f32_16x16x16_bf16 a[0:3], a[0:1], v[2:3], 0
            #   A from AccVGPR ("a" constraint), B from ArchVGPR ("v" constraint)
            #   C/D in AccVGPR ("a" constraint for output)
            # ────────────────────────────────────────────
            # Wait for loads to complete
            _asm("s_waitcnt vmcnt(0)", "", [], [], has_side_effects=True)

            f32_ty = ir.F32Type.get()
            # Result is 4xf32 = 128 bits. Use LLVM struct<(f32,f32,f32,f32)> or
            # just use vector<4xf32> as LLVM type
            llvm_v4f32 = ir.VectorType.get([4], f32_ty)

            mfma_op = _asm(
                "v_mfma_f32_16x16x16_bf16 $0, $1, $2, 0",
                "=a,a,v",
                [a_i64, _unwrap(b_i64)],
                [llvm_v4f32],
            )
            result = mfma_op.results[0]

            # ────────────────────────────────────────────
            # Store D
            # ────────────────────────────────────────────
            n_out = lm16
            m_base = buffer_ops.i32_mul(ld16, c4_i32)
            c1_i32 = arith.constant(1, type=i32)
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
    print(f"=== MFMA AccVGPR Test: buffer_load → AccVGPR → MFMA ===")
    print(f"GPU arch: {get_hip_arch()}")
    print(f"D[{M},{N}] = A[{M},{K}] @ B[{K},{N}]")
    print(f"A in AccVGPR (buffer_load ACC=1), B in ArchVGPR")
    compiled = compile_kernel()

    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    B = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')
    D = torch.zeros(M, N, dtype=torch.float32, device='cuda')
    expected = torch.matmul(A.float(), B.float())
    B_T = B.t().contiguous()

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
        print("PASS - buffer_load→AccVGPR→MFMA works correctly!")
    else:
        print(f"FAIL: max diff = {max_diff}")


if __name__ == "__main__":
    main()
