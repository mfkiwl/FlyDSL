#!/usr/bin/env python3
"""Test kernel: v_mfma_f32_16x16x16_bf16 with buffer_load directly to AccVGPR

Demonstrates CDNA3 buffer_load with ACC=1 bit: data loads from global memory
directly into AccVGPR registers, bypassing the standard VGPR → AccVGPR copy.

CDNA3 ISA evidence for ACC=1 in MUBUF encoding:
  - MUBUF bit [55]: ACC — "VDATA is Accumulation VGPR"
  - When ACC=1: buffer_load_dwordx2 a[0:1], ... (a[] = AccVGPR)
  - When ACC=0: buffer_load_dwordx2 v[0:1], ... (v[] = ArchVGPR, default)
  - "Accumulation VGPRs can be loaded directly from memory"
  - Same ACC bit exists in FLAT/GLOBAL/SCRATCH instructions

Implementation approach:
  - Use llvm.inline_asm with constraint "=a" (AccVGPR output) for buffer_load
  - Use llvm.inline_asm with constraint "a" (AccVGPR input) for buffer_store
  - The assembler sets ACC=1 automatically when it sees a[] register notation

This kernel computes D[16,16] = A[16,16] @ B[16,16] + C[16,16]:
  - A (left matrix):  loaded to AccVGPR via buffer_load ACC=1
  - B (right matrix): loaded to VGPR via standard buffer_load
  - C (accumulator):  zero-initialized (for simplicity, set C=0 → D = A @ B)
  - D (output):       stored from AccVGPR via buffer_store ACC=1

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
MODULE_NAME = "mfma_16x16x16_bf16_accvgpr_test"
KERNEL_NAME = "gemm_kernel_16x16x16_bf16_d_accvgpr"


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
# Inline-asm helpers: buffer load/store targeting AccVGPR (ACC=1)
# ─────────────────────────────────────────────────────────────────────────────

def accvgpr_buffer_load_dwordx2(rsrc, voffset_bytes):
    """buffer_load_dwordx2 → AccVGPR (ACC=1).

    Loads 8 bytes (= 4 × bf16 = 2 × i32) directly into AccVGPR.
    Constraint "=a" tells LLVM register allocator → AccVGPR destination.

    Returns: vector<2xi32> residing in a[N:N+1].
    """
    i32 = ir.IntegerType.get_signless(32)
    i32x2 = ir.VectorType.get([2], i32)
    return llvm.InlineAsmOp(
        res=i32x2,
        operands_=[_unwrap(voffset_bytes), _unwrap(rsrc)],
        asm_string="buffer_load_dwordx2 $0, $1, $2, 0 offen",
        constraints="=a,v,s",
        has_side_effects=True,
        is_align_stack=False,
    ).result


def accvgpr_buffer_store_dwordx4(data, rsrc, voffset_bytes):
    """buffer_store_dwordx4 from AccVGPR (ACC=1).

    Stores 16 bytes (= 4 × f32) from AccVGPR directly to global memory.
    Constraint "a" tells LLVM → source is AccVGPR.
    """
    i32 = ir.IntegerType.get_signless(32)
    # void return → use empty result list
    llvm.InlineAsmOp(
        res=[],
        operands_=[_unwrap(data), _unwrap(voffset_bytes), _unwrap(rsrc)],
        asm_string="buffer_store_dwordx4 $0, $1, $2, 0 offen",
        constraints="a,v,s",
        has_side_effects=True,
        is_align_stack=False,
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
        def gemm_kernel_16x16x16_bf16_d_accvgpr(
            self: flir.T.i64,
            A:   lambda: memref(M * K, T.bf16),   # A[m, k] row-major, K contiguous
            B_T: lambda: memref(N * K, T.bf16),   # B^T[n, k] transposed, K contiguous
            D:   lambda: memref(M * N, T.f32),     # D[m, n] row-major output
        ):
            i32 = T.i32
            vec4_i16 = T.i16x4
            vec4_f32 = T.f32x4

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

            # Load 8 bytes into AccVGPR: a[N:N+1] ← global memory
            a_i32x2 = accvgpr_buffer_load_dwordx2(a_rsrc, a_byte_off)
            # Bitcast vector<2xi32> → vector<4xi16> for MFMA input
            a_vec = llvm.bitcast(vec4_i16, _unwrap(a_i32x2))

            # ═══════════════════════════════════════════════════════════════
            # Load B → VGPR via standard buffer_load (ACC=0, default)
            # ═══════════════════════════════════════════════════════════════
            b_elem_off = buffer_ops.i32_add(
                buffer_ops.i32_mul(lm16, c16_i32),
                k_base_i32,
            )
            b_vec = buffer_ops.buffer_load(bt_rsrc, b_elem_off, vec_width=4,
                                           dtype=ir.IntegerType.get_signless(16))

            # ═══════════════════════════════════════════════════════════════
            # Zero accumulator (C = 0, so D = A @ B)
            # ═══════════════════════════════════════════════════════════════
            acc = arith.constant_vector(0.0, vec4_f32)

            # ═══════════════════════════════════════════════════════════════
            # MFMA: D = A * B + 0
            # ═══════════════════════════════════════════════════════════════
            result = rocdl.mfma_f32_16x16x16bf16_1k(
                vec4_f32, [a_vec, b_vec, acc, 0, 0, 0])

            # ═══════════════════════════════════════════════════════════════
            # Store D from AccVGPR via buffer_store_dwordx4 with ACC=1
            # ═══════════════════════════════════════════════════════════════
            # MFMA result is in AccVGPR by definition.
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
    print("MFMA 16x16x16 BF16 — Buffer Load to AccVGPR (ACC=1)")
    print("=" * 65)
    print(f"GPU arch: {get_hip_arch()}")
    print(f"D[{M},{N}] = A[{M},{K}] @ B[{K},{N}]  (bf16→f32)")
    print()
    print("Memory access pattern:")
    print("  A: buffer_load_dwordx2 → AccVGPR  (inline asm, ACC=1)")
    print("  B: buffer_load_dwordx2 → VGPR     (standard, ACC=0)")
    print("  D: buffer_store_dword  → global    (standard store)")
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
        print("PASS ✓ — AccVGPR buffer_load (ACC=1) + MFMA works correctly!")
    else:
        print(f"FAIL ✗: max diff = {max_diff}")
        print(f"D[0] = {D[0].tolist()}")
        print(f"E[0] = {expected[0].tolist()}")


if __name__ == "__main__":
    main()
