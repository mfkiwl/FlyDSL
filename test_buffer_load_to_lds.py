#!/usr/bin/env python3
"""Test kernel: Buffer Load to LDS (BUFFER_LOAD_DWORD with LDS flag on CDNA3)

Demonstrates the MUBUF "buffer load to LDS" instruction from ISA section 9.1.9:
  BUFFER_LOAD_{ubyte,sbyte,ushort,sshort,dword,format_x} with LDS=1
  reads data from a memory buffer directly into LDS without passing through VGPRs.

Kernel logic:
  1. Each thread uses raw_ptr_buffer_load_lds to DMA one f32 (4 bytes)
     from global memory (input A) directly into LDS.
  2. s_waitcnt + barrier to ensure all LDS writes are visible.
  3. Each thread reads its value from LDS, doubles it, and writes to output B
     via buffer_store.

Expected: B[i] = A[i] * 2
"""

import functools
import torch
import flydsl
from flydsl.dialects.ext import flir, arith, gpu, buffer_ops, rocdl
from flydsl.lang.ir.types import T, memref
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils import SmemAllocator
from _mlir import ir
from _mlir.dialects import memref as memref_ops

N = 256  # number of elements = number of threads
MODULE_NAME = "buffer_load_to_lds_test"
KERNEL_NAME = "kernel_main"


@functools.lru_cache(maxsize=32)
def compile_kernel():
    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    class _Mod(flir.MlirModule):
        GPU_MODULE_NAME = MODULE_NAME
        GPU_MODULE_TARGETS = [
            f'#rocdl.target<chip = "{gpu_arch}", abi = "500", features = "+sramecc,+xnack">'
        ]

        def init_gpu_module(self):
            # Allocate LDS: N x f32 = N*4 bytes
            _state["lds_decl"] = allocator.allocate_array(T.f32, N)
            allocator.finalize()

        @flir.kernel
        def kernel_main(
            self: flir.T.i64,
            A: lambda: memref(N, T.f32),
            B: lambda: memref(N, T.f32),
        ):
            tid = gpu.thread_id("x")
            tid_i32 = buffer_ops.index_cast_to_i32(tid)

            # ── LDS setup ──
            base_ptr = allocator.get_base()
            lds_ptr_decl = _state["lds_decl"](base_ptr)
            lds_buf = lds_ptr_decl.get()  # memref<N x f32, addrspace 3>

            # ── Buffer resource descriptor for input A ──
            # create_buffer_resource builds the 128-bit V# descriptor
            # (base_ptr, stride=0, num_records, flags) needed by MUBUF instructions.
            a_rsrc = buffer_ops.create_buffer_resource(A, num_records_bytes=N * 4)

            # ── Per-thread byte offset into global buffer ──
            four_i32 = arith.constant(4, type=T.i32)
            voffset = buffer_ops.i32_mul(tid_i32, four_i32)  # tid * 4 bytes

            # ── LDS destination pointer (address space 3) ──
            # raw_ptr_buffer_load_lds needs an LDS pointer.
            # We compute: lds_base_addr + tid * 4
            lds_base_idx = memref_ops.extract_aligned_pointer_as_index(lds_buf)
            four_idx = arith.index(4)
            tid_offset_idx = tid * four_idx
            lds_with_offset = lds_base_idx + tid_offset_idx
            lds_ptr = buffer_ops.create_llvm_ptr(lds_with_offset, address_space=3)

            # ── BUFFER_LOAD_DWORD to LDS (ISA 9.1.9) ──
            # raw_ptr_buffer_load_lds(rsrc, lds_ptr, size, voffset, soffset, offset, aux)
            #   rsrc:    buffer resource descriptor (V#)
            #   lds_ptr: LDS destination pointer (!llvm.ptr<3>)
            #   size:    transfer size in bytes (4 for dword)
            #   voffset: per-thread byte offset into the buffer (VGPR)
            #   soffset: scalar byte offset (SGPR), usually 0
            #   offset:  immediate byte offset, usually 0
            #   aux:     cache/GLC flags (0 = default)
            size_i32 = arith.constant(4, type=T.i32)
            soffset = arith.constant(0, type=T.i32)
            offset_imm = arith.constant(0, type=T.i32)
            aux = arith.constant(0, type=T.i32)

            rocdl.raw_ptr_buffer_load_lds(
                a_rsrc,
                arith.unwrap(lds_ptr),
                arith.unwrap(size_i32),
                arith.unwrap(voffset),
                arith.unwrap(soffset),
                arith.unwrap(offset_imm),
                arith.unwrap(aux),
            )

            # ── Synchronize: wait for buffer load to LDS to complete ──
            rocdl.s_waitcnt(0)
            gpu.barrier()

            # ── Read from LDS, double the value, store to output B ──
            val = memref_ops.load(lds_buf, [buffer_ops._unwrap_value(tid)])
            two = arith.constant(2.0, type=ir.F32Type.get())
            result = buffer_ops._unwrap_value(val) * two

            # Store result to B via buffer store
            rsrc_b = buffer_ops.create_buffer_resource(B, num_records_bytes=N * 4)
            buffer_ops.buffer_store(result, rsrc_b, tid_i32)

        # Host launcher
        @flir.jit
        def __call__(
            self: flir.T.i64,
            A: lambda: memref(N, T.f32),
            B: lambda: memref(N, T.f32),
        ):
            c1 = arith.constant(1, index=True)
            bdx = arith.constant(N, index=True)
            flir.gpu_ext.LaunchFuncOp(
                [MODULE_NAME, KERNEL_NAME],
                grid_size=(c1, c1, c1),
                block_size=(bdx, c1, c1),
                kernel_operands=[A, B],
            )

    m = _Mod()
    return flydsl.compile(m)


def main():
    print(f"=== Buffer Load to LDS Test (N={N}) ===")
    print(f"GPU arch: {get_hip_arch()}")
    print(f"Compiling kernel...")
    compiled = compile_kernel()

    A = torch.arange(N, dtype=torch.float32, device='cuda')
    B = torch.zeros(N, dtype=torch.float32, device='cuda')

    print(f"Input  A[:8] = {A[:8].tolist()}")
    print(f"Expect B[:8] = {(A[:8] * 2).tolist()}")

    print("Running kernel...")
    compiled(A, B)
    torch.cuda.synchronize()

    print(f"Output B[:8] = {B[:8].tolist()}")

    expected = A * 2
    if torch.allclose(B, expected):
        print("PASS - buffer_load_to_lds works correctly!")
    else:
        diff = (B - expected).abs().max().item()
        print(f"FAIL: max diff = {diff}")
        print(f"B[:16] = {B[:16].tolist()}")


if __name__ == "__main__":
    main()
