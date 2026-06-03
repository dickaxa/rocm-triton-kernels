#!/usr/bin/env python3
import triton, triton.language as tl, torch, time

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(c_ptr + offsets, tl.load(a_ptr + offsets, mask=mask) + tl.load(b_ptr + offsets, mask=mask), mask=mask)

def vector_add(a, b):
    c = torch.empty_like(a)
    vector_add_kernel[(triton.cdiv(a.numel(), 1024),)](a, b, c, a.numel(), 1024)
    return c

if __name__ == "__main__":
    N = 1 << 20
    a, b = torch.randn(N, device="cuda"), torch.randn(N, device="cuda")
    t0 = time.perf_counter()
    c = vector_add(a, b)
    dt = time.perf_counter() - t0
    assert torch.allclose(c, a + b), "Mismatch!"
    print(f"Vector add PASSED: {N} elements in {dt*1000:.2f}ms")
