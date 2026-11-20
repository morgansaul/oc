# secp256k1 X-only prefix search (CUDA)

This project searches scalar values `k` in the inclusive interval `[L,U]` on secp256k1. For each point `kG`, it extracts the top 80 bits of the affine X coordinate and minimizes the exact integer distance to an 80-bit target prefix.

Default objective:

- `L = 1`
- `U = 2^80 - 1`
- target prefix = `03000000000000000000`

The distance is

```text
abs((X(kG) >> 176) - target_prefix)
```

## Algorithm

The GPU distributes the scalar interval across independent residue lanes. With stride `S = threads`, lane `t` visits

```text
L+t, L+t+S, L+t+2S, ...
```

Each lane is initialized from one scalar multiplication and one subtraction by `S*G`. Thereafter it advances using the exact secp256k1 X-only differential recurrence

```text
x_next = 2 * (x_h*x_cur*(x_cur+x_h) + 14) / (x_cur-x_h)^2 - x_prev  (mod p)
```

where `p = 2^256 - 2^32 - 977` and `x_h = X(SG)`. The denominator inverses are computed with warp-level batch inversion: one field inversion is shared across 32 lanes. The complete 256-bit X value is retained for final verification.

The source uses 8 little-endian 32-bit limbs for field elements and Jacobian coordinates for initialization. `core_test.cpp` independently exercises the host/device arithmetic routines and the recurrence.

## Build on the RTX 5090 machine

Requires an NVIDIA CUDA Toolkit that supports `sm_120` (Blackwell) and a C++17 host compiler.

```bash
make clean
make -j CUDA_ARCH=sm_120
```

## Checks

Run the built-in GPU self-test. It scans the entire small interval `[1,10000]` and checks the expected result against the known reference vector, then independently verifies the returned point with full scalar multiplication on the host:

```bash
./xonly_cuda --selftest
```

Optional host arithmetic tests (requires Boost.Multiprecision headers):

```bash
make test-core
```

## Search options

```text
./xonly_cuda [--L N] [--U N] [--target 20_HEX]
             [--threads N] [--chunk N] [--max-rounds N] [--quiet]
```

- `--L`, `--U`: inclusive scalar bounds; decimal or `0x` hexadecimal. The program accepts `1 <= L <= U < 2^80`.
- `--target`: exactly 20 hexadecimal digits; default `03000000000000000000`.
- `--threads`: number of residue lanes and the scalar stride. Must be a multiple of 128 from 128 through 1,048,576. Default `65536`.
- `--chunk`: recurrence rounds per CUDA launch. Default `256`.
- `--max-rounds`: optional limit on recurrence rounds after the initial two candidates per lane. When used, the result is marked partial unless a zero-distance match is found.
- `--quiet`: suppress progress lines.

Example limited run:

```bash
./xonly_cuda --L 1 --U 0xffffffffffffffffffff --threads 65536 --chunk 256 --max-rounds 1000
```

## Correctness status and limits

A returned distance of zero is globally optimal immediately. Otherwise, the result is a proven global minimum only when the entire requested interval is reported as covered. A limited run reports only the best candidate among the scalars it actually scanned.

The full `2^80-1` scalar interval is an enormous search. This program does not claim a shortcut that makes exhaustive coverage free; it is an exact search engine using the differential recurrence to reduce per-candidate work. The CUDA source must be compiled and run on the target CUDA/GPU system; this development environment has no `nvcc`, so GPU compilation and execution have not been represented as completed here.

An additional independent check compares randomized 80-bit scalar multiplication
and the X-only recurrence directly against OpenSSL's secp256k1 implementation
(requires OpenSSL development headers and `libcrypto`):

```bash
make test-openssl
```

## Preparation validation

The host arithmetic suite passes 200,000 randomized field add/subtract/multiply
comparisons, inversion checks, Jacobian-point checks, recurrence checks,
batch-inversion checks, 100,000 prefix-distance checks, and a complete
`[1,10000]` lane scan (best scalar `9325`). The OpenSSL cross-check passes 120
randomized 80-bit cases. The `.cu` translation unit also passed host-side and
device-side Clang CUDA syntax checks in the preparation environment.

The preparation environment does not contain `nvcc` or an NVIDIA GPU, so an
actual NVCC build and RTX 5090 runtime self-test have **not** been performed
here. Do that on the target machine with the commands above before starting a
large search.
