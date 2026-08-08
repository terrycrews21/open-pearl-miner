# Portable build for rented GPU nodes (Modal T4, Debian 12 / glibc 2.36)

A default `build_capi.sh` link on an Ubuntu 24.04 workstation produces a
`libp40cuda.so` that **cannot be dlopened on glibc 2.36 hosts**:

```
libp40cuda.so: /usr/lib/x86_64-linux-gnu/libstdc++.so.6:
    version `GLIBCXX_3.4.31' not found
    version `GLIBCXX_3.4.32' not found  (required by ./libp40cuda.so)
```

Modal T4 images ship Debian 12 with GLIBCXX up to 3.4.30 only. Two fixes:

1. **Static-link the C++ runtime** — `build_capi.sh` now passes
   `-Xcompiler=-static-libstdc++,-static-libgcc`, which removes every
   GLIBCXX/CXXABI requirement from the shared object.

2. **glibc 2.38 symbol redirect** — glibc 2.38+ headers redirect
   `strtol/strtoul/strtoll/strtoull` to `__isoc23_*`, stamping GLIBC_2.38.
   `csrc/capi/glibc_compat.cpp` defines those entry points locally and forwards
   to the always-present `strtol/strtoul` (`GLIBC_2.2.5`).

Result (`nm -D libp40cuda_t4.so`):

```
no GLIBCXX / CXXABI requirements
max GLIBC required: 2.36   (== the Modal T4 image)
```

Reproduce (== what `packaging/build_capi.sh` now does):

```bash
nvcc -shared -o libp40cuda_t4.so <sources...> -I csrc ... -I .deps/cutlass/include \
  -Xcompiler=-fPIC,-static-libstdc++,-static-libgcc \
  -std=c++20 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math \
  -gencode arch=compute_61,code=sm_61 -gencode arch=compute_75,code=sm_75 \
  -O3 -DNDEBUG -DP40_NO_TORCH -allow-unsupported-compiler
```

Also fixed: `packaging/build_capi.sh` listed `csrc/gemm/noise_gemm_turing.cu`
(which does not exist) and had no `sm_75` gencode; the dead `s_major==7` IMMA
noise-GEMM branch in `csrc/capi/p40_capi.cu` it implied is removed (DP4A noise
kernel is the fallback, ~0.2% of grid time). The `p40_pearl_pow_split` hot path
already dispatches true Turing WIDTH kernels via `launch_pearl_turing`.
