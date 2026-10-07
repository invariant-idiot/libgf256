#include "gf256_internal.h"

/*
 * Check whether the requested backend is supported by the current CPU.
 *
 * The base backend is always available. GFNI backends require x86 CPU
 * feature detection, which is supported here through GCC/Clang builtins.
 */
int gf256_cpu_available(gf256_backend backend) {
  if (backend == GF256_BACKEND_AUTO || backend == GF256_BACKEND_BASE)
    return 1;

#if (defined(__x86_64__) || defined(__i386__)) &&                              \
    (defined(__GNUC__) || defined(__clang__))

  /*
   * Initialize CPU feature detection before using __builtin_cpu_supports().
   */
  __builtin_cpu_init();

  /*
   * AVX256 GFNI requires both AVX2 and GFNI support.
   */
  if (backend == GF256_BACKEND_GFNI_AVX256)
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("gfni");

  /*
   * AVX512 GFNI requires AVX-512 Foundation, byte/word operations,
   * and GFNI support.
   */
  if (backend == GF256_BACKEND_GFNI_AVX512)
    return __builtin_cpu_supports("avx512f") &&
           __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("gfni");

#else

  /*
   * Non-x86 platforms, or compilers without the required CPU feature
   * builtins, cannot use the architecture-specific backends.
   */
  (void)backend;

#endif

  return 0;
}
