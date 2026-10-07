#include "libgf256.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned seed = 0x6d2b79f5u;
static unsigned rnd(void) {
  seed = seed * 1664525u + 1013904223u;
  return seed;
}
static gf256_t ref(gf256_t a, gf256_t b) {
  unsigned r = 0;
  for (unsigned i = 0; i < 8; i++) {
    if ((b >> i) & 1)
      r ^= (unsigned)a << i;
  }
  for (int i = 14; i >= 8; i--)
    if (r & (1u << i))
      r ^= 0x11d << (i - 8);
  return r;
}
static void die(const char *s) {
  fprintf(stderr, "FAIL: %s (seed=%u)\n", s, seed);
  exit(1);
}
static void same(const void *a, const void *b, size_t n, const char *s) {
  if (memcmp(a, b, n))
    die(s);
}
static void scalar(void) {
  for (unsigned a = 0; a < 256; a++) {
    if (gf256_inv(a) != (a ? gf256_pow(a, 254) : 0))
      die("inverse");
    for (unsigned b = 0; b < 256; b++) {
      if (gf256_mul(a, b) != ref(a, b) || gf256_mul(a, b) != gf256_mul(b, a))
        die("multiply");
    }
    if (gf256_pow(a, 0) != 1 || gf256_pow(a, 1) != a)
      die("power");
    if (a && gf256_pow(a, 255) != 1)
      die("order");
  }
}
static void vectors(gf256_ctx *x, gf256_ctx *r, size_t n) {
  gf256_t *a = malloc(n ? n : 1), *b = malloc(n ? n : 1),
          *d = malloc(n ? n : 1), *e = malloc(n ? n : 1);
  if (!a || !b || !d || !e)
    die("allocation");
  for (size_t i = 0; i < n; i++)
    a[i] = rnd(), b[i] = rnd(), d[i] = e[i] = rnd();
  gf256_t k = rnd();
  gf256_mul_vec(r, e, a, n, k);
  gf256_mul_vec(x, d, a, n, k);
  same(d, e, n, "mul_vec");
  memcpy(d, e, b ? n : 0); /* randomized new destination */
  for (size_t i = 0; i < n; i++)
    d[i] = e[i] = b[i];
  gf256_muladd_vec(r, e, a, n, k);
  gf256_muladd_vec(x, d, a, n, k);
  same(d, e, n, "muladd");
  gf256_mul_vec_vec(r, e, a, b, n);
  gf256_mul_vec_vec(x, d, a, b, n);
  same(d, e, n, "mulvv");
  for (size_t i = 0; i < n; i++)
    d[i] = e[i] = b[i];
  gf256_add_vec(r, e, a, n);
  gf256_add_vec(x, d, a, n);
  same(d, e, n, "add");
  free(a);
  free(b);
  free(d);
  free(e);
}
static void compound(gf256_ctx *x, gf256_ctx *r) {
  enum { N = 65539, C = 5, R = 4 };
  gf256_t *src[C], *out[R], *aout[R], *d = malloc(N), *e = malloc(N), coef[C],
                                      matrix[R * C];
  for (size_t j = 0; j < C; j++) {
    src[j] = malloc(N);
    coef[j] = rnd();
    for (size_t i = 0; i < N; i++)
      src[j][i] = rnd();
  }
  for (size_t i = 0; i < N; i++)
    d[i] = e[i] = rnd();
  gf256_madd(r, e, (const gf256_t *const *)src, coef, C, N);
  gf256_madd(x, d, (const gf256_t *const *)src, coef, C, N);
  same(d, e, N, "madd");
  for (size_t q = 0; q < R; q++) {
    out[q] = malloc(N);
    aout[q] = malloc(N);
    matrix[q * C] = rnd();
    for (size_t j = 1; j < C; j++)
      matrix[q * C + j] = rnd();
  }
  gf256_matmul(r, out, (const gf256_t *const *)src, matrix, R, C, N);
  gf256_matmul(x, aout, (const gf256_t *const *)src, matrix, R, C, N);
  for (size_t q = 0; q < R; q++)
    same(out[q], aout[q], N, "matmul");
  for (size_t j = 0; j < C; j++)
    free(src[j]);
  for (size_t q = 0; q < R; q++) {
    free(out[q]);
    free(aout[q]);
  }
  free(d);
  free(e);
}
int main(void) {
  scalar();
  gf256_ctx *base = gf256_init(GF256_BACKEND_BASE, 1);
  if (!base || gf256_init(GF256_BACKEND_BASE, 0))
    die("init");
  size_t sizes[] = {0, 1, 2, 31, 32, 33, 63, 64, 65, 257, 1025, 65536};
  for (int b = GF256_BACKEND_BASE; b <= GF256_BACKEND_GFNI_AVX512; b++) {
    if (!gf256_backend_available((gf256_backend)b)) {
      printf("skip %d\n", b);
      continue;
    }
    gf256_ctx *x = gf256_init((gf256_backend)b, 4);
    if (!x)
      die("backend init");
    for (size_t i = 0; i < sizeof sizes / sizeof *sizes; i++)
      vectors(x, base, sizes[i]);
    compound(x, base);
    gf256_free(x);
  }
  gf256_ctx *automatic = gf256_init(GF256_BACKEND_AUTO, 1);
  if (!automatic)
    die("auto");
  printf("ok: auto=%s\n", gf256_get_backend_name(automatic));
  gf256_free(automatic);
  gf256_free(base);
  return 0;
}
