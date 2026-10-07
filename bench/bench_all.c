/*
 * bench_all.c -- exercise every backend libgf256 supports on this machine,
 * across every operation, and (for a couple of representative ops) across
 * thread counts, printing a plain-text throughput table.
 *
 * Usage: bench_all [size_bytes] [max_threads]
 *   size_bytes   vector length used for every measurement (default 16 MiB)
 *   max_threads  highest thread count to test for the scaling section
 *                (default: number of online CPUs)
 */
#include "libgf256.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

/* Run `body` repeatedly for at least MIN_SECONDS wall-clock time (never
 * fewer than 3 iterations), and return achieved GiB/s given `bytes` moved
 * per iteration. Using a time-based iteration count keeps fast backends
 * (GFNI) and slow ones (base) both landing on a stable measurement instead
 * of a fixed iteration count being too short for one and too long for the
 * other. */
#define MIN_SECONDS 0.15

typedef void (*bench_fn)(void *ctx);

static double time_it(bench_fn fn, void *ctx, double bytes_per_call) {
  int iters = 3;
  double elapsed;
  for (;;) {
    double t0 = now();
    for (int i = 0; i < iters; i++) fn(ctx);
    elapsed = now() - t0;
    if (elapsed >= MIN_SECONDS || iters > 1 << 20) break;
    iters *= 2;
  }
  return (bytes_per_call * iters) / elapsed / (1ULL << 30);
}

/* ---- per-operation harnesses ------------------------------------------ */

typedef struct { gf256_ctx *c; gf256_t *d, *a, *b; size_t n; } vec_ctx;
static void run_mul_vec(void *v)     { vec_ctx *x = v; gf256_mul_vec(x->c, x->d, x->a, x->n, 0x17); }
static void run_muladd_vec(void *v)  { vec_ctx *x = v; gf256_muladd_vec(x->c, x->d, x->a, x->n, 0x17); }
static void run_mul_vec_vec(void *v) { vec_ctx *x = v; gf256_mul_vec_vec(x->c, x->d, x->a, x->b, x->n); }
static void run_add_vec(void *v)     { vec_ctx *x = v; gf256_add_vec(x->c, x->d, x->a, x->n); }

#define MADD_SOURCES 8
typedef struct { gf256_ctx *c; gf256_t *d; const gf256_t *src[MADD_SOURCES]; gf256_t coef[MADD_SOURCES]; size_t n; } madd_ctx;
static void run_madd(void *v) { madd_ctx *x = v; gf256_madd(x->c, x->d, x->src, x->coef, MADD_SOURCES, x->n); }

#define MM_ROWS 10
#define MM_COLS 6
typedef struct { gf256_ctx *c; gf256_t *dst[MM_ROWS]; const gf256_t *src[MM_COLS]; gf256_t matrix[MM_ROWS*MM_COLS]; size_t n; } mm_ctx;
static void run_matmul(void *v) { mm_ctx *x = v; gf256_matmul(x->c, x->dst, x->src, x->matrix, MM_ROWS, MM_COLS, x->n); }

/* ---- driver ------------------------------------------------------------ */

static const char *backend_label(gf256_backend b) {
  switch (b) {
    case GF256_BACKEND_BASE: return "base";
    case GF256_BACKEND_GFNI_AVX256: return "gfni-avx256";
    case GF256_BACKEND_GFNI_AVX512: return "gfni-avx512";
    default: return "?";
  }
}

static gf256_t *fill_rand(size_t n) {
  gf256_t *p = malloc(n ? n : 1);
  for (size_t i = 0; i < n; i++) p[i] = (gf256_t)(i * 2654435761u >> 24);
  return p;
}

int main(int argc, char **argv) {
  size_t size = argc > 1 ? strtoull(argv[1], NULL, 0) : (16u << 20);
  long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
  if (ncpu < 1) ncpu = 1;
  long max_threads = argc > 2 ? strtol(argv[2], NULL, 0) : ncpu;

  printf("libgf256 benchmark  (vector size = %zu bytes, %ld CPU(s) online)\n\n", size, ncpu);

  gf256_backend backends[] = { GF256_BACKEND_BASE, GF256_BACKEND_GFNI_AVX256, GF256_BACKEND_GFNI_AVX512 };
  const char *op_names[] = { "mul_vec", "muladd_vec", "mul_vec_vec", "add_vec", "madd(8 src)", "matmul(10x6)" };

  /* --- single-thread throughput across every op, every available backend --- */
  printf("=== single-thread throughput, all ops ===\n");
  printf("%-14s %-14s %10s\n", "backend", "operation", "GiB/s");
  for (size_t bi = 0; bi < sizeof backends / sizeof *backends; bi++) {
    gf256_backend b = backends[bi];
    if (!gf256_backend_available(b)) { printf("%-14s (unavailable on this CPU)\n", backend_label(b)); continue; }
    gf256_ctx *c = gf256_init(b, 1);
    if (!c) { printf("%-14s init failed\n", backend_label(b)); continue; }

    vec_ctx vx = { c, fill_rand(size), fill_rand(size), fill_rand(size), size };
    double r;
    r = time_it(run_mul_vec, &vx, (double)size);      printf("%-14s %-14s %10.2f\n", backend_label(b), op_names[0], r);
    r = time_it(run_muladd_vec, &vx, (double)size);    printf("%-14s %-14s %10.2f\n", backend_label(b), op_names[1], r);
    r = time_it(run_mul_vec_vec, &vx, (double)size);   printf("%-14s %-14s %10.2f\n", backend_label(b), op_names[2], r);
    r = time_it(run_add_vec, &vx, (double)size);       printf("%-14s %-14s %10.2f\n", backend_label(b), op_names[3], r);
    free(vx.d); free(vx.a); free(vx.b);

    madd_ctx mx; mx.c = c; mx.n = size; mx.d = fill_rand(size);
    for (int j = 0; j < MADD_SOURCES; j++) { mx.src[j] = fill_rand(size); mx.coef[j] = (gf256_t)(j * 37 + 1); }
    r = time_it(run_madd, &mx, (double)size * MADD_SOURCES);
    printf("%-14s %-14s %10.2f\n", backend_label(b), op_names[4], r);
    free(mx.d); for (int j = 0; j < MADD_SOURCES; j++) free((void *)mx.src[j]);

    mm_ctx mm; mm.c = c; mm.n = size;
    for (int r_ = 0; r_ < MM_ROWS; r_++) mm.dst[r_] = fill_rand(size);
    for (int cc = 0; cc < MM_COLS; cc++) mm.src[cc] = fill_rand(size);
    for (int k = 0; k < MM_ROWS*MM_COLS; k++) mm.matrix[k] = (gf256_t)(k * 53 + 3);
    r = time_it(run_matmul, &mm, (double)size * MM_ROWS * MM_COLS);
    printf("%-14s %-14s %10.2f\n", backend_label(b), op_names[5], r);
    for (int r_ = 0; r_ < MM_ROWS; r_++) free(mm.dst[r_]);
    for (int cc = 0; cc < MM_COLS; cc++) free((void *)mm.src[cc]);

    gf256_free(c);
    printf("\n");
  }

  /* --- thread scaling for the fastest available backend --- */
  gf256_backend best = GF256_BACKEND_BASE;
  if (gf256_backend_available(GF256_BACKEND_GFNI_AVX256)) best = GF256_BACKEND_GFNI_AVX256;
  if (gf256_backend_available(GF256_BACKEND_GFNI_AVX512)) best = GF256_BACKEND_GFNI_AVX512;

  printf("=== thread scaling, backend=%s, muladd_vec, size=%zu ===\n", backend_label(best), size);
  printf("%10s %12s %10s\n", "threads", "GiB/s", "speedup");
  double base_gibps = 0;
  for (long t = 1; t <= max_threads; t = (t == 1 ? 2 : t * 2)) {
    if (t > max_threads) break;
    gf256_ctx *c = gf256_init(best, (unsigned)t);
    if (!c) { printf("%10ld init failed\n", t); continue; }
    vec_ctx vx = { c, fill_rand(size), fill_rand(size), NULL, size };
    double r = time_it(run_muladd_vec, &vx, (double)size);
    if (t == 1) base_gibps = r;
    printf("%10ld %12.2f %9.2fx\n", t, r, base_gibps > 0 ? r / base_gibps : 1.0);
    free(vx.d); free(vx.a);
    gf256_free(c);
    if (t == max_threads) break;
  }

  /* --- thread scaling for matmul (10x6, few rows -- the case the
   *     parallel-by-element fix specifically targets) --- */
  printf("\n=== thread scaling, backend=%s, matmul(10x6), size=%zu ===\n", backend_label(best), size);
  printf("%10s %12s %10s\n", "threads", "GiB/s", "speedup");
  base_gibps = 0;
  for (long t = 1; t <= max_threads; t = (t == 1 ? 2 : t * 2)) {
    if (t > max_threads) break;
    gf256_ctx *c = gf256_init(best, (unsigned)t);
    if (!c) { printf("%10ld init failed\n", t); continue; }
    mm_ctx mm; mm.c = c; mm.n = size;
    for (int r_ = 0; r_ < MM_ROWS; r_++) mm.dst[r_] = fill_rand(size);
    for (int cc = 0; cc < MM_COLS; cc++) mm.src[cc] = fill_rand(size);
    for (int k = 0; k < MM_ROWS*MM_COLS; k++) mm.matrix[k] = (gf256_t)(k * 53 + 3);
    double r = time_it(run_matmul, &mm, (double)size * MM_ROWS * MM_COLS);
    if (t == 1) base_gibps = r;
    printf("%10ld %12.2f %9.2fx\n", t, r, base_gibps > 0 ? r / base_gibps : 1.0);
    for (int r_ = 0; r_ < MM_ROWS; r_++) free(mm.dst[r_]);
    for (int cc = 0; cc < MM_COLS; cc++) free((void *)mm.src[cc]);
    gf256_free(c);
    if (t == max_threads) break;
  }

  return 0;
}
