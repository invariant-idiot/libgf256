#ifndef GF256_INTERNAL_H
#define GF256_INTERNAL_H

#include "libgf256.h"

#include <pthread.h>

/*
 * Backend operation table.
 *
 * Each backend (scalar, AVX2, AVX-512, etc.) provides implementations of
 * these operations. The public API can therefore select a backend once and
 * dispatch operations through this table without knowing how they are
 * implemented.
 */
typedef struct gf256_ops {
  /*
   * Multiply every element of `src` by `scalar` and store the result in
   * `dst`.
   *
   * dst[i] = src[i] * scalar
   */
  void (*mul)(gf256_t *dst, const gf256_t *src, size_t count, gf256_t scalar);

  /*
   * Multiply every element of `src` by `scalar` and XOR the result into
   * `dst`.
   *
   * dst[i] ^= src[i] * scalar
   */
  void (*muladd)(gf256_t *dst, const gf256_t *src, size_t count,
                 gf256_t scalar);

  /*
   * Multiply corresponding elements of two vectors and store the results
   * in `dst`.
   *
   * dst[i] = lhs[i] * rhs[i]
   */
  void (*mulvv)(gf256_t *dst, const gf256_t *lhs, const gf256_t *rhs,
                size_t count);

  /*
   * XOR `src` into `dst`.
   *
   * dst[i] ^= src[i]
   */
  void (*add)(gf256_t *dst, const gf256_t *src, size_t count);

  /*
   * Matrix/vector accumulation operation.
   *
   * The exact layout is defined by the public API, but conceptually this
   * performs multiple scalar multiplications and accumulates their results
   * into the destination vectors.
   */
  void (*madd)(gf256_t *dst, const gf256_t *const *src,
               const gf256_t *coefficients, size_t vector_count,
               size_t element_count);

  /*
   * Matrix multiplication over GF(256), processing only the element range
   * [first_element, last_element).
   *
   * The element range is intentionally split rather than the row range.
   * This allows matrices with relatively few rows but long vectors to be
   * parallelized efficiently using the same job dispatcher as the other
   * GF(256) operations.
   */
  void (*matmul)(gf256_t *const *dst, const gf256_t *const *src,
                 const gf256_t *matrix, size_t rows, size_t cols,
                 size_t first_element, size_t last_element);
} gf256_ops;

/*
 * A unit of work executed by the GF(256) thread pool.
 *
 * `start` and `end` describe the portion of the input assigned to the
 * worker. The meaning of the range depends on the operation being executed.
 */
typedef void (*gf256_job_fn)(void *job_arg, size_t start, size_t end);

/*
 * State associated with a single worker thread.
 */
typedef struct gf256_worker {
  pthread_t thread;
  struct gf256_ctx *ctx;
  unsigned id;
} gf256_worker;

/*
 * Internal GF(256) execution context.
 *
 * The context owns the selected backend and, when multithreading is enabled,
 * the worker threads used to execute parallel operations.
 */
struct gf256_ctx {
  /* Currently selected implementation/backend. */
  gf256_backend backend;

  /* Operations provided by the selected backend. */
  const gf256_ops *ops;

  /* Number of worker threads available for parallel operations. */
  unsigned threads;

  /*
   * Synchronization primitives used by the worker pool.
   *
   * mutex protects the shared job state.
   * wake signals workers that a new job is available.
   * done signals the submitting thread that the current job is complete.
   */
  pthread_mutex_t mutex;
  pthread_cond_t wake;
  pthread_cond_t done;

  /* Worker thread state. */
  gf256_worker *workers;

  /*
   * Number of workers currently participating in the active job and
   * number of workers that have finished it.
   */
  unsigned active;
  unsigned completed;

  /* Set when the worker pool is shutting down. */
  int stop;

  /*
   * Current job being executed by the worker pool.
   *
   * job      - function executed by each worker
   * job_arg  - caller-provided argument passed to the job
   * job_n    - total number of elements in the job
   */
  gf256_job_fn job;
  void *job_arg;
  size_t job_n;
};

/*
 * Backend operation tables.
 *
 * These are defined by the individual backend implementations.
 */
extern const gf256_ops gf256_base_ops;
extern const gf256_ops gf256_avx256_ops;
extern const gf256_ops gf256_avx512_ops;

/*
 * Execute a GF(256) operation in parallel using the worker pool associated
 * with `ctx`.
 *
 * The operation is divided into ranges and each worker invokes `job` for
 * its assigned range.
 */
void gf256_parallel(gf256_ctx *ctx, size_t element_count, gf256_job_fn job,
                    void *job_arg);

/*
 * Return non-zero when the requested backend is supported by the current
 * CPU, otherwise return zero.
 */
int gf256_cpu_available(gf256_backend backend);

/*
 * Multiply two GF(256) elements using the scalar implementation.
 *
 * This function is used internally when a hardware-accelerated backend is
 * unavailable or when a scalar operation is required.
 */
gf256_t gf256_mul_scalar(gf256_t lhs, gf256_t rhs);

#endif /* GF256_INTERNAL_H */
