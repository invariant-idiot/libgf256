#include "gf256_internal.h"

#include <stdlib.h>

int gf256_threads_init(gf256_ctx *);
void gf256_threads_free(gf256_ctx *);

/*
 * Scalar GF(256) operations.
 */

gf256_t gf256_mul(gf256_t a, gf256_t b) { return gf256_mul_scalar(a, b); }

gf256_t gf256_pow(gf256_t base, uint32_t exponent) {
  gf256_t result = 1;

  /*
   * Binary exponentiation reduces the number of multiplications from
   * O(exponent) to O(log(exponent)).
   */
  while (exponent) {
    if (exponent & 1)
      result = gf256_mul(result, base);

    base = gf256_mul(base, base);
    exponent >>= 1;
  }

  return result;
}

gf256_t gf256_inv(gf256_t value) {
  /*
   * For every non-zero element a in GF(256), a^254 is its
   * multiplicative inverse.
   */
  return value ? gf256_pow(value, 254) : 0;
}

int gf256_backend_available(gf256_backend backend) {
  return gf256_cpu_available(backend);
}

/*
 * Return the operation table associated with a backend.
 *
 * The caller is responsible for ensuring that the backend is valid.
 */
static const gf256_ops *get_backend_operations(gf256_backend backend) {
  if (backend == GF256_BACKEND_BASE)
    return &gf256_base_ops;

  if (backend == GF256_BACKEND_GFNI_AVX256)
    return &gf256_avx256_ops;

  return &gf256_avx512_ops;
}

/*
 * Context initialization and configuration.
 */

gf256_ctx *gf256_init(gf256_backend backend, unsigned thread_count) {
  if (!thread_count || backend < GF256_BACKEND_AUTO ||
      backend > GF256_BACKEND_GFNI_AVX512)
    return NULL;

  /*
   * When automatic selection is requested, prefer the most capable
   * implementation supported by the current CPU.
   */
  if (backend == GF256_BACKEND_AUTO)
    backend = gf256_cpu_available(GF256_BACKEND_GFNI_AVX512)
                  ? GF256_BACKEND_GFNI_AVX512
              : gf256_cpu_available(GF256_BACKEND_GFNI_AVX256)
                  ? GF256_BACKEND_GFNI_AVX256
                  : GF256_BACKEND_BASE;

  if (!gf256_cpu_available(backend))
    return NULL;

  gf256_ctx *context = calloc(1, sizeof(*context));

  if (!context)
    return NULL;

  context->backend = backend;
  context->ops = get_backend_operations(backend);
  context->threads = thread_count;

  if (!gf256_threads_init(context)) {
    free(context);
    return NULL;
  }

  return context;
}

void gf256_free(gf256_ctx *context) {
  if (context) {
    gf256_threads_free(context);
    free(context);
  }
}

gf256_backend gf256_get_backend(const gf256_ctx *context) {
  return context ? context->backend : GF256_BACKEND_AUTO;
}

const char *gf256_get_backend_name(const gf256_ctx *context) {
  switch (gf256_get_backend(context)) {
  case GF256_BACKEND_BASE:
    return "base";

  case GF256_BACKEND_GFNI_AVX256:
    return "gfni-avx256";

  case GF256_BACKEND_GFNI_AVX512:
    return "gfni-avx512";

  default:
    return "auto";
  }
}

unsigned gf256_get_threads(const gf256_ctx *context) {
  return context ? context->threads : 0;
}

/*
 * Parallel vector operations.
 *
 * gf256_parallel() invokes the worker callback with a half-open element
 * range [start_index, end_index).
 */

typedef struct {
  gf256_ctx *context;
  gf256_t *destination;
  const gf256_t *source_a;
  const gf256_t *source_b;
  gf256_t multiplier;
} vector_operation_job;

/*
 * Process a vector multiplication:
 *
 *     destination[i] = source_a[i] * multiplier
 */
static void vector_multiply(void *job_data, size_t start_index,
                            size_t end_index) {
  vector_operation_job *job = job_data;

  job->context->ops->mul(job->destination + start_index,
                         job->source_a + start_index, end_index - start_index,
                         job->multiplier);
}

/*
 * Process a vector multiply-add:
 *
 *     destination[i] += source_a[i] * multiplier
 */
static void vector_multiply_add(void *job_data, size_t start_index,
                                size_t end_index) {
  vector_operation_job *job = job_data;

  job->context->ops->muladd(job->destination + start_index,
                            job->source_a + start_index,
                            end_index - start_index, job->multiplier);
}

/*
 * Process element-wise vector multiplication:
 *
 *     destination[i] = source_a[i] * source_b[i]
 */
static void vector_element_multiply(void *job_data, size_t start_index,
                                    size_t end_index) {
  vector_operation_job *job = job_data;

  job->context->ops->mulvv(
      job->destination + start_index, job->source_a + start_index,
      job->source_b + start_index, end_index - start_index);
}

/*
 * Process element-wise vector addition:
 *
 *     destination[i] += source_a[i]
 */
static void vector_add(void *job_data, size_t start_index, size_t end_index) {
  vector_operation_job *job = job_data;

  job->context->ops->add(job->destination + start_index,
                         job->source_a + start_index, end_index - start_index);
}

void gf256_mul_vec(gf256_ctx *context, gf256_t *destination,
                   const gf256_t *source, size_t element_count,
                   gf256_t multiplier) {
  vector_operation_job job = {context, destination, source, NULL, multiplier};

  gf256_parallel(context, element_count, vector_multiply, &job);
}

void gf256_muladd_vec(gf256_ctx *context, gf256_t *destination,
                      const gf256_t *source, size_t element_count,
                      gf256_t multiplier) {
  vector_operation_job job = {context, destination, source, NULL, multiplier};

  gf256_parallel(context, element_count, vector_multiply_add, &job);
}

void gf256_mul_vec_vec(gf256_ctx *context, gf256_t *destination,
                       const gf256_t *source_a, const gf256_t *source_b,
                       size_t element_count) {
  vector_operation_job job = {context, destination, source_a, source_b, 0};

  gf256_parallel(context, element_count, vector_element_multiply, &job);
}

void gf256_add_vec(gf256_ctx *context, gf256_t *destination,
                   const gf256_t *source, size_t element_count) {
  vector_operation_job job = {context, destination, source, NULL, 0};

  gf256_parallel(context, element_count, vector_add, &job);
}

/*
 * Multiple-vector multiply/add.
 */

typedef struct {
  gf256_ctx *context;
  gf256_t *destination;
  const gf256_t *const *sources;
  const gf256_t *multipliers;
  size_t source_count;
} multiply_add_job;

/*
 * Apply all source/multiplier pairs to the assigned element range:
 *
 *     destination += sources[0] * multipliers[0]
 *                   + sources[1] * multipliers[1]
 *                   + ...
 */
static void run_multiple_vector_multiply_add(void *job_data, size_t start_index,
                                             size_t end_index) {
  multiply_add_job *job = job_data;

  for (size_t source_index = 0; source_index < job->source_count;
       source_index++) {

    job->context->ops->muladd(job->destination + start_index,
                              job->sources[source_index] + start_index,
                              end_index - start_index,
                              job->multipliers[source_index]);
  }
}

void gf256_madd(gf256_ctx *context, gf256_t *destination,
                const gf256_t *const *sources, const gf256_t *multipliers,
                size_t source_count, size_t element_count) {
  if (!source_count || !element_count)
    return;

  multiply_add_job job = {context, destination, sources, multipliers,
                          source_count};

  gf256_parallel(context, element_count, run_multiple_vector_multiply_add,
                 &job);
}

/*
 * Matrix/vector multiplication.
 *
 * Work is divided over the vector element range rather than over rows.
 *
 * Typical erasure-coding workloads have relatively few rows but very long
 * vectors. Splitting by rows would therefore give gf256_parallel() a small
 * workload size and can prevent the worker pool from being used effectively.
 *
 * Splitting by vector element count keeps the parallelization consistent
 * with gf256_madd() and the other vector operations.
 */

typedef struct {
  gf256_ctx *context;
  gf256_t *const *destination;
  const gf256_t *const *sources;
  const gf256_t *matrix;
  size_t row_count;
  size_t column_count;
} matrix_multiply_job;

static void matrix_multiply(void *job_data, size_t start_index,
                            size_t end_index) {
  matrix_multiply_job *job = job_data;

  job->context->ops->matmul(job->destination, job->sources, job->matrix,
                            job->row_count, job->column_count, start_index,
                            end_index);
}

void gf256_matmul(gf256_ctx *context, gf256_t *const *destination,
                  const gf256_t *const *sources, const gf256_t *matrix,
                  size_t row_count, size_t column_count, size_t element_count) {
  if (!row_count || !column_count || !element_count)
    return;

  matrix_multiply_job job = {context, destination, sources,
                             matrix,  row_count,   column_count};

  gf256_parallel(context, element_count, matrix_multiply, &job);
}
