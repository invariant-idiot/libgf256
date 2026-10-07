/**
 * @file libgf256.h
 * @brief Arithmetic and vector operations over the finite field GF(2^8).
 *
 * This library implements arithmetic in the finite field GF(256), also
 * commonly written as GF(2^8). Field elements are represented by 8-bit
 * unsigned integers.
 *
 * The field is constructed using the irreducible polynomial:
 *
 *     x^8 + x^4 + x^3 + x^2 + 1
 *
 * whose polynomial representation is 0x11d.
 *
 * Addition and subtraction are both implemented as bitwise XOR because
 * GF(2^8) has characteristic two.
 *
 * The library can use different computational backends. The AUTO backend
 * selects an appropriate implementation based on the capabilities of the
 * host CPU.
 *
 * @note Unless explicitly documented otherwise, vector operations operate
 * on
 *       @p n elements and expect valid, non-overlapping input/output buffers
 *       where required by the implementation.
 *
 */

#ifndef LIBGF256_H
#define LIBGF256_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief An element of the finite field GF(2^8).
 *
 * Each field element is represented by an 8-bit unsigned integer whose bits
 * represent the coefficients of a polynomial over GF(2).
 *
 * For example:
 *
 *     0x53 = x^6 + x^4 + x + 1
 *
 * Arithmetic is performed modulo GF256_REDUCTION_POLY.
 */
typedef uint8_t gf256_t;

/**
 * @brief Opaque context containing the selected GF(256) execution backend.
 *
 * A context is created with gf256_init() and must eventually be released
 * with gf256_free().
 *
 * The context may contain backend-specific state and resources, including
 * thread-pool or CPU-specific implementation details.
 */
typedef struct gf256_ctx gf256_ctx;

/**
 * @brief Reduction polynomial used to construct GF(2^8).
 *
 * The polynomial is:
 *
 *     x^8 + x^4 + x^3 + x^2 + 1
 *
 * and is represented as 0x11d.
 *
 * This is the conventional polynomial used by Reed-Solomon coding and
 * several other GF(256) applications.
 */
#define GF256_REDUCTION_POLY 0x11d

/**
 * @brief Available GF(256) computation backends.
 *
 * A backend determines how arithmetic and vector operations are implemented.
 * Hardware-accelerated backends may provide substantially higher throughput
 * when the required CPU instruction sets are available.
 */
typedef enum gf256_backend {
  /**
   * @brief Automatically select the best available backend.
   *
   * The implementation detects CPU capabilities and selects an appropriate
   * backend at initialization time.
   */
  GF256_BACKEND_AUTO = 0,

  /**
   * @brief Portable baseline implementation.
   *
   * This backend does not require GFNI or AVX2/AVX-512 instructions and is
   * intended to work on conventional CPUs supported by the library.
   */
  GF256_BACKEND_BASE,

  /**
   * @brief GFNI implementation using 256-bit AVX vectors.
   *
   * Requires CPU support for GFNI and the required 256-bit vector
   * instruction set.
   */
  GF256_BACKEND_GFNI_AVX256,

  /**
   * @brief GFNI implementation using 512-bit AVX vectors.
   *
   * Requires CPU support for GFNI and the required 512-bit vector
   * instruction set.
   */
  GF256_BACKEND_GFNI_AVX512
} gf256_backend;

/**
 * @brief Create and initialize a GF(256) execution context.
 *
 * @param backend Requested computation backend.
 * @param threads Number of worker threads to use for operations that support
 *                parallel execution. The interpretation of zero is
 *                implementation-defined; implementations commonly use an
 *                automatic/default thread count.
 *
 * @return A newly allocated and initialized context on success.
 * @return NULL if the requested backend is unavailable or initialization
 *         fails.
 *
 * @note When @p backend is GF256_BACKEND_AUTO, the implementation selects
 *       the best supported backend automatically.
 *
 * @see gf256_free()
 * @see gf256_backend_available()
 */
gf256_ctx *gf256_init(gf256_backend backend, unsigned threads);

/**
 * @brief Destroy a GF(256) execution context.
 *
 * Releases all resources associated with @p ctx.
 *
 * @param ctx Context returned by gf256_init().
 *
 * @note Passing NULL is safe and has no effect.
 */
void gf256_free(gf256_ctx *ctx);

/**
 * @brief Add two GF(256) elements.
 *
 * Addition in GF(2^8) is equivalent to bitwise XOR.
 *
 * @param a First field element.
 * @param b Second field element.
 *
 * @return a + b in GF(256).
 */
static inline gf256_t gf256_add(gf256_t a, gf256_t b) {
  return (gf256_t)(a ^ b);
}

/**
 * @brief Subtract two GF(256) elements.
 *
 * GF(2^8) has characteristic two, therefore subtraction is identical to
 * addition and is implemented as bitwise XOR.
 *
 * @param a Minuend.
 * @param b Subtrahend.
 *
 * @return a - b in GF(256).
 */
static inline gf256_t gf256_sub(gf256_t a, gf256_t b) {
  return (gf256_t)(a ^ b);
}

/**
 * @brief Multiply two GF(256) elements.
 *
 * Performs polynomial multiplication followed by reduction using
 * GF256_REDUCTION_POLY.
 *
 * @param a First field element.
 * @param b Second field element.
 *
 * @return a * b in GF(256).
 */
gf256_t gf256_mul(gf256_t a, gf256_t b);

/**
 * @brief Compute the multiplicative inverse of a GF(256) element.
 *
 * For every non-zero field element @p a, this function returns an element
 * satisfying:
 *
 *     a * gf256_inv(a) = 1
 *
 * @param a Field element to invert.
 *
 * @return The multiplicative inverse of @p a.
 * @return 0 when @p a is zero, since zero has no multiplicative inverse.
 */
gf256_t gf256_inv(gf256_t a);

/**
 * @brief Raise a GF(256) element to an integer power.
 *
 * Computes:
 *
 *     a^exponent
 *
 * using GF(256) multiplication.
 *
 * Since every non-zero element of GF(256) has multiplicative order dividing
 * 255, exponents for non-zero elements may be reduced modulo 255.
 *
 * @param a Base field element.
 * @param exponent Non-negative exponent.
 *
 * @return a raised to @p exponent in GF(256).
 *
 * @note By convention, a^0 is 1, including 0^0 if the implementation follows
 *       the usual integer-power convention. Callers requiring a specific
 *       interpretation of 0^0 should handle that case explicitly.
 */
gf256_t gf256_pow(gf256_t a, uint32_t exponent);

/**
 * @brief Multiply every element of a vector by a field coefficient.
 *
 * Computes:
 *
 *     dst[i] = src[i] * coefficient
 *
 * for @p i in the range [0, n).
 *
 * @param ctx GF(256) execution context.
 * @param dst Destination vector containing @p n elements.
 * @param src Source vector containing @p n elements.
 * @param n Number of field elements to process.
 * @param coefficient Multiplication coefficient.
 */
void gf256_mul_vec(gf256_ctx *ctx, gf256_t *dst, const gf256_t *src, size_t n,
                   gf256_t coefficient);

/**
 * @brief Multiply a vector by a coefficient and XOR the result into another.
 *
 * Computes:
 *
 *     dst[i] ^= src[i] * coefficient
 *
 * for @p i in the range [0, n).
 *
 * This operation is commonly used by Reed-Solomon and erasure-coding
 * implementations.
 *
 * @param ctx GF(256) execution context.
 * @param dst Destination vector and accumulator.
 * @param src Source vector containing @p n elements.
 * @param n Number of field elements to process.
 * @param coefficient Multiplication coefficient.
 */
void gf256_muladd_vec(gf256_ctx *ctx, gf256_t *dst, const gf256_t *src,
                      size_t n, gf256_t coefficient);

/**
 * @brief Multiply two vectors element-by-element.
 *
 * Computes:
 *
 *     dst[i] = a[i] * b[i]
 *
 * for @p i in the range [0, n).
 *
 * @param ctx GF(256) execution context.
 * @param dst Destination vector.
 * @param a First input vector.
 * @param b Second input vector.
 * @param n Number of field elements to process.
 */
void gf256_mul_vec_vec(gf256_ctx *ctx, gf256_t *dst, const gf256_t *a,
                       const gf256_t *b, size_t n);

/**
 * @brief Add one vector to another.
 *
 * Computes:
 *
 *     dst[i] ^= src[i]
 *
 * for @p i in the range [0, n).
 *
 * @param ctx GF(256) execution context.
 * @param dst Destination vector and accumulator.
 * @param src Source vector.
 * @param n Number of field elements to process.
 */
void gf256_add_vec(gf256_ctx *ctx, gf256_t *dst, const gf256_t *src, size_t n);

/**
 * @brief Subtract one vector from another.
 *
 * Subtraction and addition are identical in GF(2^8), so this function is
 * implemented in terms of gf256_add_vec().
 *
 * Computes:
 *
 *     dst[i] = dst[i] - src[i]
 *            = dst[i] ^ src[i]
 *
 * for @p i in the range [0, n).
 *
 * @param ctx GF(256) execution context.
 * @param dst Destination vector and accumulator.
 * @param src Source vector.
 * @param n Number of field elements to process.
 */
static inline void gf256_sub_vec(gf256_ctx *ctx, gf256_t *dst,
                                 const gf256_t *src, size_t n) {
  gf256_add_vec(ctx, dst, src, n);
}

/**
 * @brief Multiply multiple vectors by coefficients and accumulate them.
 *
 * Conceptually computes:
 *
 *     dst[i] ^= src[0][i] * coefficients[0]
 *             ^ src[1][i] * coefficients[1]
 *             ^ ...
 *             ^ src[count - 1][i] * coefficients[count - 1]
 *
 * for every @p i in the range [0, n).
 *
 * @param ctx GF(256) execution context.
 * @param dst Destination vector and accumulator.
 * @param src Array of @p count source-vector pointers.
 * @param coefficients Array of @p count GF(256) coefficients.
 * @param count Number of source vectors and coefficients.
 * @param n Number of elements in each vector.
 *
 * @note The caller is responsible for ensuring that each source vector
 *       contains at least @p n elements.
 */
void gf256_madd(gf256_ctx *ctx, gf256_t *dst, const gf256_t *const *src,
                const gf256_t *coefficients, size_t count, size_t n);

/**
 * @brief Multiply a matrix by a set of GF(256) vectors.
 *
 * Treats @p matrix as a @p rows-by-@p cols matrix over GF(256) and
 * @p src as @p cols input vectors, each containing @p n elements.
 *
 * The operation computes:
 *
 *     dst[r][i] = sum(matrix[r][c] * src[c][i])
 *                for c = 0 .. cols-1
 *
 * where the sum is performed using GF(256) addition (XOR).
 *
 * In other words, for every vector position @p i:
 *
 *     dst[i] = matrix * src[i]
 *
 * @param ctx GF(256) execution context.
 * @param dst Array of @p rows destination vectors.
 * @param src Array of @p cols source vectors.
 * @param matrix Matrix coefficients in row-major order. The element at
 *               row @p r and column @p c is matrix[r * cols + c].
 * @param rows Number of rows in the matrix and number of destination vectors.
 * @param cols Number of columns in the matrix and number of source vectors.
 * @param n Number of elements in each source and destination vector.
 *
 * @note The caller must provide sufficient storage for all source and
 *       destination vectors and the matrix.
 */
void gf256_matmul(gf256_ctx *ctx, gf256_t *const *dst,
                  const gf256_t *const *src, const gf256_t *matrix, size_t rows,
                  size_t cols, size_t n);

/**
 * @brief Return the backend selected by a context.
 *
 * @param ctx GF(256) execution context.
 *
 * @return The backend currently used by @p ctx.
 */
gf256_backend gf256_get_backend(const gf256_ctx *ctx);

/**
 * @brief Get a human-readable name for the active backend.
 *
 * @param ctx GF(256) execution context.
 *
 * @return Null-terminated backend name.
 *
 * @note The returned string is owned by the library and must not be freed
 *       or modified by the caller.
 */
const char *gf256_get_backend_name(const gf256_ctx *ctx);

/**
 * @brief Check whether a backend is available on the current system.
 *
 * This function can be used before gf256_init() to determine whether a
 * particular hardware-accelerated backend can be used.
 *
 * @param backend Backend to test.
 *
 * @return Non-zero if the backend is available.
 * @return Zero if the backend is unavailable or unsupported.
 *
 * @note GF256_BACKEND_AUTO is expected to be available whenever the library
 *       itself can be initialized.
 */
int gf256_backend_available(gf256_backend backend);

/**
 * @brief Return the number of worker threads configured for a context.
 *
 * @param ctx GF(256) execution context.
 *
 * @return Number of configured worker threads.
 */
unsigned gf256_get_threads(const gf256_ctx *ctx);

/**
 * @brief Return the number of non-zero elements in GF(256).
 *
 * GF(256) contains 256 elements total, of which 255 are non-zero.
 *
 * @return 255.
 */
static inline unsigned gf256_field_order(void) {
  return 255u;
}

/**
 * @brief Test whether a GF(256) element is zero.
 *
 * @param a Field element.
 *
 * @return Non-zero if @p a is zero; otherwise zero.
 */
static inline int gf256_is_zero(gf256_t a) {
  return a == 0;
}

/**
 * @brief Test whether a GF(256) element is the multiplicative identity.
 *
 * @param a Field element.
 *
 * @return Non-zero if @p a is one; otherwise zero.
 */
static inline int gf256_is_one(gf256_t a) {
  return a == 1;
}

#ifdef __cplusplus
}
#endif

#endif /* LIBGF256_H */
