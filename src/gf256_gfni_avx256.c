#include "gf256_internal.h"
#include <immintrin.h>
#include <string.h>

/*
 * GFNI's GF2P8MULB instruction operates in GF(2^8) using the AES
 * polynomial 0x11b, while this library uses polynomial 0x11d.
 *
 * The two fields are isomorphic, so operands can be mapped from the
 * 0x11d representation into the 0x11b representation, multiplied using
 * GFNI, and then mapped back:
 *
 *     a * b = T^-1(T(a) * T(b))
 *
 * The transformation T is implemented with GF2P8AFFINEQB. For the
 * selected isomorphism, T is self-inverse, so the same transformation
 * is used before and after multiplication.
 *
 * The matrix is packed according to GFNI's bit ordering: output row i
 * is stored in byte position (7 - i).
 */
#define GFNI_ISOMORPHISM_MATRIX                                                \
  _mm256_set1_epi64x((long long)0xffaacc88f0a0c080ULL)

/*
 * Multiply 32 GF(2^8) values using the GFNI instruction set.
 *
 * GFNI performs the multiplication in the AES polynomial field (0x11b),
 * so the operands are first transformed into that representation and the
 * result is transformed back into this library's 0x11d representation.
 */
static inline __m256i multiply_gfni(__m256i left, __m256i right) {
  const __m256i transformation = GFNI_ISOMORPHISM_MATRIX;

  __m256i left_transformed =
      _mm256_gf2p8affine_epi64_epi8(left, transformation, 0);

  __m256i right_transformed =
      _mm256_gf2p8affine_epi64_epi8(right, transformation, 0);

  __m256i product_in_aes_field =
      _mm256_gf2p8mul_epi8(left_transformed, right_transformed);

  return _mm256_gf2p8affine_epi64_epi8(product_in_aes_field, transformation, 0);
}

/* Multiply every element of a vector by the same GF(2^8) scalar. */
static void multiply_vector_by_scalar(gf256_t *destination,
                                      const gf256_t *source,
                                      size_t element_count, gf256_t scalar) {
  if (!scalar) {
    memset(destination, 0, element_count);
    return;
  }

  const __m256i scalar_vector = _mm256_set1_epi8((char)scalar);

  size_t element_index = 0;

  for (; element_index + 32 <= element_count; element_index += 32) {
    __m256i source_vector =
        _mm256_loadu_si256((const __m256i *)(source + element_index));

    __m256i result = multiply_gfni(source_vector, scalar_vector);

    _mm256_storeu_si256((__m256i *)(destination + element_index), result);
  }

  for (; element_index < element_count; element_index++) {
    destination[element_index] =
        gf256_mul_scalar(source[element_index], scalar);
  }
}

/*
 * Multiply every element of a vector by a scalar and XOR the result
 * into the destination vector.
 */
static void multiply_add_vector_by_scalar(gf256_t *destination,
                                          const gf256_t *source,
                                          size_t element_count,
                                          gf256_t scalar) {
  if (!scalar)
    return;

  const __m256i scalar_vector = _mm256_set1_epi8((char)scalar);

  size_t element_index = 0;

  for (; element_index + 32 <= element_count; element_index += 32) {
    __m256i source_vector =
        _mm256_loadu_si256((const __m256i *)(source + element_index));

    __m256i product = multiply_gfni(source_vector, scalar_vector);

    __m256i destination_vector =
        _mm256_loadu_si256((const __m256i *)(destination + element_index));

    _mm256_storeu_si256((__m256i *)(destination + element_index),
                        _mm256_xor_si256(destination_vector, product));
  }

  for (; element_index < element_count; element_index++) {
    destination[element_index] ^=
        gf256_mul_scalar(source[element_index], scalar);
  }
}

/* Multiply corresponding elements of two vectors. */
static void multiply_vectors_elementwise(gf256_t *destination,
                                         const gf256_t *source_a,
                                         const gf256_t *source_b,
                                         size_t element_count) {
  size_t element_index = 0;

  for (; element_index + 32 <= element_count; element_index += 32) {
    __m256i left =
        _mm256_loadu_si256((const __m256i *)(source_a + element_index));

    __m256i right =
        _mm256_loadu_si256((const __m256i *)(source_b + element_index));

    _mm256_storeu_si256((__m256i *)(destination + element_index),
                        multiply_gfni(left, right));
  }

  for (; element_index < element_count; element_index++) {
    destination[element_index] =
        gf256_mul_scalar(source_a[element_index], source_b[element_index]);
  }
}

/* XOR one vector into another. */
static void add_vectors(gf256_t *destination, const gf256_t *source,
                        size_t element_count) {
  size_t element_index = 0;

  for (; element_index + 32 <= element_count; element_index += 32) {
    __m256i destination_vector =
        _mm256_loadu_si256((const __m256i *)(destination + element_index));

    __m256i source_vector =
        _mm256_loadu_si256((const __m256i *)(source + element_index));

    _mm256_storeu_si256((__m256i *)(destination + element_index),
                        _mm256_xor_si256(destination_vector, source_vector));
  }

  for (; element_index < element_count; element_index++)
    destination[element_index] ^= source[element_index];
}

/*
 * Multiply each source vector by its corresponding scalar and XOR all
 * resulting vectors into the destination.
 *
 * This is equivalent to:
 *
 *     destination += source[0] * coefficient[0]
 *                   + source[1] * coefficient[1]
 *                   + ...
 */
static void multiply_add_multiple_vectors(gf256_t *destination,
                                          const gf256_t *const *sources,
                                          const gf256_t *coefficients,
                                          size_t source_count,
                                          size_t element_count) {
  size_t element_index = 0;

  for (; element_index + 32 <= element_count; element_index += 32) {
    __m256i result =
        _mm256_loadu_si256((const __m256i *)(destination + element_index));

    for (size_t source_index = 0; source_index < source_count; source_index++) {

      __m256i source_vector = _mm256_loadu_si256(
          (const __m256i *)(sources[source_index] + element_index));

      __m256i coefficient_vector =
          _mm256_set1_epi8((char)coefficients[source_index]);

      result = _mm256_xor_si256(
          result, multiply_gfni(source_vector, coefficient_vector));
    }

    _mm256_storeu_si256((__m256i *)(destination + element_index), result);
  }

  for (; element_index < element_count; element_index++) {
    for (size_t source_index = 0; source_index < source_count; source_index++) {

      destination[element_index] ^= gf256_mul_scalar(
          sources[source_index][element_index], coefficients[source_index]);
    }
  }
}

/*
 * Multiply a matrix of GF(2^8) coefficients by a set of source vectors.
 *
 * Only the element range [first_element, last_element) is processed.
 * This allows callers to divide the work across multiple threads.
 */
static void multiply_matrix_vectors(gf256_t *const *destination_vectors,
                                    const gf256_t *const *source_vectors,
                                    const gf256_t *matrix, size_t row_count,
                                    size_t column_count, size_t first_element,
                                    size_t last_element) {
  const size_t element_count = last_element - first_element;

  /*
   * Shift each source vector to the requested element range once,
   * rather than repeatedly adding the offset inside the inner loop.
   */
  const gf256_t *shifted_sources[column_count];

  for (size_t column_index = 0; column_index < column_count; column_index++) {

    shifted_sources[column_index] =
        source_vectors[column_index] + first_element;
  }

  for (size_t row_index = 0; row_index < row_count; row_index++) {

    gf256_t *destination_row = destination_vectors[row_index] + first_element;

    memset(destination_row, 0, element_count);

    /*
     * Each matrix row contains the coefficients used to combine
     * the source vectors for the corresponding destination row.
     */
    multiply_add_multiple_vectors(destination_row, shifted_sources,
                                  matrix + row_index * column_count,
                                  column_count, element_count);
  }
}

const gf256_ops gf256_avx256_ops = {
    multiply_vector_by_scalar,     multiply_add_vector_by_scalar,
    multiply_vectors_elementwise,  add_vectors,
    multiply_add_multiple_vectors, multiply_matrix_vectors};
