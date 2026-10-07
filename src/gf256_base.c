#include "gf256_internal.h"

#include <string.h>

/*
 * Multiply two elements of GF(256) using the standard polynomial-basis
 * multiplication algorithm.
 *
 * Each element is treated as a polynomial over GF(2). Multiplication is
 * performed using shift-and-XOR arithmetic, with reduction by the field's
 * irreducible polynomial whenever the intermediate value grows beyond
 * 8 bits.
 *
 * This is the portable/reference implementation used by the base backend.
 */
gf256_t gf256_mul_scalar(gf256_t a, gf256_t b) {
  unsigned multiplicand = a;
  unsigned multiplier = b;
  unsigned product = 0;

  while (multiplier) {
    /*
     * If the current multiplier bit is set, add the current
     * multiplicand to the product. Addition in GF(256) is XOR.
     */
    if (multiplier & 1u)
      product ^= multiplicand;

    /*
     * Move to the next bit of the multiplier and multiply the
     * multiplicand by x for the next iteration.
     */
    multiplier >>= 1;
    multiplicand <<= 1;

    /*
     * If the multiplication by x produced a ninth bit, reduce
     * the polynomial back into the GF(256) field.
     */
    if (multiplicand & 0x100u)
      multiplicand ^= GF256_REDUCTION_POLY;
  }

  return (gf256_t)product;
}

/*
 * Multiply every element of a vector by the same GF(256) coefficient.
 *
 *     destination[i] = source[i] * coefficient
 *
 * This function is the base/portable implementation of the backend's
 * vector multiplication operation.
 */
static void multiply_vector_by_scalar(gf256_t *destination,
                                      const gf256_t *source,
                                      size_t element_count,
                                      gf256_t coefficient) {
  /*
   * Multiplication by zero always produces zero. Handle this separately
   * so that the multiplication loop can be skipped entirely.
   */
  if (!coefficient) {
    memset(destination, 0, element_count);
    return;
  }

  for (size_t element_index = 0; element_index < element_count;
       element_index++) {

    destination[element_index] =
        gf256_mul_scalar(source[element_index], coefficient);
  }
}

/*
 * Multiply every element of a vector by a GF(256) coefficient and
 * accumulate the result into the destination.
 *
 *     destination[i] += source[i] * coefficient
 *
 * Since addition in GF(256) is XOR:
 *
 *     destination[i] ^= source[i] * coefficient
 */
static void multiply_add_vector_by_scalar(gf256_t *destination,
                                          const gf256_t *source,
                                          size_t element_count,
                                          gf256_t coefficient) {
  /*
   * Multiplication by zero contributes nothing to the destination.
   */
  if (!coefficient)
    return;

  for (size_t element_index = 0; element_index < element_count;
       element_index++) {

    destination[element_index] ^=
        gf256_mul_scalar(source[element_index], coefficient);
  }
}

/*
 * Multiply two vectors element-by-element.
 *
 *     destination[i] = source_a[i] * source_b[i]
 */
static void multiply_vectors_elementwise(gf256_t *destination,
                                         const gf256_t *source_a,
                                         const gf256_t *source_b,
                                         size_t element_count) {
  for (size_t element_index = 0; element_index < element_count;
       element_index++) {

    destination[element_index] =
        gf256_mul_scalar(source_a[element_index], source_b[element_index]);
  }
}

/*
 * Add one vector to another element-by-element.
 *
 *     destination[i] += source[i]
 *
 * Addition in GF(256) is XOR, so this is equivalent to:
 *
 *     destination[i] ^= source[i]
 */
static void add_vectors(gf256_t *destination, const gf256_t *source,
                        size_t element_count) {
  for (size_t element_index = 0; element_index < element_count;
       element_index++) {

    destination[element_index] ^= source[element_index];
  }
}

/*
 * Perform multiple vector multiply-add operations.
 *
 * For each source vector, apply its corresponding coefficient and
 * accumulate the result into the destination:
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
  for (size_t source_index = 0; source_index < source_count; source_index++) {

    multiply_add_vector_by_scalar(destination, sources[source_index],
                                  element_count, coefficients[source_index]);
  }
}

/*
 * Perform matrix multiplication over GF(256).
 *
 * The matrix has 'row_count' rows and 'column_count' columns.
 * Each column corresponds to one source vector.
 *
 * For each output row:
 *
 *     destination[row][i] =
 *         matrix[row][0] * source[0][i]
 *       + matrix[row][1] * source[1][i]
 *       + ...
 *
 * Only the element range [first_element, last_element) is processed.
 *
 * The range parameters allow the higher-level threading code to divide
 * the vector across multiple worker threads.
 */
static void multiply_matrix_vectors(gf256_t *const *destination,
                                    const gf256_t *const *sources,
                                    const gf256_t *matrix, size_t row_count,
                                    size_t column_count, size_t first_element,
                                    size_t last_element) {
  for (size_t row_index = 0; row_index < row_count; row_index++) {

    for (size_t element_index = first_element; element_index < last_element;
         element_index++) {

      gf256_t result = 0;

      for (size_t column_index = 0; column_index < column_count;
           column_index++) {

        /*
         * Matrix addition and field addition are both XOR.
         */
        result ^=
            gf256_mul_scalar(matrix[row_index * column_count + column_index],
                             sources[column_index][element_index]);
      }

      destination[row_index][element_index] = result;
    }
  }
}

/*
 * Portable GF(256) backend operation table.
 *
 * These functions provide the reference implementation used when no
 * architecture-specific GFNI/AVX backend is selected.
 */
const gf256_ops gf256_base_ops = {
    multiply_vector_by_scalar,     multiply_add_vector_by_scalar,
    multiply_vectors_elementwise,  add_vectors,
    multiply_add_multiple_vectors, multiply_matrix_vectors};
