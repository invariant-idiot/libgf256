#include "gf256_internal.h"
#include <immintrin.h>
#include <string.h>

/*
 * GFNI operates in GF(2^8) using the AES polynomial 0x11b, while this
 * library uses polynomial 0x11d.
 *
 * The two fields are isomorphic. The transformation below maps values
 * between the two representations so that GFNI multiplication can be
 * used without changing the library's field semantics.
 *
 * The transformation is self-inverse for the selected mapping, so the
 * same matrix is used both before and after GFNI multiplication.
 *
 * The matrix layout follows GFNI's required bit ordering. See the
 * corresponding AVX-256 backend for the derivation of this constant.
 */
#define GFNI_ISOMORPHISM_MATRIX \
    _mm512_set1_epi64((long long)0xffaacc88f0a0c080ULL)

/*
 * Multiply 64 GF(2^8) values using AVX-512 GFNI instructions.
 *
 * Operands are transformed from the library's 0x11d representation into
 * the AES 0x11b representation, multiplied by GFNI, and transformed back.
 */
static inline __m512i multiply_gfni(__m512i left, __m512i right)
{
    const __m512i transformation = GFNI_ISOMORPHISM_MATRIX;

    __m512i left_transformed =
        _mm512_gf2p8affine_epi64_epi8(left, transformation, 0);

    __m512i right_transformed =
        _mm512_gf2p8affine_epi64_epi8(right, transformation, 0);

    __m512i product_in_aes_field =
        _mm512_gf2p8mul_epi8(left_transformed, right_transformed);

    return _mm512_gf2p8affine_epi64_epi8(
        product_in_aes_field,
        transformation,
        0);
}

/* Multiply every element of a vector by the same GF(2^8) scalar. */
static void multiply_vector_by_scalar(
    gf256_t *destination,
    const gf256_t *source,
    size_t element_count,
    gf256_t scalar)
{
    if (!scalar) {
        memset(destination, 0, element_count);
        return;
    }

    const __m512i scalar_vector = _mm512_set1_epi8((char)scalar);

    size_t element_index = 0;

    for (; element_index + 64 <= element_count; element_index += 64) {
        __m512i source_vector =
            _mm512_loadu_si512(source + element_index);

        __m512i result =
            multiply_gfni(source_vector, scalar_vector);

        _mm512_storeu_si512(
            destination + element_index,
            result);
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
static void multiply_add_vector_by_scalar(
    gf256_t *destination,
    const gf256_t *source,
    size_t element_count,
    gf256_t scalar)
{
    if (!scalar)
        return;

    const __m512i scalar_vector = _mm512_set1_epi8((char)scalar);

    size_t element_index = 0;

    for (; element_index + 64 <= element_count; element_index += 64) {
        __m512i source_vector =
            _mm512_loadu_si512(source + element_index);

        __m512i product =
            multiply_gfni(source_vector, scalar_vector);

        __m512i destination_vector =
            _mm512_loadu_si512(destination + element_index);

        _mm512_storeu_si512(
            destination + element_index,
            _mm512_xor_si512(destination_vector, product));
    }

    for (; element_index < element_count; element_index++) {
        destination[element_index] ^=
            gf256_mul_scalar(source[element_index], scalar);
    }
}

/* Multiply corresponding elements of two vectors. */
static void multiply_vectors_elementwise(
    gf256_t *destination,
    const gf256_t *source_a,
    const gf256_t *source_b,
    size_t element_count)
{
    size_t element_index = 0;

    for (; element_index + 64 <= element_count; element_index += 64) {
        __m512i left =
            _mm512_loadu_si512(source_a + element_index);

        __m512i right =
            _mm512_loadu_si512(source_b + element_index);

        _mm512_storeu_si512(
            destination + element_index,
            multiply_gfni(left, right));
    }

    for (; element_index < element_count; element_index++) {
        destination[element_index] =
            gf256_mul_scalar(
                source_a[element_index],
                source_b[element_index]);
    }
}

/* XOR one vector into another. */
static void add_vectors(
    gf256_t *destination,
    const gf256_t *source,
    size_t element_count)
{
    size_t element_index = 0;

    for (; element_index + 64 <= element_count; element_index += 64) {
        __m512i destination_vector =
            _mm512_loadu_si512(destination + element_index);

        __m512i source_vector =
            _mm512_loadu_si512(source + element_index);

        _mm512_storeu_si512(
            destination + element_index,
            _mm512_xor_si512(destination_vector, source_vector));
    }

    for (; element_index < element_count; element_index++)
        destination[element_index] ^= source[element_index];
}

/*
 * Multiply each source vector by its corresponding scalar and XOR all
 * resulting vectors into the destination.
 *
 * Equivalent to:
 *
 *     destination += source[0] * coefficient[0]
 *                   + source[1] * coefficient[1]
 *                   + ...
 */
static void multiply_add_multiple_vectors(
    gf256_t *destination,
    const gf256_t *const *sources,
    const gf256_t *coefficients,
    size_t source_count,
    size_t element_count)
{
    size_t element_index = 0;

    for (; element_index + 64 <= element_count; element_index += 64) {
        __m512i result =
            _mm512_loadu_si512(destination + element_index);

        for (size_t source_index = 0;
             source_index < source_count;
             source_index++) {

            __m512i source_vector =
                _mm512_loadu_si512(
                    sources[source_index] + element_index);

            __m512i coefficient_vector =
                _mm512_set1_epi8(
                    (char)coefficients[source_index]);

            result = _mm512_xor_si512(
                result,
                multiply_gfni(source_vector, coefficient_vector));
        }

        _mm512_storeu_si512(
            destination + element_index,
            result);
    }

    for (; element_index < element_count; element_index++) {
        for (size_t source_index = 0;
             source_index < source_count;
             source_index++) {

            destination[element_index] ^=
                gf256_mul_scalar(
                    sources[source_index][element_index],
                    coefficients[source_index]);
        }
    }
}

/*
 * Multiply a GF(2^8) coefficient matrix by a set of source vectors.
 *
 * Only the element range [first_element, last_element) is processed.
 * This allows the operation to be split across multiple threads.
 */
static void multiply_matrix_vectors(
    gf256_t *const *destination_vectors,
    const gf256_t *const *source_vectors,
    const gf256_t *matrix,
    size_t row_count,
    size_t column_count,
    size_t first_element,
    size_t last_element)
{
    const size_t element_count =
        last_element - first_element;

    /*
     * Adjust the source pointers once so the inner multiplication loop
     * can work directly with the requested element range.
     */
    const gf256_t *shifted_sources[column_count];

    for (size_t column_index = 0;
         column_index < column_count;
         column_index++) {

        shifted_sources[column_index] =
            source_vectors[column_index] + first_element;
    }

    for (size_t row_index = 0;
         row_index < row_count;
         row_index++) {

        gf256_t *destination_row =
            destination_vectors[row_index] + first_element;

        memset(destination_row, 0, element_count);

        /*
         * Each matrix row supplies the coefficients used to combine
         * all source vectors into the corresponding destination row.
         */
        multiply_add_multiple_vectors(
            destination_row,
            shifted_sources,
            matrix + row_index * column_count,
            column_count,
            element_count);
    }
}

const gf256_ops gf256_avx512_ops = {
    multiply_vector_by_scalar,
    multiply_add_vector_by_scalar,
    multiply_vectors_elementwise,
    add_vectors,
    multiply_add_multiple_vectors,
    multiply_matrix_vectors
};
