/**
 * @file quant.h
 * @brief Input quantization and top-k selection for an int8 classifier.
 */
#ifndef QUANT_H
#define QUANT_H

#include <stddef.h>
#include <stdint.h>

#include "app_status.h"

/** Number of classes reported per image. */
#define TOP_K 5U

/**
 * @brief Quantize one 8-bit pixel the way board/infer.py does.
 *
 * Computes round_half_even(((px / 127.5 - 1) / scale) + zero_point) in float32, clamped to int8.
 *
 * @param px          pixel value 0..255
 * @param scale       input tensor quantization scale (must be greater than zero)
 * @param zero_point  input tensor quantization zero point
 * @return quantized value in [-128, 127]
 */
int8_t quant_pixel(uint8_t px, float scale, int32_t zero_point);

/**
 * @brief Quantize an array of pixels.
 *
 * @param[in]  src         pixels
 * @param[in]  count       number of elements
 * @param[in]  scale       input tensor scale
 * @param[in]  zero_point  input tensor zero point
 * @param[out] dst         output, count elements
 * @return APP_OK or APP_ERR_ARG (null pointer or non positive scale)
 */
app_status_t quant_buffer(const uint8_t *src, size_t count, float scale, int32_t zero_point,
                          int8_t *dst);

/**
 * @brief Select the k highest scores, highest first. Ties resolve to the lower index.
 *
 * @param[in]  scores   class scores
 * @param[in]  count    number of scores
 * @param[in]  k        how many to return, 1..TOP_K and not more than count
 * @param[out] indices  k indices into scores
 * @return APP_OK or APP_ERR_ARG
 */
app_status_t topk_i8(const int8_t *scores, size_t count, size_t k, size_t *indices);

#endif /* QUANT_H */
