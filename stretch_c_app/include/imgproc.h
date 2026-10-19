/**
 * @file imgproc.h
 * @brief Image preprocessing for MobileNetV2: resize short side to 256, center crop 224.
 *
 * Mirrors model/quantize.py and board/infer.py (Pillow BILINEAR). Pillow's bilinear resize is a
 * separable triangle filter whose support grows when shrinking (it antialiases); this module
 * implements the same filter, horizontal pass first, with an 8-bit rounding between passes.
 */
#ifndef IMGPROC_H
#define IMGPROC_H

#include <stddef.h>
#include <stdint.h>

#include "app_status.h"

#define IMG_CHANNELS 3U
#define IMG_RESIZE_SHORT_PX 256U
#define IMG_CROP_PX 224U
/** Largest accepted input width or height in pixels. */
#define IMG_MAX_INPUT_PX 2048U
/** Largest accepted resized width or height in pixels. */
#define IMG_MAX_RESIZED_PX 2048U
/** Bytes of one preprocessed image (224 x 224 x 3). */
#define IMG_CROP_BYTES ((size_t)IMG_CROP_PX * IMG_CROP_PX * IMG_CHANNELS)

/**
 * @brief Resize an RGB image so its short side is 256 px, then center crop 224 x 224.
 *
 * Uses internal static work buffers, so it is not reentrant.
 *
 * @param[in]  rgb    interleaved RGB, 8 bit, row major, width * height * 3 bytes
 * @param[in]  width  image width in pixels, 1..IMG_MAX_INPUT_PX
 * @param[in]  height image height in pixels, 1..IMG_MAX_INPUT_PX
 * @param[out] out    buffer of IMG_CROP_BYTES bytes, interleaved RGB
 * @return APP_OK, APP_ERR_ARG for null pointers, APP_ERR_RANGE if a dimension is unsupported
 */
app_status_t img_preprocess(const uint8_t *rgb, uint32_t width, uint32_t height, uint8_t *out);

#endif /* IMGPROC_H */
