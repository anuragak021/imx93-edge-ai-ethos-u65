#include <stdint.h>
#include <stdio.h>

#include "quant.h"
#include "test_util.h"

static void test_quant_pixel(void)
{
    /* scale 1/128 is exact in float32: x = (px / 127.5 - 1) * 128 + zp. */
    const float scale = 0.0078125F;
    CHECK(quant_pixel(0U, scale, -1) == -128);   /* -129 clamps low */
    CHECK(quant_pixel(255U, scale, -1) == 127);  /* exactly 127 */
    CHECK(quant_pixel(128U, scale, -1) == 0);    /* -0.498 rounds to 0 */
    CHECK(quant_pixel(127U, scale, -1) == -2);   /* -1.502 rounds to -2 */
    CHECK(quant_pixel(255U, scale, 100) == 127); /* clamps high */
    CHECK(quant_pixel(0U, scale, -100) == -128); /* clamps low */
}

static void test_quant_buffer(void)
{
    const uint8_t src[3] = {0U, 128U, 255U};
    int8_t dst[3] = {0, 0, 0};
    CHECK(quant_buffer(src, 3U, 0.0078125F, -1, dst) == APP_OK);
    CHECK(dst[0] == -128 && dst[1] == 0 && dst[2] == 127);
    CHECK(quant_buffer(NULL, 3U, 1.0F, 0, dst) == APP_ERR_ARG);
    CHECK(quant_buffer(src, 3U, 0.0F, 0, dst) == APP_ERR_ARG);
    CHECK(quant_buffer(src, 3U, -1.0F, 0, dst) == APP_ERR_ARG);
}

static void test_topk(void)
{
    const int8_t scores[8] = {5, -3, 90, 90, 7, -128, 127, 0};
    size_t idx[TOP_K] = {0};
    CHECK(topk_i8(scores, 8U, 5U, idx) == APP_OK);
    CHECK(idx[0] == 6U); /* 127 */
    CHECK(idx[1] == 2U); /* tie 90: lower index first */
    CHECK(idx[2] == 3U);
    CHECK(idx[3] == 4U); /* 7 */
    CHECK(idx[4] == 0U); /* 5 */
    CHECK(topk_i8(scores, 8U, 9U, idx) == APP_ERR_ARG);
    CHECK(topk_i8(scores, 8U, 0U, idx) == APP_ERR_ARG);
    CHECK(topk_i8(scores, 3U, 5U, idx) == APP_ERR_ARG);
    CHECK(topk_i8(NULL, 8U, 1U, idx) == APP_ERR_ARG);
}

int main(void)
{
    test_quant_pixel();
    test_quant_buffer();
    test_topk();
    (void)puts(g_failures == 0 ? "test_quant: PASS" : "test_quant: FAIL");
    return TEST_RESULT();
}
