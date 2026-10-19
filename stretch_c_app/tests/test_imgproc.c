#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "imgproc.h"
#include "test_util.h"

#include "golden_resize.inc"

#define TOLERANCE 1
#define MAX_TEST_PX 1024U

static uint8_t s_input[(size_t)MAX_TEST_PX * MAX_TEST_PX * IMG_CHANNELS];
static uint8_t s_output[IMG_CROP_BYTES];

/* Must match pattern() in tools/gen_resize_golden.py. */
static uint8_t pattern_byte(uint32_t x, uint32_t y, uint32_t c)
{
    return (uint8_t)(((x * 2U + y * 3U + c * 50U) % 200U) + 20U);
}

static void fill_pattern(uint32_t w, uint32_t h)
{
    for (uint32_t y = 0U; y < h; y++)
    {
        for (uint32_t x = 0U; x < w; x++)
        {
            for (uint32_t c = 0U; c < IMG_CHANNELS; c++)
            {
                s_input[((size_t)y * w + x) * IMG_CHANNELS + c] = pattern_byte(x, y, c);
            }
        }
    }
}

static void test_matches_pillow(void)
{
    for (size_t i = 0U; i < sizeof(golden_cases) / sizeof(golden_cases[0]); i++)
    {
        const uint32_t w = golden_cases[i].w;
        const uint32_t h = golden_cases[i].h;
        fill_pattern(w, h);
        CHECK(img_preprocess(s_input, w, h, s_output) == APP_OK);
        int max_diff = 0;
        for (size_t k = 0U; k < GOLDEN_COUNT; k++)
        {
            const int d = abs((int)s_output[k * GOLDEN_STRIDE] - (int)golden_cases[i].data[k]);
            max_diff = (d > max_diff) ? d : max_diff;
        }
        (void)printf("  %ux%u: max |C - Pillow| over %u samples = %d\n", w, h, GOLDEN_COUNT,
                     max_diff);
        CHECK(max_diff <= TOLERANCE);
    }
}

static void test_rejects_bad_input(void)
{
    CHECK(img_preprocess(NULL, 10U, 10U, s_output) == APP_ERR_ARG);
    CHECK(img_preprocess(s_input, 10U, 10U, NULL) == APP_ERR_ARG);
    CHECK(img_preprocess(s_input, 0U, 10U, s_output) == APP_ERR_RANGE);
    CHECK(img_preprocess(s_input, IMG_MAX_INPUT_PX + 1U, 10U, s_output) == APP_ERR_RANGE);
    /* 2048 x 100 would resize to 5243 x 256, beyond the supported resized size. */
    CHECK(img_preprocess(s_input, 2048U, 100U, s_output) == APP_ERR_RANGE);
}

int main(void)
{
    test_matches_pillow();
    test_rejects_bad_input();
    (void)puts(g_failures == 0 ? "test_imgproc: PASS" : "test_imgproc: FAIL");
    return TEST_RESULT();
}
