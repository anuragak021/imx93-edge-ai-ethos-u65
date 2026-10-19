#include "imgproc.h"

#include <math.h>
#include <string.h>

/* Triangle filter taps per output pixel. Support is the downscale factor (at most 2048 / 256 = 8),
 * so a tap window is at most 2 * 8 + 1 = 17 wide; 24 leaves margin. */
#define MAX_TAPS 24U
#define ROUND_HALF 0.5
#define PIXEL_MAX 255.0

/* Work buffers are static: no dynamic allocation in this module. Not reentrant. */
static double s_coeff[IMG_MAX_RESIZED_PX][MAX_TAPS];
static uint32_t s_first[IMG_MAX_RESIZED_PX];
static uint32_t s_taps[IMG_MAX_RESIZED_PX];
static uint8_t s_horiz[(size_t)IMG_MAX_INPUT_PX * IMG_MAX_RESIZED_PX * IMG_CHANNELS];
static uint8_t s_resized[(size_t)IMG_MAX_RESIZED_PX * IMG_RESIZE_SHORT_PX * IMG_CHANNELS];

static double triangle(double x)
{
    const double a = fabs(x);
    return (a < 1.0) ? (1.0 - a) : 0.0;
}

static uint8_t to_pixel(double v)
{
    const double r = v + ROUND_HALF;
    if (r <= 0.0)
    {
        return 0U;
    }
    if (r >= PIXEL_MAX)
    {
        return (uint8_t)PIXEL_MAX;
    }
    return (uint8_t)r;
}

/* Same coefficient construction as Pillow's precompute_coeffs for a triangle filter. */
static app_status_t compute_coeffs(uint32_t in_px, uint32_t out_px)
{
    const double scale = (double)in_px / (double)out_px;
    const double filter_scale = (scale < 1.0) ? 1.0 : scale;
    const double support = filter_scale;

    for (uint32_t o = 0U; o < out_px; o++)
    {
        const double center = ((double)o + ROUND_HALF) * scale;
        double lo = center - support + ROUND_HALF;
        double hi = center + support + ROUND_HALF;
        lo = (lo < 0.0) ? 0.0 : lo;
        hi = (hi > (double)in_px) ? (double)in_px : hi;
        const uint32_t first = (uint32_t)lo;
        const uint32_t taps = (uint32_t)hi - first;
        if (taps == 0U || taps > MAX_TAPS)
        {
            return APP_ERR_RANGE;
        }
        double sum = 0.0;
        for (uint32_t t = 0U; t < taps; t++)
        {
            const double w = triangle(((double)(first + t) - center + ROUND_HALF) / filter_scale);
            s_coeff[o][t] = w;
            sum += w;
        }
        if (sum == 0.0)
        {
            return APP_ERR_RANGE;
        }
        for (uint32_t t = 0U; t < taps; t++)
        {
            s_coeff[o][t] /= sum;
        }
        s_first[o] = first;
        s_taps[o] = taps;
    }
    return APP_OK;
}

static void pass_horizontal(const uint8_t *src, uint32_t in_w, uint32_t h, uint32_t out_w)
{
    for (uint32_t y = 0U; y < h; y++)
    {
        for (uint32_t x = 0U; x < out_w; x++)
        {
            for (uint32_t c = 0U; c < IMG_CHANNELS; c++)
            {
                double acc = 0.0;
                for (uint32_t t = 0U; t < s_taps[x]; t++)
                {
                    const size_t i = ((size_t)y * in_w + s_first[x] + t) * IMG_CHANNELS + c;
                    acc += s_coeff[x][t] * (double)src[i];
                }
                s_horiz[((size_t)y * out_w + x) * IMG_CHANNELS + c] = to_pixel(acc);
            }
        }
    }
}

static void pass_vertical(uint32_t w, uint32_t out_h)
{
    for (uint32_t y = 0U; y < out_h; y++)
    {
        for (uint32_t x = 0U; x < w; x++)
        {
            for (uint32_t c = 0U; c < IMG_CHANNELS; c++)
            {
                double acc = 0.0;
                for (uint32_t t = 0U; t < s_taps[y]; t++)
                {
                    const size_t i = ((size_t)(s_first[y] + t) * w + x) * IMG_CHANNELS + c;
                    acc += s_coeff[y][t] * (double)s_horiz[i];
                }
                s_resized[((size_t)y * w + x) * IMG_CHANNELS + c] = to_pixel(acc);
            }
        }
    }
}

static uint32_t scaled_size(uint32_t px, uint32_t short_side)
{
    const double scale = (double)IMG_RESIZE_SHORT_PX / (double)short_side;
    return (uint32_t)floor((double)px * scale + ROUND_HALF);
}

app_status_t img_preprocess(const uint8_t *rgb, uint32_t width, uint32_t height, uint8_t *out)
{
    if (rgb == NULL || out == NULL)
    {
        return APP_ERR_ARG;
    }
    if (width == 0U || height == 0U || width > IMG_MAX_INPUT_PX || height > IMG_MAX_INPUT_PX)
    {
        return APP_ERR_RANGE;
    }
    const uint32_t short_side = (width < height) ? width : height;
    const uint32_t out_w = scaled_size(width, short_side);
    const uint32_t out_h = scaled_size(height, short_side);
    if (out_w > IMG_MAX_RESIZED_PX || out_h > IMG_MAX_RESIZED_PX)
    {
        return APP_ERR_RANGE;
    }

    app_status_t st = compute_coeffs(width, out_w);
    if (st != APP_OK)
    {
        return st;
    }
    pass_horizontal(rgb, width, height, out_w);
    st = compute_coeffs(height, out_h);
    if (st != APP_OK)
    {
        return st;
    }
    pass_vertical(out_w, out_h);

    const uint32_t left = (out_w - IMG_CROP_PX) / 2U;
    const uint32_t top = (out_h - IMG_CROP_PX) / 2U;
    for (uint32_t y = 0U; y < IMG_CROP_PX; y++)
    {
        const size_t src = ((size_t)(top + y) * out_w + left) * IMG_CHANNELS;
        memcpy(&out[(size_t)y * IMG_CROP_PX * IMG_CHANNELS], &s_resized[src],
               (size_t)IMG_CROP_PX * IMG_CHANNELS);
    }
    return APP_OK;
}
