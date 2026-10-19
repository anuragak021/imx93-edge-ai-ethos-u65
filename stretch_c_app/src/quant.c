#include "quant.h"

#include <math.h>
#include <stdbool.h>

#define PIXEL_HALF_RANGE 127.5F
#define INT8_MIN_F (-128.0F)
#define INT8_MAX_F 127.0F

int8_t quant_pixel(uint8_t px, float scale, int32_t zero_point)
{
    const float x = (((float)px / PIXEL_HALF_RANGE) - 1.0F) / scale + (float)zero_point;
    float r = nearbyintf(x); /* round half to even in the default rounding mode, like np.round */
    if (r < INT8_MIN_F)
    {
        r = INT8_MIN_F;
    }
    if (r > INT8_MAX_F)
    {
        r = INT8_MAX_F;
    }
    return (int8_t)r;
}

app_status_t quant_buffer(const uint8_t *src, size_t count, float scale, int32_t zero_point,
                          int8_t *dst)
{
    if (src == NULL || dst == NULL || !(scale > 0.0F))
    {
        return APP_ERR_ARG;
    }
    for (size_t i = 0U; i < count; i++)
    {
        dst[i] = quant_pixel(src[i], scale, zero_point);
    }
    return APP_OK;
}

app_status_t topk_i8(const int8_t *scores, size_t count, size_t k, size_t *indices)
{
    if (scores == NULL || indices == NULL || k == 0U || k > TOP_K || k > count)
    {
        return APP_ERR_ARG;
    }
    for (size_t rank = 0U; rank < k; rank++)
    {
        size_t best = count; /* count means "none chosen yet" */
        for (size_t i = 0U; i < count; i++)
        {
            bool taken = false;
            for (size_t r = 0U; r < rank; r++)
            {
                taken = taken || (indices[r] == i);
            }
            if (!taken && (best == count || scores[i] > scores[best]))
            {
                best = i;
            }
        }
        indices[rank] = best;
    }
    return APP_OK;
}
