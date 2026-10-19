/* Minimal assert-based test helper: counts failures instead of aborting. */
#ifndef TEST_UTIL_H
#define TEST_UTIL_H

#include <stdio.h>

static int g_failures = 0;

#define CHECK(cond)                                                                                \
    do                                                                                             \
    {                                                                                              \
        if (!(cond))                                                                               \
        {                                                                                          \
            (void)fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                  \
            g_failures++;                                                                          \
        }                                                                                          \
    } while (0)

#define TEST_RESULT() ((g_failures == 0) ? 0 : 1)

#endif /* TEST_UTIL_H */
