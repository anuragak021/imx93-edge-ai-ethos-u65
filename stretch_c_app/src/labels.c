#include "labels.h"

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#define BACKGROUND_LABEL "background"

static char s_file_buf[LABELS_FILE_MAX];

static app_status_t add_line(labels_t *out, const char *line, size_t len)
{
    if (out->count >= LABELS_MAX)
    {
        return APP_ERR_RANGE;
    }
    if (len > 0U && line[len - 1U] == '\r')
    {
        len--;
    }
    if (len >= LABEL_NAME_MAX)
    {
        len = LABEL_NAME_MAX - 1U;
    }
    memcpy(out->name[out->count], line, len);
    out->name[out->count][len] = '\0';
    out->count++;
    return APP_OK;
}

app_status_t labels_parse(const char *text, size_t length, size_t expected_count, labels_t *out)
{
    if (text == NULL || out == NULL)
    {
        return APP_ERR_ARG;
    }
    out->count = 0U;
    size_t start = 0U;
    for (size_t i = 0U; i < length; i++)
    {
        if (text[i] == '\n')
        {
            const app_status_t st = add_line(out, &text[start], i - start);
            if (st != APP_OK)
            {
                return st;
            }
            start = i + 1U;
        }
    }
    if (start < length)
    {
        const app_status_t st = add_line(out, &text[start], length - start);
        if (st != APP_OK)
        {
            return st;
        }
    }
    if (out->count == expected_count + 1U && strcmp(out->name[0], BACKGROUND_LABEL) == 0)
    {
        for (size_t i = 1U; i < out->count; i++)
        {
            memcpy(out->name[i - 1U], out->name[i], LABEL_NAME_MAX);
        }
        out->count--;
    }
    return (out->count == expected_count) ? APP_OK : APP_ERR_FORMAT;
}

app_status_t labels_load(const char *path, size_t expected_count, labels_t *out)
{
    if (path == NULL || out == NULL)
    {
        return APP_ERR_ARG;
    }
    FILE *f = fopen(path, "rb");
    if (f == NULL)
    {
        return APP_ERR_IO;
    }
    const size_t n = fread(s_file_buf, 1U, sizeof(s_file_buf), f);
    const bool too_big = (n == sizeof(s_file_buf)) && (fgetc(f) != EOF);
    const bool read_err = (ferror(f) != 0);
    (void)fclose(f);
    if (read_err)
    {
        return APP_ERR_IO;
    }
    if (too_big)
    {
        return APP_ERR_RANGE;
    }
    return labels_parse(s_file_buf, n, expected_count, out);
}
