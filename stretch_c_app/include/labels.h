/**
 * @file labels.h
 * @brief Class label list loading (one label per line).
 */
#ifndef LABELS_H
#define LABELS_H

#include <stddef.h>

#include "app_status.h"

#define LABELS_MAX 1024U
#define LABEL_NAME_MAX 64U
/** Largest accepted labels file in bytes. */
#define LABELS_FILE_MAX 65536U

/** Label table. Names longer than LABEL_NAME_MAX - 1 are truncated. */
typedef struct
{
    char name[LABELS_MAX][LABEL_NAME_MAX];
    size_t count;
} labels_t;

/**
 * @brief Parse labels from text. A leading "background" line is dropped when the line count is
 *        expected_count + 1 (TF-slim style file; Keras MobileNetV2 has no background class).
 *
 * @param[in]  text            file content
 * @param[in]  length          bytes in text
 * @param[in]  expected_count  number of model outputs
 * @param[out] out             parsed labels
 * @return APP_OK, APP_ERR_ARG, or APP_ERR_FORMAT if the final count differs from expected_count
 */
app_status_t labels_parse(const char *text, size_t length, size_t expected_count, labels_t *out);

/**
 * @brief Read a labels file and parse it with labels_parse().
 *
 * @param[in]  path            file path
 * @param[in]  expected_count  number of model outputs
 * @param[out] out             parsed labels
 * @return APP_OK, APP_ERR_IO, APP_ERR_RANGE (file too large), or any labels_parse() status
 */
app_status_t labels_load(const char *path, size_t expected_count, labels_t *out);

#endif /* LABELS_H */
