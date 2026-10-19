#include <string.h>

#include "labels.h"
#include "test_util.h"

static labels_t s_labels;

static void test_plain(void)
{
    const char text[] = "cat\ndog\r\nbird";
    CHECK(labels_parse(text, strlen(text), 3U, &s_labels) == APP_OK);
    CHECK(s_labels.count == 3U);
    CHECK(strcmp(s_labels.name[1], "dog") == 0);  /* CRLF stripped */
    CHECK(strcmp(s_labels.name[2], "bird") == 0); /* last line without newline */
}

static void test_background_dropped(void)
{
    const char text[] = "background\ncat\ndog\n";
    CHECK(labels_parse(text, strlen(text), 2U, &s_labels) == APP_OK);
    CHECK(s_labels.count == 2U);
    CHECK(strcmp(s_labels.name[0], "cat") == 0);
}

static void test_count_mismatch(void)
{
    const char text[] = "cat\ndog\n";
    CHECK(labels_parse(text, strlen(text), 3U, &s_labels) == APP_ERR_FORMAT);
    /* An extra line that is not "background" must not be silently dropped. */
    const char text2[] = "x\ncat\ndog\n";
    CHECK(labels_parse(text2, strlen(text2), 2U, &s_labels) == APP_ERR_FORMAT);
}

static void test_args_and_files(void)
{
    CHECK(labels_parse(NULL, 0U, 1U, &s_labels) == APP_ERR_ARG);
    CHECK(labels_load("/nonexistent/labels.txt", 1U, &s_labels) == APP_ERR_IO);
    CHECK(labels_load(NULL, 1U, &s_labels) == APP_ERR_ARG);
}

int main(void)
{
    test_plain();
    test_background_dropped();
    test_count_mismatch();
    test_args_and_files();
    (void)puts(g_failures == 0 ? "test_labels: PASS" : "test_labels: FAIL");
    return TEST_RESULT();
}
