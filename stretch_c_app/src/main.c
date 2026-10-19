/**
 * @file main.c
 * @brief Classify a folder of images with an int8 TFLite model on the CPU or the Ethos-U NPU.
 *
 * Same flags and output format as board/infer.py. The NPU path loads the Ethos-U external
 * delegate through the TFLite C API; the model must be the Vela-compiled one.
 */
#define _POSIX_C_SOURCE 200809L

#include <dirent.h>
#include <getopt.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <unistd.h>

#include "app_status.h"
#include "imgproc.h"
#include "labels.h"
#include "quant.h"
#include "stb_image.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/delegates/external/external_delegate.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

#define NUM_CLASSES 1000U
#define MAX_IMAGES 4096U
#define NAME_LEN 256U
#define PATH_LEN 512U
#define TOP5_TEXT_LEN 1024U
#define MIN_THREADS 1L
#define MAX_THREADS 8L
#define NS_PER_MS 1000000.0
#define NS_PER_S 1000000000LL
#define DEFAULT_LABELS "/usr/bin/tensorflow-lite-2.16.2/examples/labels.txt"

static const char *const DELEGATE_SEARCH[] = {"/usr/lib/libethosu_delegate.so",
                                              "/usr/lib64/libethosu_delegate.so"};

typedef struct
{
    const char *device; /* "cpu" or "npu" */
    const char *model;
    const char *images;
    const char *labels;
    const char *delegate;
    int threads;
    bool xnnpack;
} cli_args_t;

typedef struct
{
    TfLiteModel *model;
    TfLiteInterpreterOptions *options;
    TfLiteDelegate *delegate;
    bool delegate_is_xnnpack;
    TfLiteInterpreter *interp;
} engine_t;

static char s_names[MAX_IMAGES][NAME_LEN];
static labels_t s_labels;
static uint8_t s_crop[IMG_CROP_BYTES];
static int8_t s_input_q[IMG_CROP_BYTES];
static int8_t s_scores[NUM_CLASSES];

static void usage(const char *prog)
{
    (void)fprintf(stderr,
                  "usage: %s --device cpu|npu --model FILE --images DIR [--labels FILE]\n"
                  "          [--threads N] [--delegate LIB] [--xnnpack]\n",
                  prog);
}

static int64_t now_ns(void)
{
    struct timespec ts;
    (void)clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * NS_PER_S + (int64_t)ts.tv_nsec;
}

static app_status_t parse_args(int argc, char **argv, cli_args_t *args)
{
    static const struct option opts[] = {
        {"device", required_argument, NULL, 'd'},  {"model", required_argument, NULL, 'm'},
        {"images", required_argument, NULL, 'i'},  {"labels", required_argument, NULL, 'l'},
        {"threads", required_argument, NULL, 't'}, {"delegate", required_argument, NULL, 'e'},
        {"xnnpack", no_argument, NULL, 'x'},       {NULL, 0, NULL, 0}};
    args->labels = DEFAULT_LABELS;
    args->threads = 1;
    int c;
    while ((c = getopt_long(argc, argv, "", opts, NULL)) != -1)
    {
        char *end = NULL;
        switch (c)
        {
        case 'd':
            args->device = optarg;
            break;
        case 'm':
            args->model = optarg;
            break;
        case 'i':
            args->images = optarg;
            break;
        case 'l':
            args->labels = optarg;
            break;
        case 'e':
            args->delegate = optarg;
            break;
        case 'x':
            args->xnnpack = true;
            break;
        case 't':
        {
            const long v = strtol(optarg, &end, 10);
            if (*optarg == '\0' || *end != '\0' || v < MIN_THREADS || v > MAX_THREADS)
            {
                return APP_ERR_ARG;
            }
            args->threads = (int)v;
            break;
        }
        default:
            return APP_ERR_ARG;
        }
    }
    const bool device_ok = args->device != NULL &&
                           (strcmp(args->device, "cpu") == 0 || strcmp(args->device, "npu") == 0);
    return (device_ok && args->model != NULL && args->images != NULL) ? APP_OK : APP_ERR_ARG;
}

static bool has_image_suffix(const char *name)
{
    const char *dot = strrchr(name, '.');
    return dot != NULL && (strcasecmp(dot, ".jpg") == 0 || strcasecmp(dot, ".jpeg") == 0 ||
                           strcasecmp(dot, ".png") == 0);
}

static int compare_names(const void *a, const void *b)
{
    return strcmp((const char *)a, (const char *)b);
}

/* Fills s_names with the sorted image file names of dir; returns the count through *count. */
static app_status_t list_images(const char *dir, size_t *count)
{
    DIR *d = opendir(dir);
    if (d == NULL)
    {
        return APP_ERR_IO;
    }
    size_t n = 0U;
    app_status_t st = APP_OK;
    const struct dirent *e;
    while ((e = readdir(d)) != NULL)
    {
        if (!has_image_suffix(e->d_name))
        {
            continue;
        }
        const size_t name_len = strlen(e->d_name);
        if (n >= MAX_IMAGES || name_len >= NAME_LEN)
        {
            st = APP_ERR_RANGE;
            break;
        }
        memcpy(s_names[n], e->d_name, name_len + 1U); /* length checked above */
        n++;
    }
    (void)closedir(d);
    qsort(s_names, n, NAME_LEN, compare_names);
    *count = n;
    return (st == APP_OK && n == 0U) ? APP_ERR_IO : st;
}

static const char *find_delegate(const char *explicit_path)
{
    if (explicit_path != NULL)
    {
        return explicit_path;
    }
    for (size_t i = 0U; i < sizeof(DELEGATE_SEARCH) / sizeof(DELEGATE_SEARCH[0]); i++)
    {
        if (access(DELEGATE_SEARCH[i], R_OK) == 0)
        {
            return DELEGATE_SEARCH[i];
        }
    }
    return NULL;
}

static void engine_close(engine_t *eng)
{
    /* Interpreter first, then what it references. */
    if (eng->interp != NULL)
    {
        TfLiteInterpreterDelete(eng->interp);
    }
    if (eng->delegate != NULL)
    {
        if (eng->delegate_is_xnnpack)
        {
            TfLiteXNNPackDelegateDelete(eng->delegate);
        }
        else
        {
            TfLiteExternalDelegateDelete(eng->delegate);
        }
    }
    if (eng->options != NULL)
    {
        TfLiteInterpreterOptionsDelete(eng->options);
    }
    if (eng->model != NULL)
    {
        TfLiteModelDelete(eng->model);
    }
    memset(eng, 0, sizeof(*eng));
}

static app_status_t engine_add_delegate(engine_t *eng, const char *explicit_path)
{
    const char *lib = find_delegate(explicit_path);
    if (lib == NULL)
    {
        (void)fprintf(stderr, "Ethos-U delegate not found; pass --delegate\n");
        return APP_ERR_IO;
    }
    TfLiteExternalDelegateOptions dopts = TfLiteExternalDelegateOptionsDefault(lib);
    eng->delegate = TfLiteExternalDelegateCreate(&dopts);
    if (eng->delegate == NULL)
    {
        return APP_ERR_TFLITE;
    }
    TfLiteInterpreterOptionsAddDelegate(eng->options, eng->delegate);
    return APP_OK;
}

static app_status_t engine_add_xnnpack(engine_t *eng, int threads)
{
    TfLiteXNNPackDelegateOptions xopts = TfLiteXNNPackDelegateOptionsDefault();
    xopts.num_threads = threads;
    eng->delegate = TfLiteXNNPackDelegateCreate(&xopts);
    if (eng->delegate == NULL)
    {
        return APP_ERR_TFLITE;
    }
    eng->delegate_is_xnnpack = true;
    TfLiteInterpreterOptionsAddDelegate(eng->options, eng->delegate);
    return APP_OK;
}

static app_status_t engine_open(engine_t *eng, const cli_args_t *args)
{
    memset(eng, 0, sizeof(*eng));
    eng->model = TfLiteModelCreateFromFile(args->model);
    eng->options = TfLiteInterpreterOptionsCreate();
    if (eng->model == NULL || eng->options == NULL)
    {
        return APP_ERR_TFLITE;
    }
    if (strcmp(args->device, "npu") == 0)
    {
        const app_status_t st = engine_add_delegate(eng, args->delegate);
        if (st != APP_OK)
        {
            return st;
        }
    }
    else
    {
        TfLiteInterpreterOptionsSetNumThreads(eng->options, args->threads);
        if (args->xnnpack)
        {
            const app_status_t st = engine_add_xnnpack(eng, args->threads);
            if (st != APP_OK)
            {
                return st;
            }
        }
    }
    eng->interp = TfLiteInterpreterCreate(eng->model, eng->options);
    if (eng->interp == NULL || TfLiteInterpreterAllocateTensors(eng->interp) != kTfLiteOk)
    {
        return APP_ERR_TFLITE;
    }
    return APP_OK;
}

/* The app assumes a single int8 224x224x3 input and 1000 int8 outputs. */
static app_status_t check_tensors(const engine_t *eng)
{
    if (TfLiteInterpreterGetInputTensorCount(eng->interp) != 1 ||
        TfLiteInterpreterGetOutputTensorCount(eng->interp) != 1)
    {
        return APP_ERR_FORMAT;
    }
    const TfLiteTensor *in = TfLiteInterpreterGetInputTensor(eng->interp, 0);
    const TfLiteTensor *out = TfLiteInterpreterGetOutputTensor(eng->interp, 0);
    const bool ok = TfLiteTensorType(in) == kTfLiteInt8 && TfLiteTensorType(out) == kTfLiteInt8 &&
                    TfLiteTensorByteSize(in) == IMG_CROP_BYTES &&
                    TfLiteTensorByteSize(out) == NUM_CLASSES;
    return ok ? APP_OK : APP_ERR_FORMAT;
}

/* Decode, preprocess and quantize one image into s_input_q. */
static app_status_t prepare_input(const char *dir, const char *name, const engine_t *eng)
{
    char path[PATH_LEN];
    if (snprintf(path, sizeof(path), "%s/%s", dir, name) >= (int)sizeof(path))
    {
        return APP_ERR_RANGE;
    }
    int w = 0;
    int h = 0;
    int channels = 0;
    unsigned char *pixels = stbi_load(path, &w, &h, &channels, (int)IMG_CHANNELS);
    if (pixels == NULL || w <= 0 || h <= 0)
    {
        stbi_image_free(pixels);
        return APP_ERR_IO;
    }
    app_status_t st = img_preprocess(pixels, (uint32_t)w, (uint32_t)h, s_crop);
    stbi_image_free(pixels);
    if (st != APP_OK)
    {
        return st;
    }
    const TfLiteTensor *in = TfLiteInterpreterGetInputTensor(eng->interp, 0);
    const TfLiteQuantizationParams q = TfLiteTensorQuantizationParams(in);
    return quant_buffer(s_crop, IMG_CROP_BYTES, q.scale, q.zero_point, s_input_q);
}

/* Runs one inference on s_input_q; returns scores and the invoke time in milliseconds. */
static app_status_t run_inference(const engine_t *eng, int8_t *scores, double *ms)
{
    TfLiteTensor *in = TfLiteInterpreterGetInputTensor(eng->interp, 0);
    if (TfLiteTensorCopyFromBuffer(in, s_input_q, IMG_CROP_BYTES) != kTfLiteOk)
    {
        return APP_ERR_TFLITE;
    }
    const int64_t t0 = now_ns();
    const TfLiteStatus st = TfLiteInterpreterInvoke(eng->interp);
    *ms = (double)(now_ns() - t0) / NS_PER_MS;
    if (st != kTfLiteOk)
    {
        return APP_ERR_TFLITE;
    }
    const TfLiteTensor *out = TfLiteInterpreterGetOutputTensor(eng->interp, 0);
    return (TfLiteTensorCopyToBuffer(out, scores, NUM_CLASSES) == kTfLiteOk) ? APP_OK
                                                                             : APP_ERR_TFLITE;
}

static void format_top5(char *buf, size_t size, const int8_t *scores, const size_t *idx)
{
    size_t used = 0U;
    buf[0] = '\0';
    for (size_t r = 0U; r < TOP_K && used < size; r++)
    {
        const int n = snprintf(&buf[used], size - used, "%s%s(%d)", (r == 0U) ? "" : ", ",
                               s_labels.name[idx[r]], (int)scores[idx[r]]);
        if (n < 0 || (size_t)n >= size - used)
        {
            break; /* truncated: acceptable for a display string */
        }
        used += (size_t)n;
    }
}

static app_status_t classify_one(const cli_args_t *args, const engine_t *eng, const char *name)
{
    size_t idx[TOP_K];
    double ms = 0.0;
    app_status_t st = prepare_input(args->images, name, eng);
    if (st == APP_OK)
    {
        st = run_inference(eng, s_scores, &ms);
    }
    if (st == APP_OK)
    {
        st = topk_i8(s_scores, NUM_CLASSES, TOP_K, idx);
    }
    if (st != APP_OK)
    {
        (void)fprintf(stderr, "%s: failed (status %d)\n", name, (int)st);
        return st;
    }
    char top5[TOP5_TEXT_LEN];
    format_top5(top5, sizeof(top5), s_scores, idx);
    (void)printf("%s\t%.3f ms\ttop1=%s\ttop1_idx=%zu\ttop5=%s\n", name, ms, s_labels.name[idx[0]],
                 idx[0], top5);
    return APP_OK;
}

/* First inference is reported separately so first-run setup cost is not mixed into the table. */
static app_status_t warmup(const cli_args_t *args, const engine_t *eng, const char *name)
{
    double ms = 0.0;
    app_status_t st = prepare_input(args->images, name, eng);
    if (st == APP_OK)
    {
        st = run_inference(eng, s_scores, &ms);
    }
    if (st == APP_OK)
    {
        (void)printf("# warmup_ms=%.3f device=%s\n", ms, args->device);
    }
    return st;
}

static app_status_t run_all(const cli_args_t *args, size_t image_count)
{
    engine_t eng;
    app_status_t st = engine_open(&eng, args);
    if (st == APP_OK)
    {
        st = check_tensors(&eng);
    }
    if (st == APP_OK)
    {
        st = warmup(args, &eng, s_names[0]);
    }
    for (size_t i = 0U; st == APP_OK && i < image_count; i++)
    {
        st = classify_one(args, &eng, s_names[i]);
    }
    engine_close(&eng);
    return st;
}

int main(int argc, char **argv)
{
    cli_args_t args = {0};
    if (parse_args(argc, argv, &args) != APP_OK)
    {
        usage(argv[0]);
        return EXIT_FAILURE;
    }
    app_status_t st = labels_load(args.labels, NUM_CLASSES, &s_labels);
    size_t count = 0U;
    if (st == APP_OK)
    {
        st = list_images(args.images, &count);
    }
    if (st == APP_OK)
    {
        st = run_all(&args, count);
    }
    if (st != APP_OK)
    {
        (void)fprintf(stderr, "error: status %d\n", (int)st);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
