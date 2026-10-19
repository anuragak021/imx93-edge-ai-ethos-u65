/**
 * @file app_status.h
 * @brief Status codes shared by all modules of the C inference app.
 */
#ifndef APP_STATUS_H
#define APP_STATUS_H

/** Result of every fallible function in the app. APP_OK is zero. */
typedef enum
{
    APP_OK = 0,     /**< success */
    APP_ERR_ARG,    /**< invalid argument or null pointer */
    APP_ERR_IO,     /**< file or directory could not be read */
    APP_ERR_FORMAT, /**< input data has an unexpected format */
    APP_ERR_RANGE,  /**< a size or value is outside the supported bounds */
    APP_ERR_TFLITE  /**< TensorFlow Lite or delegate failure */
} app_status_t;

#endif /* APP_STATUS_H */
