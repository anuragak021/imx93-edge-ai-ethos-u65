cmake_minimum_required(VERSION 3.16)
project(imx93_infer C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(BUILD_APP "Build the TFLite inference app (needs libtensorflow-lite and C API headers)" ON)
option(BUILD_TESTS "Build host unit tests" ON)

set(STRICT_WARNINGS -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wdouble-promotion -Werror)

# Pure logic: no TFLite, no hardware. Builds and tests on any host.
add_library(core STATIC src/imgproc.c src/quant.c src/labels.c)
target_include_directories(core PUBLIC include)
target_compile_options(core PRIVATE ${STRICT_WARNINGS})
target_link_libraries(core PUBLIC m)

if(BUILD_TESTS)
    enable_testing()
    foreach(t test_imgproc test_quant test_labels)
        add_executable(${t} tests/${t}.c)
        target_link_libraries(${t} PRIVATE core)
        target_compile_options(${t} PRIVATE ${STRICT_WARNINGS})
        add_test(NAME ${t} COMMAND ${t})
    endforeach()
endif()

if(BUILD_APP)
    set(TFLITE_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tflite" CACHE PATH "TFLite C API headers")
    set(STB_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb" CACHE PATH "stb_image.h directory")
    # The board ships only the versioned library file.
    find_library(TFLITE_LIB NAMES tensorflow-lite libtensorflow-lite.so.2.16.2 REQUIRED)

    add_library(stb_impl OBJECT src/stb_impl.c)
    target_include_directories(stb_impl SYSTEM PRIVATE ${STB_HEADERS})
    target_compile_options(stb_impl PRIVATE -w)

    add_executable(imx93_infer src/main.c $<TARGET_OBJECTS:stb_impl>)
    target_include_directories(imx93_infer SYSTEM PRIVATE ${TFLITE_HEADERS} ${STB_HEADERS})
    target_compile_options(imx93_infer PRIVATE ${STRICT_WARNINGS})
    target_link_libraries(imx93_infer PRIVATE core ${TFLITE_LIB})
endif()
