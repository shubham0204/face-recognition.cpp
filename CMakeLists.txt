# ============= CMake project configuration =============
cmake_minimum_required(VERSION 4.2)
project(facenet_vectorsearch)
set(CMAKE_CXX_STANDARD 23)
# Build Options
option(BUILD_EXAMPLES "Build examples" OFF)
option(BUILD_JNI_BINDING "Build JNI binding" OFF)
option(BUILD_TESTS "Build unit tests" OFF)

# ============= External Dependencies =============
# ExecuTorch (ML runtime)
set(EXECUTORCH_BUILD_XNNPACK ON)
set(EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP ON)
set(EXECUTORCH_BUILD_EXTENSION_MODULE ON)
set(EXECUTORCH_BUILD_EXTENSION_TENSOR ON)
add_subdirectory(external/executorch)
# dlib (Face Detection)
set(DLIB_NO_GUI_SUPPORT OFF CACHE BOOL "Enable GUI support" FORCE)
add_subdirectory(external/dlib)
# GoogleTest (unit testing framework)
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

# ============= facenet_vectorsearch library target configuration ======================
add_library(facenet_vectorsearch SHARED
        src/FaceDetector.cpp
        src/FaceDetector.h
        src/FaceEmbedder.cpp
        src/FaceEmbedder.h
        src/VectorIndex.cpp
        src/VectorIndex.h
        src/FaceRecognizer.cpp
        src/FaceRecognizer.h
        src/models/DetectedFaceEmbeddings.h
)
target_link_libraries(
        facenet_vectorsearch
        PRIVATE
        executorch
        executorch::backends
        executorch::extensions
        executorch::kernels
        extension_module
        dlib::dlib
)
include_directories(include)

# ============ Project components =============
# Examples
if (BUILD_EXAMPLES)
    message(STATUS "Building examples...")
    add_subdirectory(examples)
endif ()
# Unit tests
if (BUILD_TESTS)
    message(STATUS "Building unit tests...")
    add_subdirectory(tests)
endif ()
# JNI binding
if (BUILD_JNI_BINDING)
    message(STATUS "Building JNI binding...")
    add_subdirectory(jni)
endif ()