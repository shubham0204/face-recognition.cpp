# ============ GoogleTest configuration ================
set(TEST_TARGET_NAME facenet_vectorsearch_tests)
enable_testing()
add_executable(
        ${TEST_TARGET_NAME}
        # add the sources of the facenet_vectorsearch library target
        # manually, to avoid re-linking executorch
        ${CMAKE_SOURCE_DIR}/src/FaceEmbedder.cpp
        ${CMAKE_SOURCE_DIR}/src/FaceDetector.cpp
        ${CMAKE_SOURCE_DIR}/src/FaceRecognizer.cpp
        ${CMAKE_SOURCE_DIR}/src/VectorIndex.cpp
        FaceDetectorTests.cpp
        TestUtils.h
        FaceEmbedderTests.cpp
        VectorDBTests.cpp
        FaceRecognizerTests.cpp
)
target_link_libraries(
        ${TEST_TARGET_NAME}
        GTest::gtest
        GTest::gtest_main
        executorch
        executorch::backends
        executorch::extensions
        executorch::kernels
        dlib::dlib
)
target_include_directories(
        ${TEST_TARGET_NAME}
        PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)
target_compile_definitions(
        ${TEST_TARGET_NAME}
        PRIVATE
        TEST_RESOURCE_DIR="${CMAKE_SOURCE_DIR}/tests/resources/"
)
include(GoogleTest)
gtest_discover_tests(${TEST_TARGET_NAME})