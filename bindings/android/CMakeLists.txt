# ============= facenet_vectorsearch_jni library target configuration =============
find_package(JNI REQUIRED)
set(JNI_TARGET_NAME facenet_vectorsearch_jni)
add_library(
        ${JNI_TARGET_NAME}
        SHARED
        binding.cpp
)
target_link_libraries(
        ${JNI_TARGET_NAME}
        JNI::JNI
        facenet_vectorsearch
)