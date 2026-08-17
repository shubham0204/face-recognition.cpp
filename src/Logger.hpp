#ifndef FACENET_VECTORSEARCH_LOGGER_H
#define FACENET_VECTORSEARCH_LOGGER_H
#ifdef IS_ANDROID
#include <android/log.h>
#endif
#include <iostream>

class Logger {
  public:
    static void log(const std::string& message) {
#ifndef NDEBUG
#ifdef IS_ANDROID
        __android_log_print(ANDROID_LOG_INFO, "[FaceRecognitionCpp]", "%s", message.c_str());
#else
        std::cout << "[FaceRecognitionCpp] " << message << '\n';
#endif
#endif
    }
};

#endif // FACENET_VECTORSEARCH_LOGGER_H
