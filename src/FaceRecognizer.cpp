#include "FaceRecognizer.hpp"
#include "FaceRecognizerInternal.hpp"

std::unique_ptr<FaceRecognizer> createFaceRecognizer(const std::string& dbFilePath, const std::string& faceNetModelPath) {
    return std::make_unique<FaceRecognizerInternal>(dbFilePath, faceNetModelPath);
}