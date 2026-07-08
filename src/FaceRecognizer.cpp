#include "FaceRecognizer.h"
#include "FaceRecognizerInternal.h"

std::unique_ptr<FaceRecognizer> createFaceRecognizer(const std::string& dbFilePath, const std::string& faceNetModelPath) {
    return std::make_unique<FaceRecognizerInternal>(dbFilePath, faceNetModelPath);
}