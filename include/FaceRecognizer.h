#ifndef FACENET_VECTORSEARCH_API_H
#define FACENET_VECTORSEARCH_API_H

#include "IntBufferImage.h"
#include "NNQueryResult.h"

#include <memory>
#include <string>
#include <vector>

class FaceRecognizer {
  public:
    virtual ~FaceRecognizer() = default;

    virtual void insert(const std::string& personName, const std::vector<std::string>& imageFilePaths) = 0;

    virtual void insert(const std::string& personName, const std::vector<IntBufferImage>& images) = 0;

    virtual std::vector<NNQueryResult> recognize(const std::string& imageFilePath) = 0;

    virtual std::vector<NNQueryResult> recognize(const IntBufferImage& image) = 0;
};

std::unique_ptr<FaceRecognizer> createFaceRecognizer(const std::string& dbFilePath, const std::string& faceNetModelPath);

#endif // FACENET_VECTORSEARCH_API_H
