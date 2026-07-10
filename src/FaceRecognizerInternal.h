#ifndef FACENET_VECTORSEARCH_FACERECOGNIZER_H
#define FACENET_VECTORSEARCH_FACERECOGNIZER_H
#include "../include/IntBufferImage.h"
#include "../include/NNQueryResult.h"
#include "DetectedFaceEmbeddings.h"
#include "FaceDetector.h"
#include "FaceRecognizer.h"
#include "VectorIndex.h"
#include "dlib/matrix/matrix.h"
#include "dlib/pixel.h"

#include <string>

class FaceRecognizerInternal : public FaceRecognizer {

    FaceDetector faceDetector;
    FaceEmbedder faceEmbedder;
    VectorIndex vectorIndex;

    std::optional<DetectedFaceEmbeddings> detectFacesAndComputeEmbeddings(const std::vector<dlib::matrix<dlib::rgb_pixel>>& images,
                                                                          bool returnBoundingBoxes = false);

    void insert(const std::string& personName, const std::vector<dlib::matrix<dlib::rgb_pixel>>& images);

    std::vector<NNQueryResult> recognize(const dlib::matrix<dlib::rgb_pixel>& image);

  public:
    explicit FaceRecognizerInternal(const std::string& dbFilePath, const std::string& faceNetModelPath)
        : faceEmbedder(faceNetModelPath), vectorIndex(dbFilePath) {}

    void insert(const std::string& personName, const std::vector<std::string>& imageFilePaths) override;

    void insert(const std::string& personName, const std::vector<IntBufferImage>& images) override;

    std::vector<NNQueryResult> recognize(const std::string& imageFilePath) override;

    std::vector<NNQueryResult> recognize(const IntBufferImage& image) override;
};

#endif // FACENET_VECTORSEARCH_FACERECOGNIZER_H
