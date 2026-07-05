#ifndef FACENET_VECTORSEARCH_FACERECOGNIZER_H
#define FACENET_VECTORSEARCH_FACERECOGNIZER_H
#include "FaceDetector.h"
#include "VectorIndex.h"
#include "dlib/matrix/matrix.h"
#include "dlib/pixel.h"
#include "models/DetectedFaceEmbeddings.h"

#include <string>

class FaceRecognizer {

    FaceDetector faceDetector;
    FaceEmbedder faceEmbedder;
    VectorIndex vectorIndex;

    DetectedFaceEmbeddings detectFacesAndComputeEmbeddings(const std::vector<dlib::matrix<dlib::rgb_pixel>>& images,
                                                           bool returnBoundingBoxes = false);

  public:
    explicit FaceRecognizer(const std::string& dbFilePath, const std::string& faceNetModelPath)
        : faceEmbedder(faceNetModelPath), vectorIndex(dbFilePath) {}

    void insert(const std::string& personName, const std::vector<dlib::matrix<dlib::rgb_pixel>>& images);

    std::vector<NNQueryResult> recognize(const dlib::matrix<dlib::rgb_pixel>& image);
};

#endif // FACENET_VECTORSEARCH_FACERECOGNIZER_H
