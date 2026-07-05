#ifndef FACENET_VECTORSEARCH_FACERECOGNIZER_H
#define FACENET_VECTORSEARCH_FACERECOGNIZER_H
#include "FaceDetector.h"
#include "VectorIndex.h"
#include "dlib/matrix/matrix.h"
#include "dlib/pixel.h"

#include <string>

class FaceRecognizer {

    FaceDetector faceDetector;
    FaceEmbedder faceEmbedder;
    VectorIndex vectorIndex;

public:
    explicit FaceRecognizer(const std::string& dbFilePath, const std::string& faceNetModelPath): vectorIndex(dbFilePath), faceEmbedder(faceNetModelPath) {
    }

    void insert(const std::string& personName, const std::vector<dlib::matrix<dlib::rgb_pixel>>& images);
    std::vector<std::array<float, 512>> detectFacesAndComputeEmbeddings(const std::vector<dlib::matrix<dlib::rgb_pixel>>& images);

    std::vector<NNQueryResult> recognize(const std::vector<dlib::matrix<dlib::rgb_pixel>>& images);
};

#endif // FACENET_VECTORSEARCH_FACERECOGNIZER_H

