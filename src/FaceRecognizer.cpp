#include "FaceRecognizer.h"

#include "FaceDetector.h"

void FaceRecognizer::insert(const std::string& personName, const std::vector<dlib::matrix<dlib::rgb_pixel>>& images) {
    std::vector<std::array<float, EMBEDDING_DIM>> embeddings = detectFacesAndComputeEmbeddings(images);
    for (const auto& embedding : embeddings) {
        vectorIndex.insert(personName, embedding);
    }
}

std::vector<NNQueryResult> FaceRecognizer::recognize(const std::vector<dlib::matrix<dlib::rgb_pixel>>& images) {
    std::vector<std::array<float, 512>> embeddings = detectFacesAndComputeEmbeddings(images);
    std::vector<NNQueryResult> results;
    for (const auto& embedding : embeddings) {
        results.push_back(vectorIndex.nearestNeighbor(embedding));
    }
    return results;
}

std::vector<std::array<float, 512>> FaceRecognizer::detectFacesAndComputeEmbeddings(const std::vector<dlib::matrix<dlib::rgb_pixel>>& images) {
    std::vector<dlib::matrix<dlib::rgb_pixel>> croppedImages;
    for (const auto& image : images) {
        const auto detectedFaces = faceDetector.detectFaces(image);
        const auto croppedFaces = FaceDetector::cropFaces(image, detectedFaces);
        std::copy_n(croppedFaces.begin(), croppedFaces.size(), std::back_inserter(croppedImages));
    }
    std::vector<std::array<float, EMBEDDING_DIM>> embeddings = faceEmbedder.computeFaceEmbedding(croppedImages);
    return embeddings;
}