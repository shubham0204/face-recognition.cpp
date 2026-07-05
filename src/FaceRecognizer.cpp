#include "FaceRecognizer.h"

#include "FaceDetector.h"
#include "models/DetectedFaceEmbeddings.h"

void FaceRecognizer::insert(const std::string& personName, const std::vector<dlib::matrix<dlib::rgb_pixel>>& images) {
    const auto [embeddings, _] = detectFacesAndComputeEmbeddings(images, false);
    for (const auto& embedding : embeddings) {
        vectorIndex.insert(personName, embedding);
    }
}

std::vector<NNQueryResult> FaceRecognizer::recognize(const dlib::matrix<dlib::rgb_pixel>& image) {
    const auto images = {image};
    const auto [embeddings, faceBoundingBoxes] = detectFacesAndComputeEmbeddings(images, true);
    std::vector<NNQueryResult> results;
    for (int i = 0; i < embeddings.size(); i++) {
        auto nnResult = vectorIndex.nearestNeighbor(embeddings[i]);
        nnResult.boundingBox = faceBoundingBoxes[i];
        results.push_back(nnResult);
    }
    return results;
}

DetectedFaceEmbeddings FaceRecognizer::detectFacesAndComputeEmbeddings(const std::vector<dlib::matrix<dlib::rgb_pixel>>& images,
                                                                       const bool returnBoundingBoxes) {
    std::vector<dlib::matrix<dlib::rgb_pixel>> croppedImages;
    std::vector<dlib::rectangle> boundingBoxes;
    for (const auto& image : images) {
        const auto detectedFaces = faceDetector.detectFaces(image);
        const auto croppedFaces = FaceDetector::cropFaces(image, detectedFaces);
        std::copy_n(croppedFaces.begin(), croppedFaces.size(), std::back_inserter(croppedImages));
        std::copy_n(detectedFaces.begin(), detectedFaces.size(), std::back_inserter(boundingBoxes));
    }
    const std::vector<std::array<float, EMBEDDING_DIM>> embeddings = faceEmbedder.computeFaceEmbedding(croppedImages);
    DetectedFaceEmbeddings detectedFaceEmbeddings;
    detectedFaceEmbeddings.embeddings = embeddings;
    if (returnBoundingBoxes) {
        detectedFaceEmbeddings.faceBoundingBoxes = boundingBoxes;
    }
    return detectedFaceEmbeddings;
}