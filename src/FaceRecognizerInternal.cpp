#include "FaceRecognizerInternal.h"

#include "DetectedFaceEmbeddings.h"
#include "FaceDetector.h"
#include "Utils.h"

void FaceRecognizerInternal::insert(const std::string& personName, const std::vector<dlib::matrix<dlib::rgb_pixel>>& images) {
    const auto [embeddings, _] = this->detectFacesAndComputeEmbeddings(images, false);
    for (const auto& embedding : embeddings) {
        vectorIndex.insert(personName, embedding);
    }
}

void FaceRecognizerInternal::insert(const std::string& personName, const std::vector<std::string>& imageFilePaths) {
    std::vector<dlib::matrix<dlib::rgb_pixel>> dlibMatrixImages;
    dlibMatrixImages.reserve(imageFilePaths.size());
    for (const auto& filePath : imageFilePaths) {
        dlibMatrixImages.push_back(Utils::loadImageFromFile(filePath));
    }
    this->insert(personName, dlibMatrixImages);
}

void FaceRecognizerInternal::insert(const std::string& personName, const std::vector<IntBufferImage>& images) {
    std::vector<dlib::matrix<dlib::rgb_pixel>> dlibMatrixImages;
    dlibMatrixImages.reserve(images.size());
    for (const auto& [buffer, width, height] : images) {
        dlibMatrixImages.push_back(Utils::loadImageFromIntBuffer(buffer, width, height));
    }
    this->insert(personName, dlibMatrixImages);
}
std::vector<NNQueryResult> FaceRecognizerInternal::recognize(const std::string& imageFilePath) {
    return this->recognize(Utils::loadImageFromFile(imageFilePath));
}

std::vector<NNQueryResult> FaceRecognizerInternal::recognize(const IntBufferImage& image) {
    return this->recognize(Utils::loadImageFromIntBuffer(image.buffer, image.width, image.height));
}

std::vector<NNQueryResult> FaceRecognizerInternal::recognize(const dlib::matrix<dlib::rgb_pixel>& image) {
    const auto images = {image};
    const auto [embeddings, faceBoundingBoxes] = detectFacesAndComputeEmbeddings(images, true);
    std::vector<NNQueryResult> results;
    for (int i = 0; i < embeddings.size(); i++) {
        auto nnResult = vectorIndex.nearestNeighbor(embeddings[i]);
        nnResult.faceBoundingBox = Utils::fromDlibRectangle(faceBoundingBoxes[i]);
        results.push_back(nnResult);
    }
    return results;
}

DetectedFaceEmbeddings FaceRecognizerInternal::detectFacesAndComputeEmbeddings(const std::vector<dlib::matrix<dlib::rgb_pixel>>& images,
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