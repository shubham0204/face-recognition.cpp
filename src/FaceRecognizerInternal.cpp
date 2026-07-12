#include "FaceRecognizerInternal.h"

#include "DetectedFaceEmbeddings.h"
#include "FaceDetector.h"
#include "Utils.h"

void FaceRecognizerInternal::insert(const std::string& personName, const std::vector<DlibRgbImage>& images) {
    const auto faceDetectionResult = this->detectFacesAndComputeEmbeddings(images, false);
    if (!faceDetectionResult.has_value()) {
        return;
    }
    for (const auto& embedding : faceDetectionResult->embeddings) {
        vectorIndex.insert(personName, embedding);
    }
}

void FaceRecognizerInternal::insert(const std::string& personName, const std::vector<std::string>& imageFilePaths) {
    std::vector<DlibRgbImage> dlibMatrixImages;
    dlibMatrixImages.reserve(imageFilePaths.size());
    for (const auto& filePath : imageFilePaths) {
        dlibMatrixImages.push_back(Utils::loadImageFromFile(filePath));
    }
    this->insert(personName, dlibMatrixImages);
}

void FaceRecognizerInternal::insert(const std::string& personName, const std::vector<IntBufferImage>& images) {
    std::vector<DlibRgbImage> dlibMatrixImages;
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

bool FaceRecognizerInternal::remove(const std::string& personName) { return this->vectorIndex.remove(personName); }

void FaceRecognizerInternal::clear() { return this->vectorIndex.clear(); }

std::vector<NNQueryResult> FaceRecognizerInternal::recognize(const DlibRgbImage& image) {
    const auto faceDetectionResult = detectFacesAndComputeEmbeddings({image}, true);
    if (!faceDetectionResult.has_value()) {
        return {};
    }
    const auto [embeddings, faceBoundingBoxes] = faceDetectionResult.value();
    std::vector<NNQueryResult> results(embeddings.size());
    for (int i = 0; i < embeddings.size(); i++) {
        auto nnResult = vectorIndex.nearestNeighbor(embeddings[i]);
        nnResult.faceBoundingBox = Utils::fromDlibRectangle(faceBoundingBoxes[i]);
        results[i] = nnResult;
    }
    return results;
}

std::optional<DetectedFaceEmbeddings>
FaceRecognizerInternal::detectFacesAndComputeEmbeddings(const std::vector<DlibRgbImage>& images,
                                                                                              const bool returnBoundingBoxes) {
    std::vector<DlibRgbImage> croppedImages;
    std::vector<dlib::rectangle> boundingBoxes;
    for (const auto& image : images) {
        const auto detectedFaces = faceDetector.detectFaces(image);
        const auto croppedFaces = FaceDetector::cropFaces(image, detectedFaces);
        std::copy_n(croppedFaces.begin(), croppedFaces.size(), std::back_inserter(croppedImages));
        std::copy_n(detectedFaces.begin(), detectedFaces.size(), std::back_inserter(boundingBoxes));
    }
    const std::vector<Embedding> embeddings = faceEmbedder.computeFaceEmbedding(croppedImages);
    DetectedFaceEmbeddings detectedFaceEmbeddings;
    detectedFaceEmbeddings.embeddings = embeddings;
    if (returnBoundingBoxes) {
        detectedFaceEmbeddings.faceBoundingBoxes = boundingBoxes;
    }
    return detectedFaceEmbeddings;
};