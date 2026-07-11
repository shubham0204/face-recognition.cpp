#include "FaceDetector.h"

#include "FaceEmbedder.h"
#include <gtest/gtest.h>
#include "TestUtils.h"
#include "dlib/image_loader/png_loader.h"

static std::vector<Embedding> getEmbeddingFromFace(const std::string& resourcePath) {
    const auto imageFilePath = getResourcePath(resourcePath);
    DlibRgbImage inputImage;
    dlib::load_image(inputImage, imageFilePath);

    FaceDetector faceDetector;
    const auto faces = faceDetector.detectFaces(inputImage);

    std::vector<dlib::chip_details> chipDetails;
    chipDetails.reserve(faces.size());
    for (const auto& face : faces) {
        chipDetails.emplace_back(face);
    }
    dlib::array<DlibRgbImage> croppedImagesArray(chipDetails.size());
    dlib::extract_image_chips(inputImage, chipDetails, croppedImagesArray);
    std::vector<DlibRgbImage> croppedImages;
    for (const auto& face : croppedImagesArray) {
        croppedImages.push_back(face);
    }

    FaceEmbedder faceEmbedder(getResourcePath("model.pte"));
    const auto embeddings = faceEmbedder.computeFaceEmbedding(croppedImages);
    return embeddings;
};

TEST(FaceEmbedderSmokeTests, CreateEmbeddings) {
    const auto embeddings = getEmbeddingFromFace("img.png");
    EXPECT_EQ(embeddings.size(), 1);
}

TEST(FaceEmbedderSmokeTests, DifferentImagesProduceDifferentEmbeddings) {
    const auto embedding1 = getEmbeddingFromFace("Steven_Hatfill_0002.jpg")[0];
    const auto embedding2 = getEmbeddingFromFace("Steven_Hatfill_0001.jpg")[0];
    const auto embedding3 = getEmbeddingFromFace("Sue_Grafton_0001.jpg")[0];
    const double p1 = l2Norm(embedding1, embedding2);
    const double p2 = l2Norm(embedding1, embedding3);
    EXPECT_TRUE(p2 > p1);
}