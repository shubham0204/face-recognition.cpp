#include "FaceDetector.h"

#include "FaceEmbedder.h"
#include <gtest/gtest.h>
#include "TestUtils.h"
#include "dlib/image_loader/png_loader.h"

TEST(FaceEmbedderSmokeTests, CreateEmbeddings) {
    const auto imageFilePath = getResourcePath("img.png");
    DlibRgbImage inputImage;
    dlib::load_png(inputImage, imageFilePath);

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

    EXPECT_EQ(embeddings.size(), 1);
}