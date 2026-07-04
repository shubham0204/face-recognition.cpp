#include "FaceDetector.h"
#include "TestUtils.h"
#include <dlib/image_io.h>
#include <gtest/gtest.h>

TEST(FaceDetectorSmokeTests, LoadImage) {
    const auto imageFilePath = getResourcePath("img.png");
    dlib::matrix<dlib::rgb_pixel> inputImage;
    dlib::load_png(inputImage, imageFilePath);
    EXPECT_EQ(inputImage.nr(), 493);
    EXPECT_EQ(inputImage.nc(), 740);
}

TEST(FaceDetectorSmokeTests, DetectFaces) {
    const auto imageFilePath = getResourcePath("img.png");
    dlib::matrix<dlib::rgb_pixel> inputImage;
    dlib::load_png(inputImage, imageFilePath);

    FaceDetector faceDetector;
    const auto faces = faceDetector.detectFaces(inputImage);
    EXPECT_EQ(faces.size(), 1);
}

TEST(FaceDetectorSmokeTests, CropFaces) {
    const auto imageFilePath = getResourcePath("img.png");
    dlib::matrix<dlib::rgb_pixel> inputImage;
    dlib::load_png(inputImage, imageFilePath);

    FaceDetector faceDetector;
    const auto faces = faceDetector.detectFaces(inputImage);
    EXPECT_EQ(faces.size(), 1);

    std::vector<dlib::chip_details> chipDetails;
    chipDetails.reserve(faces.size());
for (const auto& face : faces) {
        chipDetails.emplace_back(face);
    }
    dlib::array<dlib::matrix<dlib::rgb_pixel>> croppedImages(chipDetails.size());
    dlib::extract_image_chips(inputImage, chipDetails, croppedImages);
    EXPECT_EQ(croppedImages.size(), faces.size());
}