#include "FaceDetector.h"

std::vector<dlib::rectangle> FaceDetector::detectFaces(const DlibRgbImage& inputImage) {
    dlib::matrix<unsigned char> grayImage;
    dlib::assign_image(grayImage, inputImage);
    std::vector<dlib::rectangle> faces = detector(grayImage);
    return faces;
}
dlib::array<DlibRgbImage> FaceDetector::cropFaces(const DlibRgbImage& inputImage, const std::vector<dlib::rectangle>& rectangles) {
    std::vector<dlib::chip_details> chipDetails;
    chipDetails.reserve(rectangles.size());
    for (const auto& face : rectangles) {
        chipDetails.emplace_back(face);
    }
    dlib::array<DlibRgbImage> croppedImages(chipDetails.size());
    dlib::extract_image_chips(inputImage, chipDetails, croppedImages);
    return croppedImages;
}