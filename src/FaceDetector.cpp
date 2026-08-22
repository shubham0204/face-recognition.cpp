#include "FaceDetector.hpp"

std::vector<dlib::rectangle> FaceDetector::detectFaces(const DlibRgbImage& inputImage) {
    dlib::matrix<unsigned char> grayImage;
    dlib::assign_image(grayImage, inputImage);

    const long newRows = grayImage.nr() / FACE_DETECTION_IMG_RESIZE_FACTOR;
    const long newCols = grayImage.nc() / FACE_DETECTION_IMG_RESIZE_FACTOR;
    dlib::matrix<unsigned char> resizedGrayImage(newRows, newCols);
    dlib::resize_image(grayImage, resizedGrayImage, dlib::interpolate_bilinear());

    std::vector<dlib::rectangle> faces = detector(resizedGrayImage);

    std::ranges::transform(faces, faces.begin(),
                           [](const dlib::rectangle& rectangle) { return dlib::scale_rect(rectangle, FACE_DETECTION_IMG_RESIZE_FACTOR); });
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