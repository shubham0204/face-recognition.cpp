#include "FaceDetector.h"

std::vector<dlib::rectangle> FaceDetector::detectFaces(const dlib::matrix<dlib::rgb_pixel>& inputImage) {
    dlib::matrix<unsigned char> grayImage;
    dlib::assign_image(grayImage, inputImage);
    std::vector<dlib::rectangle> faces = detector(grayImage);
    return faces;
}