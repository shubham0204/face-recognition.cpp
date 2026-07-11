#ifndef FACENET_VECTORSEARCH_FACEDETECTOR_H
#define FACENET_VECTORSEARCH_FACEDETECTOR_H

#include "Types.h"

#include "dlib/image_loader/load_image.h"
#include <dlib/image_processing/frontal_face_detector.h>

class FaceDetector {
    dlib::frontal_face_detector detector;

  public:
    FaceDetector() { detector = dlib::get_frontal_face_detector(); };

    std::vector<dlib::rectangle> detectFaces(const DlibRgbImage& inputImage);

    static dlib::array<DlibRgbImage> cropFaces(const DlibRgbImage& inputImage, const std::vector<dlib::rectangle>& rectangles);
};

#endif // FACENET_VECTORSEARCH_FACEDETECTOR_H
