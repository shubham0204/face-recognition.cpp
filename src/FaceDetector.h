#ifndef FACENET_VECTORSEARCH_FACEDETECTOR_H
#define FACENET_VECTORSEARCH_FACEDETECTOR_H

#include <dlib/image_processing/frontal_face_detector.h>
#include "dlib/image_loader/load_image.h"

class FaceDetector {
    dlib::frontal_face_detector detector;

  public:
    FaceDetector() { detector = dlib::get_frontal_face_detector(); };

    std::vector<dlib::rectangle> detectFaces(const dlib::matrix<dlib::rgb_pixel>& inputImage);
};

#endif // FACENET_VECTORSEARCH_FACEDETECTOR_H
