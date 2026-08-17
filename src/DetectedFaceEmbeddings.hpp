#ifndef FACENET_VECTORSEARCH_DETECTEDFACEEMBEDDINGS_H
#define FACENET_VECTORSEARCH_DETECTEDFACEEMBEDDINGS_H
#include "FaceEmbedder.hpp"
#include "Types.hpp"
#include "dlib/geometry/rectangle.h"

class DetectedFaceEmbeddings {
  public:
    std::vector<Embedding> embeddings;
    std::vector<dlib::rectangle> faceBoundingBoxes;
};

#endif // FACENET_VECTORSEARCH_DETECTEDFACEEMBEDDINGS_H
