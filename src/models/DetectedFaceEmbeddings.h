#ifndef FACENET_VECTORSEARCH_DETECTEDFACEEMBEDDINGS_H
#define FACENET_VECTORSEARCH_DETECTEDFACEEMBEDDINGS_H
#include "FaceEmbedder.h"
#include "dlib/geometry/rectangle.h"

class DetectedFaceEmbeddings {
  public:
    std::vector<std::array<float, EMBEDDING_DIM>> embeddings;
    std::vector<dlib::rectangle> faceBoundingBoxes;
};

#endif // FACENET_VECTORSEARCH_DETECTEDFACEEMBEDDINGS_H
