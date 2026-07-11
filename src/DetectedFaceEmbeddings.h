#ifndef FACENET_VECTORSEARCH_DETECTEDFACEEMBEDDINGS_H
#define FACENET_VECTORSEARCH_DETECTEDFACEEMBEDDINGS_H
#include "FaceEmbedder.h"
#include "Types.h"
#include "dlib/geometry/rectangle.h"

class DetectedFaceEmbeddings {
  public:
    std::vector<Embedding> embeddings;
    std::vector<dlib::rectangle> faceBoundingBoxes;
};

#endif // FACENET_VECTORSEARCH_DETECTEDFACEEMBEDDINGS_H
