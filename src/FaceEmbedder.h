#ifndef FACENET_VECTORSEARCH_FACEEMBEDDER_H
#define FACENET_VECTORSEARCH_FACEEMBEDDER_H

#include "Types.h"
#include "dlib/matrix/matrix.h"
#include "dlib/pixel.h"
#include "executorch/extension/module/module.h"
#include <vector>

class FaceEmbedder {

    executorch::extension::Module faceNetModule;

  public:
    explicit FaceEmbedder(const std::string& modelFilePath) : faceNetModule(modelFilePath) {};

    std::vector<Embedding> computeFaceEmbedding(const std::vector<dlib::matrix<dlib::rgb_pixel>>& inputImages);
};

#endif // FACENET_VECTORSEARCH_FACEEMBEDDER_H
