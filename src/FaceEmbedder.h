#ifndef FACENET_VECTORSEARCH_FACEEMBEDDER_H
#define FACENET_VECTORSEARCH_FACEEMBEDDER_H

#include "dlib/matrix/matrix.h"
#include "dlib/pixel.h"
#include "executorch/extension/module/module.h"

#include <array>
#include <vector>

#define EMBEDDING_DIM 512
#define INPUT_IMG_DIM 160

class FaceEmbedder {

    executorch::extension::Module faceNetModule;

  public:
    explicit FaceEmbedder(const std::string& modelFilePath) : faceNetModule(modelFilePath) {};

    std::vector<std::array<float, EMBEDDING_DIM>> computeFaceEmbedding(const std::vector<dlib::matrix<dlib::rgb_pixel>>& inputImages);
};

#endif // FACENET_VECTORSEARCH_FACEEMBEDDER_H
