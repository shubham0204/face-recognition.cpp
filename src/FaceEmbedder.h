#ifndef FACENET_VECTORSEARCH_FACEEMBEDDER_H
#define FACENET_VECTORSEARCH_FACEEMBEDDER_H

#include "Types.h"
#include "dlib/matrix/matrix.h"
#include "dlib/pixel.h"
#include "executorch/extension/module/module.h"
#include <vector>

class FaceEmbedder {

    executorch::extension::Module faceNetModule;
    std::vector<float> resizedImagePixelData;

  public:
    explicit FaceEmbedder(const std::string& modelFilePath) : faceNetModule(modelFilePath) {
        resizedImagePixelData.reserve(INPUT_IMG_DIM * INPUT_IMG_DIM * 3);
    };

    std::vector<Embedding> computeFaceEmbedding(const std::vector<DlibRgbImage>& inputImages);
};

#endif // FACENET_VECTORSEARCH_FACEEMBEDDER_H
