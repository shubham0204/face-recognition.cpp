#include "FaceEmbedder.h"

#include "Logger.h"
#include "Types.h"
#include "dlib/image_transforms/interpolation.h"
#include "dlib/matrix/matrix.h"
#include "dlib/pixel.h"
#include "executorch/extension/module/module.h"
#include "executorch/extension/tensor/tensor_ptr_maker.h"

std::vector<Embedding> FaceEmbedder::computeFaceEmbedding(const std::vector<DlibRgbImage>& inputImages) {
    std::vector<Embedding> outputEmbeddings;
    outputEmbeddings.reserve(inputImages.size());
    for (const auto& inputImage : inputImages) {
        DlibRgbImage resizedImage;
        dlib::set_image_size(resizedImage, INPUT_IMG_DIM, INPUT_IMG_DIM);
        resize_image(inputImage, resizedImage);

        std::vector<float> pixelFloatData(INPUT_IMG_DIM * INPUT_IMG_DIM * 3);
        for (int row = 0; row < INPUT_IMG_DIM; row++) {
            for (int col = 0; col < INPUT_IMG_DIM; col++) {
                const dlib::rgb_pixel& pixel = resizedImage(row, col);
                const size_t idx = (row * INPUT_IMG_DIM + col) * 3;
                resizedImagePixelData[idx + 0] = pixel.red;
                resizedImagePixelData[idx + 1] = pixel.green;
                resizedImagePixelData[idx + 2] = pixel.blue;
            }
        }

        auto inputTensor = executorch::extension::from_blob(resizedImagePixelData.data(), {1, INPUT_IMG_DIM, INPUT_IMG_DIM, 3});
        const executorch::runtime::Result<std::vector<executorch::runtime::EValue>> result = faceNetModule.forward(inputTensor);

        if (!result.ok()) {
            Logger::log(to_string(result.error()));
        }
        if (result.ok()) {
            const auto outputTensor = result.get().at(0).toTensor();
            const auto outputTensorFloatData = static_cast<const float*>(outputTensor.const_data_ptr());
            Embedding embedding{};
            std::copy_n(outputTensorFloatData, EMBEDDING_DIM, embedding.begin());
            outputEmbeddings.push_back(embedding);
        }
    }

    return outputEmbeddings;
}