#include "FaceEmbedder.h"

#include "Types.h"
#include "dlib/image_transforms/interpolation.h"
#include "dlib/matrix/matrix.h"
#include "dlib/pixel.h"
#include "executorch/extension/module/module.h"
#include "executorch/extension/tensor/tensor_ptr_maker.h"

std::vector<Embedding> FaceEmbedder::computeFaceEmbedding(const std::vector<DlibRgbImage>& inputImages) {
    std::vector<Embedding> outputEmbeddings;

    for (const auto& inputImage: inputImages) {
        DlibRgbImage resizedImage;
        dlib::set_image_size(resizedImage, INPUT_IMG_DIM, INPUT_IMG_DIM);
        resize_image(inputImage, resizedImage);

        std::vector<float> pixelFloatData(INPUT_IMG_DIM * INPUT_IMG_DIM * 3);
        for (int row = 0; row < INPUT_IMG_DIM; row++) {
            for (int col = 0; col < INPUT_IMG_DIM; col++) {
                const dlib::rgb_pixel& pixel = resizedImage(row, col);
                pixelFloatData.push_back(pixel.red);
                pixelFloatData.push_back(pixel.green);
                pixelFloatData.push_back(pixel.blue);
            }
        }

        auto inputTensor = executorch::extension::from_blob(pixelFloatData.data(), {1, INPUT_IMG_DIM, INPUT_IMG_DIM, 3});
        const executorch::runtime::Result<std::vector<executorch::runtime::EValue>> result = faceNetModule.forward(inputTensor);

        if (!result.ok()) {
            std::cerr << to_string(result.error()) << '\n';
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