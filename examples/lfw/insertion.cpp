#include "../../external/dlib/dlib/data_io/image_dataset_metadata.h"
#include "Utils.h"

#include <FaceRecognizer.h>
#include <algorithm>
#include <filesystem>
#include <iostream>

int main() {
    const auto faceRecognizer = createFaceRecognizer(std::format("{}/data/vectordb.bin", SRC_DIR), std::format("{}/data/model.pte", SRC_DIR));
    const auto csvData = readCsvData(std::string(SRC_DIR) + "/data/matchpairsDevTrain.csv");
    long insertedCount = 0;
    for (const auto& record : csvData) {
        const std::string& personName = record[0];
        const std::string paddedImageNumString = std::format("{:04}", std::stoi(record[1]));
        const std::string imgPath = std::format("{}/data/lfw-deepfunneled/{}/{}_{}.jpg", SRC_DIR, personName, personName, paddedImageNumString);
        faceRecognizer->insert(personName, {imgPath});
        insertedCount++;
    }
    return 0;
}