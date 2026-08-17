#include "FaceRecognizer.hpp"
#include "Utils.h"

#include <iostream>

int main() {
    const auto faceRecognizer = createFaceRecognizer(std::format("{}/data/vectordb.bin", SRC_DIR), std::format("{}/data/qmodel.pte", SRC_DIR));
    const auto csvData = readCsvData(std::string(SRC_DIR) + "/data/matchpairsDevTrain.csv");
    long correct = 0;
    long total = 0;
    for (const auto& record : csvData) {
        const std::string& personName = record[0];
        const std::string paddedImageNumString = std::format("{:04}", std::stoi(record[2]));
        const std::string imgPath = std::format("{}/data/lfw-deepfunneled/{}/{}_{}.jpg", SRC_DIR, personName, personName, paddedImageNumString);
        const auto results = faceRecognizer->recognize(imgPath);
        total++;
        if (results.empty())
            continue;
        if (results[0].personName == personName) {
            correct++;
        }
    }
    std::cout << correct << '\n';
    std::cout << total << '\n';
    return 0;
}
