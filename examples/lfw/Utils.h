#pragma once
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

inline std::vector<std::string> gatherImagesFromDir(const std::filesystem::path& dirPath) {
    std::vector<std::string> paths;
    for (const auto& imgPath : std::filesystem::directory_iterator(dirPath)) {
        if (imgPath.is_directory())
            continue;
        paths.push_back(absolute(imgPath));
    }
    return paths;
}

inline std::vector<std::vector<std::string>> readCsvData(const std::string& csvFilePath) {
    std::ifstream file(csvFilePath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file." << std::endl;
        return {};
    }
    std::string line;
    std::vector<std::vector<std::string>> csvData;
    while (std::getline(file, line)) {
        std::vector<std::string> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            row.push_back(cell);
        }
        csvData.push_back(row);
    }
    file.close();
    // remove first row, the headers
    csvData.erase(csvData.begin());
    return csvData;
}