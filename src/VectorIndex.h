#ifndef FACENET_VECTORSEARCH_VECTORDB_H
#define FACENET_VECTORSEARCH_VECTORDB_H

#include "FaceEmbedder.h"
#include "schema/VectorRecords_generated.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

struct VectorRecordData {
    std::array<float, EMBEDDING_DIM> vector{};
    float norm = 0.0f;
    std::string personName;
};

struct NNQueryResult {
    std::string personName;
    float cosineSimilarity = 0.0f;
};

class VectorIndex {
    std::string dbFilePath;
    std::vector<VectorRecordData> records;
    bool loadedSuccessfully = false;

    void loadRecordsFromFile();

    void writeRecordsToFile() const;

    static float computeNorm(const std::array<float, EMBEDDING_DIM>& embedding);

    static float computeDotProduct(const std::array<float, EMBEDDING_DIM>& embedding1, const std::array<float, EMBEDDING_DIM>& embedding2);

  public:
    explicit VectorIndex(std::string dbFilePath) : dbFilePath(std::move(dbFilePath)) { this->loadRecordsFromFile(); }

    ~VectorIndex();

    void insert(const std::string& personName, const std::array<float, EMBEDDING_DIM>& embedding);

    NNQueryResult nearestNeighbor(const std::array<float, EMBEDDING_DIM>& embedding) const;

    [[nodiscard]] const std::vector<VectorRecordData>& getRecords() const;
};

#endif // FACENET_VECTORSEARCH_VECTORDB_H