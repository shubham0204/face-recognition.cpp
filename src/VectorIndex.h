#ifndef FACENET_VECTORSEARCH_VECTORDB_H
#define FACENET_VECTORSEARCH_VECTORDB_H

#include "FaceEmbedder.h"
#include "NNQueryResult.h"
#include "Types.h"
#include "dlib/geometry/rectangle.h"
#include "schema/VectorRecords_generated.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

struct VectorRecordData {
    Embedding vector{};
    double norm = 0.0;
    std::string personName;
};

class VectorIndex {
    std::string dbFilePath;
    std::vector<VectorRecordData> records;
    bool loadedSuccessfully = false;

    void readFromDisk();

    void writeToDisk() const;

    static double computeNorm(const Embedding& embedding);

    static double computeDotProduct(const Embedding& embedding1, const Embedding& embedding2);

  public:
    explicit VectorIndex(std::string dbFilePath) : dbFilePath(std::move(dbFilePath)) { this->readFromDisk(); }

    ~VectorIndex();

    void insert(const std::string& personName, const Embedding& embedding);

    [[nodiscard]] NNQueryResult nearestNeighbor(const Embedding& embedding) const;

    [[nodiscard]] const std::vector<VectorRecordData>& getRecords() const;

    bool remove(const std::string& personName);

    void clear();
};

#endif // FACENET_VECTORSEARCH_VECTORDB_H