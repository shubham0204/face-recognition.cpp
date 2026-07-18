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
#include <thread>
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

    bool isMemoryCopyDirty = false;
    std::atomic<bool> isWriteToDiskScheduled = false;
    std::thread writeToDiskThread;

    void readFromDisk();

    void writeToDisk();

    static double computeNorm(const Embedding& embedding);

    static double computeDotProduct(const Embedding& embedding1, const Embedding& embedding2);

    void scheduleWriteToDisk();

    void descheduleWriteToDisk();

  public:
    static constexpr int WRITE_INTERVAL_IN_SECONDS = 10;

    explicit VectorIndex(std::string dbFilePath) : dbFilePath(std::move(dbFilePath)) {
        this->readFromDisk();
        this->scheduleWriteToDisk();
    }

    ~VectorIndex();

    void insert(const std::string& personName, const Embedding& embedding);

    [[nodiscard]] NNQueryResult nearestNeighbor(const Embedding& embedding) const;

    [[nodiscard]] const std::vector<VectorRecordData>& getRecords() const;

    bool remove(const std::string& personName);

    void clear();
};

#endif // FACENET_VECTORSEARCH_VECTORDB_H