#ifndef FACENET_VECTORSEARCH_VECTORDB_H
#define FACENET_VECTORSEARCH_VECTORDB_H
#include <string>
#include <utility>

class VectorDB {

    std::string dbFilePath;

    void loadRecordsFromFile() {
    }

  public:
    explicit VectorDB(std::string dbFilePath): dbFilePath(std::move(dbFilePath)) {
        this->loadRecordsFromFile();
    };
};

#endif // FACENET_VECTORSEARCH_VECTORDB_H
