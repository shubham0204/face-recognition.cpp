#include "TestUtils.h"
#include "VectorIndex.h"
#include <gtest/gtest.h>

#define VECTORDB_FILE_NAME "vectordb.bin"

TEST(VectorDBTests, CreateInsertReadDelete) {
    deleteResource(VECTORDB_FILE_NAME);
    VectorIndex index(getResourcePath(VECTORDB_FILE_NAME));
    index.insert(std::string("HelloWorld"), std::array<float, EMBEDDING_DIM>{});
    const auto records = index.getRecords();
    EXPECT_EQ(records.size(), 1);
    EXPECT_EQ(records[0].personName, "HelloWorld");
    EXPECT_NEAR(records[0].norm, 0.0f, 0.01f);
}

TEST(VectorDBTests, NearestNeighbor) {
    deleteResource(VECTORDB_FILE_NAME);
    VectorIndex index(getResourcePath(VECTORDB_FILE_NAME));

    for (int i = 0; i < 10; i++) {
        index.insert(std::format("Person{}", i + 1), createRandomEmbedding());
    }
    const auto records = index.getRecords();
    EXPECT_EQ(records.size(), 10);

    const NNQueryResult nearestNeighbor = index.nearestNeighbor(createRandomEmbedding());
    EXPECT_TRUE(nearestNeighbor.personName.starts_with("Person"));
    EXPECT_TRUE(-1 < nearestNeighbor.cosineSimilarity && nearestNeighbor.cosineSimilarity < 1);
}