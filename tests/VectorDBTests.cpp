#include "TestUtils.h"
#include "VectorIndex.h"
#include <gtest/gtest.h>

#define VECTORDB_FILE_NAME "vectordb.bin"

TEST(VectorDBTests, CreateInsertReadDelete) {
    deleteResource(VECTORDB_FILE_NAME);
    VectorIndex index(getResourcePath(VECTORDB_FILE_NAME));
    index.insert(std::string("HelloWorld"), Embedding{});
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

    const auto [personName, cosineSimilarity, _] = index.nearestNeighbor(createRandomEmbedding());
    EXPECT_TRUE(personName.starts_with("Person"));
    EXPECT_TRUE(-1 < cosineSimilarity && cosineSimilarity < 1);
}

TEST(VectorDBTests, DeleteRecords) {
    deleteResource(VECTORDB_FILE_NAME);
    VectorIndex index(getResourcePath(VECTORDB_FILE_NAME));

    for (int i = 0; i < 10; i++) {
        index.insert(std::format("Person{}", i + 1), createRandomEmbedding());
    }
    const auto records1 = index.getRecords();
    EXPECT_EQ(records1.size(), 10);

    index.remove("Person1");
    index.remove("Person2");
    const auto records2 = index.getRecords();
    EXPECT_EQ(records2.size(), 8);

    index.clear();
    const auto records3 = index.getRecords();
    EXPECT_EQ(records3.size(), 0);
}

TEST(VectorDBTests, VerifyScheduledDiskWrites) {
    deleteResource(VECTORDB_FILE_NAME);

    VectorIndex index(getResourcePath(VECTORDB_FILE_NAME));
    for (int i = 0; i < 10; i++) {
        index.insert(std::format("Person{}", i + 1), createRandomEmbedding());
    }
    const auto records1 = index.getRecords();
    EXPECT_EQ(records1.size(), 10);
    std::this_thread::sleep_for(std::chrono::seconds(VectorIndex::WRITE_INTERVAL_IN_SECONDS + 2));

    // We expect that `index` has already written the records to disk
    // as we waited for VectorIndex::WRITE_INTERVAL_IN_SECONDS + 2
    // Constructing another index on the same disk file should contain the same
    // number of records as that of `index`
    VectorIndex index2(getResourcePath(VECTORDB_FILE_NAME));
    EXPECT_EQ(index2.getRecords().size(), 10);
}