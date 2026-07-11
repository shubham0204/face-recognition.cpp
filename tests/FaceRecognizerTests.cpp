#include "../src/FaceRecognizerInternal.h"
#include "TestUtils.h"
#include "VectorIndex.h"
#include <gtest/gtest.h>

#define VECTORDB_FILE_NAME "vectordb.bin"
#define FACENET_MODEL_FILE_NAME "model.pte"

TEST(FaceRecognizerTests, CreateInsertReadDelete) {
    deleteResource(getResourcePath(VECTORDB_FILE_NAME));
    FaceRecognizerInternal faceRecognizer(getResourcePath(VECTORDB_FILE_NAME), getResourcePath(FACENET_MODEL_FILE_NAME));

    const std::string personName = "Shubham";

    auto start = std::chrono::steady_clock::now();
    faceRecognizer.insert(personName, {getResourcePath("img.png")});
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "[PERF] [Insertion time] " << duration.count() << " milliseconds" << '\n';

    start = std::chrono::steady_clock::now();
    const auto results = faceRecognizer.recognize(getResourcePath("img.png"));
    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "[PERF] [Retrieval time] " << duration.count() << " milliseconds" << '\n';

    EXPECT_EQ(results.size(), 1);
    EXPECT_TRUE(results[0].faceBoundingBox.top > 0);
    EXPECT_TRUE(results[0].faceBoundingBox.bottom > 0);
    EXPECT_TRUE(results[0].faceBoundingBox.left > 0);
    EXPECT_TRUE(results[0].faceBoundingBox.right > 0);
    EXPECT_EQ(results[0].personName, personName);
}

TEST(FaceRecognizerTests, Recognition) {
    deleteResource(getResourcePath(VECTORDB_FILE_NAME));
    FaceRecognizerInternal faceRecognizer(getResourcePath(VECTORDB_FILE_NAME), getResourcePath(FACENET_MODEL_FILE_NAME));

    faceRecognizer.insert("A", {getResourcePath("Steven_Hatfill_0001.jpg")});
    faceRecognizer.insert("B", {getResourcePath("Sue_Grafton_0001.jpg")});

    auto result = faceRecognizer.recognize(getResourcePath("Steven_Hatfill_0002.jpg"));
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].personName, "A");
}