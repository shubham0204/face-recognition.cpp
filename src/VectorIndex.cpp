#include "VectorIndex.h"

void VectorIndex::loadRecordsFromFile() {
    std::ifstream inputStream(dbFilePath, std::ios::binary);
    if (!inputStream.is_open()) {
        loadedSuccessfully = true;
        return;
    }
    inputStream.seekg(0, std::ios::end);
    const std::streamoff length = inputStream.tellg();
    inputStream.seekg(0, std::ios::beg);
    if (length <= 0) {
        loadedSuccessfully = true;
        return;
    }

    std::vector<char> data(length);
    if (!inputStream.read(data.data(), length)) {
        throw std::runtime_error("Failed to read vector DB file: " + dbFilePath);
    }
    inputStream.close();

    if (const auto* fbRecords = GetVectorRecords(data.data())->records(); fbRecords != nullptr) {
        records.reserve(fbRecords->size());
        for (const auto* record : *fbRecords) {
            VectorRecordData recordData;
            const auto* fbVector = record->vector();
            std::copy_n(fbVector->begin(), std::min<size_t>(fbVector->size(), EMBEDDING_DIM), recordData.vector.begin());
            recordData.norm = record->norm();
            if (record->person_name() != nullptr) {
                recordData.personName = record->person_name()->str();
            }
            records.push_back(std::move(recordData));
        }
    }

    loadedSuccessfully = true;
}

void VectorIndex::writeRecordsToFile() const {
    flatbuffers::FlatBufferBuilder builder(1024);

    std::vector<flatbuffers::Offset<VectorRecord>> recordOffsets;
    recordOffsets.reserve(records.size());
    for (const auto& [vector, norm, personName] : records) {
        const auto vectorOffset = builder.CreateVector(vector.data(), vector.size());
        const auto nameOffset = builder.CreateString(personName);
        recordOffsets.push_back(CreateVectorRecord(builder, vectorOffset, norm, nameOffset));
    }

    const auto recordsVectorOffset = builder.CreateVector(recordOffsets);
    const auto tableOffset = CreateVectorRecords(builder, recordsVectorOffset);
    builder.Finish(tableOffset);

    const std::string tmpPath = dbFilePath + ".tmp";
    {
        std::ofstream outputStream(tmpPath, std::ios::binary | std::ios::trunc);
        if (!outputStream.is_open()) {
            throw std::runtime_error("Failed to open temp file for writing: " + tmpPath);
        }
        outputStream.write(reinterpret_cast<const char*>(builder.GetBufferPointer()), static_cast<std::streamsize>(builder.GetSize()));
        if (!outputStream) {
            throw std::runtime_error("Failed to write vector DB file: " + tmpPath);
        }
    }

    if (std::rename(tmpPath.c_str(), dbFilePath.c_str()) != 0) {
        throw std::runtime_error("Failed to finalize vector DB file: " + dbFilePath);
    }
}

float VectorIndex::computeNorm(const std::array<float, 512>& embedding) {
    float sumSquares = 0.0f;
    for (const auto& element : embedding) {
        sumSquares += element * element;
    }
    return std::sqrt(sumSquares);
}

float VectorIndex::computeDotProduct(const std::array<float, EMBEDDING_DIM>& embedding1, const std::array<float, EMBEDDING_DIM>& embedding2) {
    float dotProduct = 0.0f;
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        dotProduct += embedding1[i] * embedding2[i];
    }
    return dotProduct;
}

VectorIndex::~VectorIndex() {
    if (loadedSuccessfully) {
        try {
            writeRecordsToFile();
        } catch (...) {
            // Destructors shouldn't throw; swallow and let the caller
            // discover issues via an explicit save() if one is added.
        }
    }
}

void VectorIndex::insert(const std::string& personName, const std::array<float, 512>& embedding) {
    VectorRecordData record;
    record.vector = embedding;
    record.norm = computeNorm(embedding);
    record.personName = personName;
    records.push_back(std::move(record));
}

NNQueryResult VectorIndex::nearestNeighbor(const std::array<float, 512>& embedding) const {
    const float n = computeNorm(embedding);
    float maxCosineSimilarity = 0.0f;
    std::string maxCosineSimilarityPersonName;
    for (const auto& record : records) {
        float cosine = computeDotProduct(record.vector, embedding) / (n * record.norm);
        if (cosine > maxCosineSimilarity) {
            maxCosineSimilarity = cosine;
            maxCosineSimilarityPersonName = record.personName;
        }
    }
    NNQueryResult result;
    result.personName = maxCosineSimilarityPersonName;
    result.cosineSimilarity = maxCosineSimilarity;
    return result;
}

const std::vector<VectorRecordData>& VectorIndex::getRecords() const { return records; }