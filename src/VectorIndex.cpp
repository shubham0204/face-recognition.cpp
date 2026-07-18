#include "VectorIndex.h"

#include "NNQueryResult.h"

#include <thread>

void VectorIndex::readFromDisk() {
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
    this->isMemoryCopyDirty = false;
}

void VectorIndex::writeToDisk() {
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

    this->isMemoryCopyDirty = false;
}

double VectorIndex::computeNorm(const Embedding& embedding) {
    double sumSquares = 0.0;
    for (const auto& element : embedding) {
        sumSquares += element * element;
    }
    return std::sqrt(sumSquares);
}

double VectorIndex::computeDotProduct(const Embedding& embedding1, const Embedding& embedding2) {
    double dotProduct = 0.0f;
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        dotProduct += embedding1[i] * embedding2[i];
    }
    return dotProduct;
}

void VectorIndex::scheduleWriteToDisk() {
    if (isWriteToDiskScheduled) {
        return;
    }
    isWriteToDiskScheduled = true;
    this->writeToDiskThread = std::thread([this] {
        while (isWriteToDiskScheduled) {
            if (isMemoryCopyDirty) {
                writeToDisk();
            }
            std::this_thread::sleep_for(std::chrono::seconds(VectorIndex::WRITE_INTERVAL_IN_SECONDS));
        }
    });
}

void VectorIndex::descheduleWriteToDisk() {
    if (!isWriteToDiskScheduled) {
        return;
    }
    isWriteToDiskScheduled = false;
    if (this->writeToDiskThread.joinable()) {
        this->writeToDiskThread.join();
    }
}

VectorIndex::~VectorIndex() {
    descheduleWriteToDisk();
    if (loadedSuccessfully && isMemoryCopyDirty) {
        writeToDisk();
    }
}

void VectorIndex::insert(const std::string& personName, const Embedding& embedding) {
    VectorRecordData record;
    record.vector = embedding;
    record.norm = computeNorm(embedding);
    record.personName = personName;
    records.push_back(std::move(record));
    this->isMemoryCopyDirty = true;
}

NNQueryResult VectorIndex::nearestNeighbor(const Embedding& embedding) const {
    const double n = computeNorm(embedding);
    double maxCosineSimilarity = -std::numeric_limits<double>::infinity();
    ;
    std::string maxCosineSimilarityPersonName;
    for (const auto& record : records) {
        double cosine = computeDotProduct(record.vector, embedding) / (n * record.norm);
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

bool VectorIndex::remove(const std::string& personName) {
    const auto numDeletedRecords = std::erase_if(records, [personName](const auto& record) { return record.personName == personName; });
    this->isMemoryCopyDirty = numDeletedRecords > 0;
    return this->isMemoryCopyDirty;
}

void VectorIndex::clear() {
    this->records.clear();
    this->isMemoryCopyDirty = true;
}