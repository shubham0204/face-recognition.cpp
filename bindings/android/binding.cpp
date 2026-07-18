#include <jni.h>

#include <memory>
#include <string>
#include <vector>

#include "FaceBoundingBox.h"
#include "FaceRecognizer.h"
#include "IntBufferImage.h"
#include "NNQueryResult.h"

#define LOG_TAG "NativeFaceRecognitionModule"
#define LOGE(...)                                                                                                                                    \
    printf(__VA_ARGS__);                                                                                                                             \
    printf("\n");

#define CLASS_PATH_FACE_BOUNDING_BOX "com/ml/shubham0204/facenet_android/domain/NativeFaceRecognitionModule$FaceBoundingBox"
#define CLASS_PATH_NN_QUERY_RESULT "com/ml/shubham0204/facenet_android/domain/NativeFaceRecognitionModule$NNQueryResult"

namespace {

// The recognizer is process-wide; created once via createFaceRecognizer().
std::unique_ptr<FaceRecognizer> faceRecognizer;

// ---------------------------------------------------------------------------
// String helpers
// ---------------------------------------------------------------------------

std::string jstringToStdString(JNIEnv* env, jstring jStr) {
    if (jStr == nullptr)
        return {};
    const char* chars = env->GetStringUTFChars(jStr, nullptr);
    std::string result(chars);
    env->ReleaseStringUTFChars(jStr, chars);
    return result;
}

// ---------------------------------------------------------------------------
// Bitmap -> IntBufferImage
//
// We call android.graphics.Bitmap.getPixels() via JNI rather than walking the
// bitmap's native pixel memory ourselves. getPixels() always returns pixels
// packed as (A << 24) | (R << 16) | (G << 8) | B regardless of the bitmap's
// underlying storage config (ARGB_8888, RGB_565, etc.), so this matches the
// app's existing Kotlin preprocessing (which reads R/G/B via shr 16/8/0 and
// ignores alpha) without needing to special-case pixel formats natively.
//
// `storage` must outlive the IntBufferImage that borrows its buffer pointer.
// ---------------------------------------------------------------------------

bool bitmapToIntBufferImage(JNIEnv* env, jobject bitmap, std::vector<int>& storage, IntBufferImage& outImage) {
    const auto bitmapClass = env->GetObjectClass(bitmap);
    const auto getWidthMethod = env->GetMethodID(bitmapClass, "getWidth", "()I");
    const auto getHeightMethod = env->GetMethodID(bitmapClass, "getHeight", "()I");
    const auto getPixelsMethod = env->GetMethodID(bitmapClass, "getPixels", "([IIIIIII)V");

    const auto width = env->CallIntMethod(bitmap, getWidthMethod);
    const auto height = env->CallIntMethod(bitmap, getHeightMethod);
    env->DeleteLocalRef(bitmapClass);

    if (width <= 0 || height <= 0) {
        LOGE("Invalid bitmap dimensions: %d x %d", width, height);
        return false;
    }

    const auto pixelsArray = env->NewIntArray(width * height);
    if (pixelsArray == nullptr) {
        LOGE("Failed to allocate pixel array (%d x %d)", width, height);
        return false;
    }

    // getPixels(int[] pixels, int offset, int stride, int x, int y, int width, int height)
    env->CallVoidMethod(bitmap, getPixelsMethod, pixelsArray, 0, width, 0, 0, width, height);
    if (env->ExceptionCheck()) {
        env->ExceptionClear();
        env->DeleteLocalRef(pixelsArray);
        LOGE("Bitmap.getPixels() threw an exception");
        return false;
    }

    storage.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
    env->GetIntArrayRegion(pixelsArray, 0, width * height, storage.data());
    env->DeleteLocalRef(pixelsArray);

    outImage.buffer = storage.data();
    outImage.width = width;
    outImage.height = height;
    return true;
}

bool javaBitmapListToImages(JNIEnv* env, jobject listObj, std::vector<std::vector<int>>& storages, std::vector<IntBufferImage>& outImages) {
    const auto listClass = env->GetObjectClass(listObj);
    const auto sizeMethod = env->GetMethodID(listClass, "size", "()I");
    const auto getMethod = env->GetMethodID(listClass, "get", "(I)Ljava/lang/Object;");

    const auto count = env->CallIntMethod(listObj, sizeMethod);
    storages.resize(count);
    outImages.resize(count);

    for (jint i = 0; i < count; ++i) {
        const auto bitmap = env->CallObjectMethod(listObj, getMethod, i);
        const bool ok = bitmapToIntBufferImage(env, bitmap, storages[i], outImages[i]);
        env->DeleteLocalRef(bitmap);
        if (!ok)
            return false;
    }
    return true;
}

jobject buildFaceBoundingBox(JNIEnv* env, const FaceBoundingBox& box) {
    const auto cls = env->FindClass(CLASS_PATH_FACE_BOUNDING_BOX);
    const auto ctor = env->GetMethodID(cls, "<init>", "(JJJJ)V");
    // Kotlin constructor order: top, left, right, bottom
    const auto result = env->NewObject(cls, ctor, static_cast<jlong>(box.top), static_cast<jlong>(box.left), static_cast<jlong>(box.right),
                                       static_cast<jlong>(box.bottom));
    env->DeleteLocalRef(cls);
    return result;
}

jobject buildNNQueryResult(JNIEnv* env, const NNQueryResult& r) {
    const auto cls = env->FindClass(CLASS_PATH_NN_QUERY_RESULT);
    const auto constructor = env->GetMethodID(cls, "<init>",
                                              "(Ljava/lang/String;D"
                                              "Lcom/ml/shubham0204/facenet_android/domain/NativeFaceRecognitionModule$FaceBoundingBox;)V");

    const auto personName = env->NewStringUTF(r.personName.c_str());
    const auto cosineSimilarity = static_cast<jdouble>(r.cosineSimilarity);
    const auto boundingBox = buildFaceBoundingBox(env, r.faceBoundingBox);

    const auto result = env->NewObject(cls, constructor, personName, cosineSimilarity, boundingBox);

    env->DeleteLocalRef(cls);
    env->DeleteLocalRef(personName);
    env->DeleteLocalRef(boundingBox);
    return result;
}

jobject buildResultList(JNIEnv* env, const std::vector<NNQueryResult>& results) {
    const auto arrayListClass = env->FindClass("java/util/ArrayList");
    const auto arrayListConstructor = env->GetMethodID(arrayListClass, "<init>", "(I)V");
    const auto addMethod = env->GetMethodID(arrayListClass, "add", "(Ljava/lang/Object;)Z");

    const auto list = env->NewObject(arrayListClass, arrayListConstructor, static_cast<jint>(results.size()));
    for (const auto& r : results) {
        const auto item = buildNNQueryResult(env, r);
        env->CallBooleanMethod(list, addMethod, item);
        env->DeleteLocalRef(item);
    }
    env->DeleteLocalRef(arrayListClass);
    return list;
}

} // namespace

// ---------------------------------------------------------------------------
// JNI exports
// ---------------------------------------------------------------------------

extern "C" JNIEXPORT void JNICALL Java_com_ml_shubham0204_facenet_1android_domain_NativeFaceRecognitionModule_createFaceRecognizer(
    JNIEnv* env, jobject /* thiz */, jstring dbPath, jstring faceNetModelPath) {
    const std::string dbPathStr = jstringToStdString(env, dbPath);
    const std::string modelPathStr = jstringToStdString(env, faceNetModelPath);

    faceRecognizer = createFaceRecognizer(dbPathStr, modelPathStr);
    if (!faceRecognizer) {
        LOGE("createFaceRecognizer returned null");
    }
}

extern "C" JNIEXPORT void JNICALL Java_com_ml_shubham0204_facenet_1android_domain_NativeFaceRecognitionModule_insert(JNIEnv* env, jobject /* thiz */,
                                                                                                                     jstring personName,
                                                                                                                     jobject images) {
    if (!faceRecognizer) {
        LOGE("insert() called before createFaceRecognizer()");
        return;
    }

    const std::string personNameStr = jstringToStdString(env, personName);

    std::vector<std::vector<int>> storages;
    std::vector<IntBufferImage> imageBuffers;
    if (!javaBitmapListToImages(env, images, storages, imageBuffers)) {
        LOGE("Failed to convert Bitmap list for insert()");
        return;
    }

    faceRecognizer->insert(personNameStr, imageBuffers);
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_ml_shubham0204_facenet_1android_domain_NativeFaceRecognitionModule_remove(JNIEnv* env, jobject,
                                                                                                                         jstring personName) {
    const std::string personNameStr = jstringToStdString(env, personName);
    return faceRecognizer->remove(personNameStr);
}

extern "C" JNIEXPORT void JNICALL Java_com_ml_shubham0204_facenet_1android_domain_NativeFaceRecognitionModule_clear(JNIEnv* env, jobject) {
    faceRecognizer->clear();
}

extern "C" JNIEXPORT jobject JNICALL Java_com_ml_shubham0204_facenet_1android_domain_NativeFaceRecognitionModule_recognize(JNIEnv* env,
                                                                                                                           jobject /* thiz */,
                                                                                                                           jobject image) {
    if (!faceRecognizer) {
        LOGE("recognize() called before createFaceRecognizer()");
        return buildResultList(env, {});
    }

    std::vector<int> storage;
    IntBufferImage nativeImage{};
    if (!bitmapToIntBufferImage(env, image, storage, nativeImage)) {
        LOGE("Failed to convert Bitmap for recognize()");
        return buildResultList(env, {});
    }

    const std::vector<NNQueryResult> results = faceRecognizer->recognize(nativeImage);
    return buildResultList(env, results);
}