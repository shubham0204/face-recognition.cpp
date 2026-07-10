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

namespace {

// The recognizer is process-wide; created once via createFaceRecognizer().
std::unique_ptr<FaceRecognizer> g_recognizer;

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
    jclass bitmapClass = env->GetObjectClass(bitmap);
    jmethodID getWidthMethod = env->GetMethodID(bitmapClass, "getWidth", "()I");
    jmethodID getHeightMethod = env->GetMethodID(bitmapClass, "getHeight", "()I");
    jmethodID getPixelsMethod = env->GetMethodID(bitmapClass, "getPixels", "([IIIIIII)V");

    const jint width = env->CallIntMethod(bitmap, getWidthMethod);
    const jint height = env->CallIntMethod(bitmap, getHeightMethod);
    env->DeleteLocalRef(bitmapClass);

    if (width <= 0 || height <= 0) {
        LOGE("Invalid bitmap dimensions: %d x %d", width, height);
        return false;
    }

    jintArray pixelsArray = env->NewIntArray(width * height);
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

// ---------------------------------------------------------------------------
// java.util.List<Bitmap> -> std::vector<IntBufferImage>
//
// `storages` holds the backing pixel buffers so their memory stays alive for
// as long as `outImages` (whose IntBufferImage entries point into them) is used.
// ---------------------------------------------------------------------------

bool javaBitmapListToImages(JNIEnv* env, jobject listObj, std::vector<std::vector<int>>& storages, std::vector<IntBufferImage>& outImages) {
    jclass listClass = env->GetObjectClass(listObj);
    jmethodID sizeMethod = env->GetMethodID(listClass, "size", "()I");
    jmethodID getMethod = env->GetMethodID(listClass, "get", "(I)Ljava/lang/Object;");

    const jint count = env->CallIntMethod(listObj, sizeMethod);
    storages.resize(count);
    outImages.resize(count);

    for (jint i = 0; i < count; ++i) {
        auto bitmap = (jobject)env->CallObjectMethod(listObj, getMethod, i);
        const bool ok = bitmapToIntBufferImage(env, bitmap, storages[i], outImages[i]);
        env->DeleteLocalRef(bitmap);
        if (!ok)
            return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// NNQueryResult -> Kotlin NativeFaceRecognitionModule.NNQueryResult
// FaceBoundingBox  -> Kotlin NativeFaceRecognitionModule.FaceBoundingBox
// ---------------------------------------------------------------------------

jobject buildFaceBoundingBox(JNIEnv* env, const FaceBoundingBox& box) {
    jclass cls = env->FindClass("com/ml/shubham0204/facenet_android/domain/NativeFaceRecognitionModule$FaceBoundingBox");
    jmethodID ctor = env->GetMethodID(cls, "<init>", "(JJJJ)V");
    // Kotlin constructor order: top, left, right, bottom
    jobject result = env->NewObject(cls, ctor, static_cast<jlong>(box.top), static_cast<jlong>(box.left), static_cast<jlong>(box.right),
                                    static_cast<jlong>(box.bottom));
    env->DeleteLocalRef(cls);
    return result;
}

jobject buildNNQueryResult(JNIEnv* env, const NNQueryResult& r) {
    jclass cls = env->FindClass("com/ml/shubham0204/facenet_android/domain/NativeFaceRecognitionModule$NNQueryResult");
    jmethodID ctor = env->GetMethodID(cls, "<init>",
                                      "(Ljava/lang/String;Ljava/lang/String;"
                                      "Lcom/ml/shubham0204/facenet_android/domain/NativeFaceRecognitionModule$FaceBoundingBox;)V");

    jstring personName = env->NewStringUTF(r.personName.c_str());
    jstring cosineSimilarity = env->NewStringUTF(std::to_string(r.cosineSimilarity).c_str());
    jobject boundingBox = buildFaceBoundingBox(env, r.faceBoundingBox);

    jobject result = env->NewObject(cls, ctor, personName, cosineSimilarity, boundingBox);

    env->DeleteLocalRef(cls);
    env->DeleteLocalRef(personName);
    env->DeleteLocalRef(cosineSimilarity);
    env->DeleteLocalRef(boundingBox);
    return result;
}

jobject buildResultList(JNIEnv* env, const std::vector<NNQueryResult>& results) {
    jclass arrayListClass = env->FindClass("java/util/ArrayList");
    jmethodID arrayListCtor = env->GetMethodID(arrayListClass, "<init>", "(I)V");
    jmethodID addMethod = env->GetMethodID(arrayListClass, "add", "(Ljava/lang/Object;)Z");

    jobject list = env->NewObject(arrayListClass, arrayListCtor, static_cast<jint>(results.size()));
    for (const auto& r : results) {
        jobject item = buildNNQueryResult(env, r);
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

    g_recognizer = createFaceRecognizer(dbPathStr, modelPathStr);
    if (!g_recognizer) {
        LOGE("createFaceRecognizer returned null");
    }
}

extern "C" JNIEXPORT void JNICALL Java_com_ml_shubham0204_facenet_1android_domain_NativeFaceRecognitionModule_insert(JNIEnv* env, jobject /* thiz */,
                                                                                                                     jstring personName,
                                                                                                                     jobject images) {
    if (!g_recognizer) {
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

    g_recognizer->insert(personNameStr, imageBuffers);
}

extern "C" JNIEXPORT jobject JNICALL Java_com_ml_shubham0204_facenet_1android_domain_NativeFaceRecognitionModule_recognize(JNIEnv* env,
                                                                                                                           jobject /* thiz */,
                                                                                                                           jobject image) {
    if (!g_recognizer) {
        LOGE("recognize() called before createFaceRecognizer()");
        return buildResultList(env, {});
    }

    std::vector<int> storage;
    IntBufferImage nativeImage{};
    if (!bitmapToIntBufferImage(env, image, storage, nativeImage)) {
        LOGE("Failed to convert Bitmap for recognize()");
        return buildResultList(env, {});
    }

    const std::vector<NNQueryResult> results = g_recognizer->recognize(nativeImage);
    return buildResultList(env, results);
}