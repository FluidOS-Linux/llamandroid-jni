#include <jni.h>
#include <string>
#include <vector>
#include <thread>
#include "llama.h"
#include "ggml.h"

// ── Globals per inference session ───────────────────────────────────────────

struct InferenceContext {
    llama_model* model;
    llama_context* ctx;
    JavaVM* jvm;
    jobject javaObj;  // reference to LlamaAndroid instance
};

// ── Helper: call back into Java ──────────────────────────────────────────────

static void fireOnToken(InferenceContext* ic, const std::string& token) {
    JNIEnv* env;
    ic->jvm->AttachCurrentThread(&env, nullptr);
    jclass cls = env->GetObjectClass(ic->javaObj);
    jmethodID mid = env->GetMethodID(cls, "onToken", "(Ljava/lang/String;)V");
    jstring jtok = env->NewStringUTF(token.c_str());
    env->CallVoidMethod(ic->javaObj, mid, jtok);
    env->DeleteLocalRef(jtok);
    env->DeleteLocalRef(cls);
    ic->jvm->DetachCurrentThread();
}

static void fireOnComplete(InferenceContext* ic, const std::string& fullText) {
    JNIEnv* env;
    ic->jvm->AttachCurrentThread(&env, nullptr);
    jclass cls = env->GetObjectClass(ic->javaObj);
    jmethodID mid = env->GetMethodID(cls, "onComplete", "(Ljava/lang/String;)V");
    jstring jtext = env->NewStringUTF(fullText.c_str());
    env->CallVoidMethod(ic->javaObj, mid, jtext);
    env->DeleteLocalRef(jtext);
    env->DeleteLocalRef(cls);
    ic->jvm->DetachCurrentThread();
}

// ── nativeLoadModel ──────────────────────────────────────────────────────────

extern "C" JNIEXPORT jlong JNICALL
Java_com_pocketive_llamandroid_LlamaAndroid_nativeLoadModel(
        JNIEnv* env, jobject obj,
        jstring modelPath, jint contextSize, jint threads) {

    const char* path = env->GetStringUTFChars(modelPath, nullptr);

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // CPU only on Android for now

    llama_model* model = llama_load_model_from_file(path, mparams);
    env->ReleaseStringUTFChars(modelPath, path);

    if (!model) return 0;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = (uint32_t) contextSize;
    cparams.n_threads = (uint32_t) threads;

    llama_context* ctx = llama_new_context_with_model(model, cparams);
    if (!ctx) {
        llama_free_model(model);
        return 0;
    }

    InferenceContext* ic = new InferenceContext();
    ic->model   = model;
    ic->ctx     = ctx;
    ic->javaObj = env->NewGlobalRef(obj);
    env->GetJavaVM(&ic->jvm);

    return (jlong) ic;
}

// ── nativeInfer ──────────────────────────────────────────────────────────────

extern "C" JNIEXPORT void JNICALL
Java_com_pocketive_llamandroid_LlamaAndroid_nativeInfer(
        JNIEnv* env, jobject obj,
        jlong handle, jstring prompt, jint maxTokens) {

    InferenceContext* ic = (InferenceContext*) handle;
    if (!ic) return;

    const char* promptStr = env->GetStringUTFChars(prompt, nullptr);
    std::string promptCpp(promptStr);
    env->ReleaseStringUTFChars(prompt, promptStr);

    // Tokenize
    std::vector<llama_token> tokens(promptCpp.size() + 64);
    int nTokens = llama_tokenize(
        llama_get_model(ic->ctx),
        promptCpp.c_str(),
        (int) promptCpp.size(),
        tokens.data(),
        (int) tokens.size(),
        true,  // add_bos
        false  // special
    );

    if (nTokens < 0) {
        fireOnComplete(ic, "[tokenization failed]");
        return;
    }
    tokens.resize(nTokens);

    // Eval prompt
    llama_batch batch = llama_batch_get_one(tokens.data(), nTokens);
    if (llama_decode(ic->ctx, batch) != 0) {
        fireOnComplete(ic, "[decode failed]");
        return;
    }

    // Generate tokens one by one
    std::string fullOutput;
    // batch tokens in chunks per the Reddit advice
    std::string tokenBatch;
    int batchSize = 6;
    int batchCount = 0;

    for (int i = 0; i < (int) maxTokens; i++) {
        llama_token id = llama_sampler_sample(
            llama_sampler_chain_init(llama_sampler_chain_default_params()),
            ic->ctx, -1
        );

        if (llama_token_is_eog(llama_get_model(ic->ctx), id)) break;

        char buf[256];
        int n = llama_token_to_piece(llama_get_model(ic->ctx), id, buf, sizeof(buf), 0, false);
        if (n < 0) break;

        std::string piece(buf, n);
        fullOutput += piece;
        tokenBatch += piece;
        batchCount++;

        // Fire to Java every 6 tokens instead of every single one
        if (batchCount >= batchSize) {
            fireOnToken(ic, tokenBatch);
            tokenBatch.clear();
            batchCount = 0;
        }

        // Continue decoding
        llama_batch next = llama_batch_get_one(&id, 1);
        if (llama_decode(ic->ctx, next) != 0) break;
    }

    // Fire any remaining tokens
    if (!tokenBatch.empty()) fireOnToken(ic, tokenBatch);

    fireOnComplete(ic, fullOutput);

    // Reset context for next inference
    llama_kv_cache_clear(ic->ctx);
}

// ── nativeFreeModel ──────────────────────────────────────────────────────────

extern "C" JNIEXPORT void JNICALL
Java_com_pocketive_llamandroid_LlamaAndroid_nativeFreeModel(
        JNIEnv* env, jobject obj, jlong handle) {

    InferenceContext* ic = (InferenceContext*) handle;
    if (!ic) return;

    llama_free(ic->ctx);
    llama_free_model(ic->model);
    llama_backend_free();
    env->DeleteGlobalRef(ic->javaObj);
    delete ic;
}
