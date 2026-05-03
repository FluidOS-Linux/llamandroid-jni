#include <jni.h>
#include <string>
#include <vector>
#include "llama.h"
#include "ggml.h"

static bool gBackendInitialized = false;

struct InferenceContext {
    llama_model*   model;
    llama_context* ctx;
    JavaVM*        jvm;
    jobject        javaObj;
    uint32_t       n_ctx;
    uint32_t       n_threads;
};

static void fireOnToken(JNIEnv* env, InferenceContext* ic, const std::string& token) {
    if (token.empty()) return;
    jclass    cls  = env->GetObjectClass(ic->javaObj);
    jmethodID mid  = env->GetMethodID(cls, "onToken", "(Ljava/lang/String;)V");
    jstring   jtok = env->NewStringUTF(token.c_str());
    env->CallVoidMethod(ic->javaObj, mid, jtok);
    env->DeleteLocalRef(jtok);
    env->DeleteLocalRef(cls);
}

static void fireOnComplete(JNIEnv* env, InferenceContext* ic, const std::string& fullText) {
    jclass    cls   = env->GetObjectClass(ic->javaObj);
    jmethodID mid   = env->GetMethodID(cls, "onComplete", "(Ljava/lang/String;)V");
    jstring   jtext = env->NewStringUTF(fullText.c_str());
    env->CallVoidMethod(ic->javaObj, mid, jtext);
    env->DeleteLocalRef(jtext);
    env->DeleteLocalRef(cls);
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_pocketive_llamandroid_LlamaAndroid_nativeLoadModel(
        JNIEnv* env, jobject obj,
        jstring modelPath, jint contextSize, jint threads) {

    const char* path = env->GetStringUTFChars(modelPath, nullptr);

    if (!gBackendInitialized) {
        llama_backend_init();
        gBackendInitialized = true;
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    llama_model* model = llama_model_load_from_file(path, mparams);
    env->ReleaseStringUTFChars(modelPath, path);
    if (!model) return 0;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = (uint32_t) contextSize;
    cparams.n_threads = (uint32_t) threads;

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        llama_model_free(model);
        return 0;
    }

    InferenceContext* ic = new InferenceContext();
    ic->model     = model;
    ic->ctx       = ctx;
    ic->n_ctx     = (uint32_t) contextSize;
    ic->n_threads = (uint32_t) threads;
    ic->javaObj   = env->NewGlobalRef(obj);
    env->GetJavaVM(&ic->jvm);

    return (jlong) ic;
}

extern "C" JNIEXPORT void JNICALL
Java_com_pocketive_llamandroid_LlamaAndroid_nativeInfer(
        JNIEnv* env, jobject,
        jlong handle, jstring prompt, jint maxTokens, jstring stopStr) {

    InferenceContext* ic = (InferenceContext*) handle;
    if (!ic) return;

    const char* stopChars = env->GetStringUTFChars(stopStr, nullptr);
    std::string stopString(stopChars);
    env->ReleaseStringUTFChars(stopStr, stopChars);

    llama_free(ic->ctx);
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = ic->n_ctx;
    cparams.n_threads = ic->n_threads;
    ic->ctx = llama_init_from_model(ic->model, cparams);
    if (!ic->ctx) {
        fireOnComplete(env, ic, "[context reset failed]");
        return;
    }

    const char* promptStr = env->GetStringUTFChars(prompt, nullptr);
    std::string promptCpp(promptStr);
    env->ReleaseStringUTFChars(prompt, promptStr);

    const llama_vocab* vocab = llama_model_get_vocab(ic->model);

    int bufSize = (int)promptCpp.size() + 128;
    std::vector<llama_token> tokens(bufSize);
    int nTokens = llama_tokenize(
        vocab, promptCpp.c_str(), (int)promptCpp.size(),
        tokens.data(), (int)tokens.size(), true, true);
    if (nTokens < 0) {
        tokens.resize(-nTokens + 1);
        nTokens = llama_tokenize(
            vocab, promptCpp.c_str(), (int)promptCpp.size(),
            tokens.data(), (int)tokens.size(), true, true);
    }
    if (nTokens < 0) { fireOnComplete(env, ic, "[tokenization failed]"); return; }
    tokens.resize(nTokens);

    llama_batch batch = llama_batch_get_one(tokens.data(), nTokens);
    if (llama_decode(ic->ctx, batch) != 0) {
        fireOnComplete(env, ic, "[decode failed]");
        return;
    }

    llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    const size_t holdSize = stopString.empty() ? 0 : stopString.size();

    std::string fullOutput;
    std::string holdback;
    std::string tokenBatch;
    const int BATCH_SIZE = 6;

    for (int i = 0; i < (int)maxTokens; i++) {
        llama_token id = llama_sampler_sample(sampler, ic->ctx, -1);

        if (llama_vocab_is_eog(vocab, id)) break;

        char buf[256];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n <= 0) break;

        std::string piece(buf, n);
        fullOutput += piece;
        holdback   += piece;

        if (!stopString.empty()) {
            size_t stopPos = holdback.find(stopString);
            if (stopPos != std::string::npos) {
                std::string safeChunk = holdback.substr(0, stopPos);
                tokenBatch += safeChunk;
                if (!tokenBatch.empty()) {
                    fireOnToken(env, ic, tokenBatch);
                    tokenBatch.clear();
                }
                size_t fullStopPos = fullOutput.find(stopString);
                if (fullStopPos != std::string::npos) {
                    fullOutput = fullOutput.substr(0, fullStopPos);
                }
                break;
            }

            if (holdback.size() > holdSize) {
                std::string safe = holdback.substr(0, holdback.size() - holdSize);
                holdback = holdback.substr(holdback.size() - holdSize);
                tokenBatch += safe;
            }
        } else {
            tokenBatch += piece;
        }

        if ((int)tokenBatch.size() >= BATCH_SIZE) {
            fireOnToken(env, ic, tokenBatch);
            tokenBatch.clear();
        }

        llama_sampler_accept(sampler, id);

        llama_batch next = llama_batch_get_one(&id, 1);
        if (llama_decode(ic->ctx, next) != 0) break;
    }

    llama_sampler_free(sampler);

    if (!stopString.empty() && !holdback.empty()) {
        for (size_t len = stopString.size() - 1; len >= 1; len--) {
            if (holdback.size() >= len &&
                holdback.compare(holdback.size() - len, len,
                                 stopString, 0, len) == 0) {
                holdback.erase(holdback.size() - len);
                fullOutput.erase(fullOutput.size() - len);
                break;
            }
        }
        tokenBatch += holdback;
    }

    if (!tokenBatch.empty()) {
        fireOnToken(env, ic, tokenBatch);
    }

    fireOnComplete(env, ic, fullOutput);
}

extern "C" JNIEXPORT void JNICALL
Java_com_pocketive_llamandroid_LlamaAndroid_nativeFreeModel(
        JNIEnv* env, jobject, jlong handle) {

    InferenceContext* ic = (InferenceContext*) handle;
    if (!ic) return;

    llama_free(ic->ctx);
    llama_model_free(ic->model);

    env->DeleteGlobalRef(ic->javaObj);
    delete ic;
}
