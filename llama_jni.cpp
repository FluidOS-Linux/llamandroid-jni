#include <jni.h>
#include <string>
#include <vector>
#include "llama.h"
#include "ggml.h"

struct InferenceContext {
    llama_model*   model;
    llama_context* ctx;
    JavaVM*        jvm;
    jobject        javaObj;
    // Store context params so we can recreate ctx between inferences
    uint32_t       n_ctx;
    uint32_t       n_threads;
};

static void fireOnToken(InferenceContext* ic, const std::string& token) {
    JNIEnv* env;
    ic->jvm->AttachCurrentThread(&env, nullptr);
    jclass    cls  = env->GetObjectClass(ic->javaObj);
    jmethodID mid  = env->GetMethodID(cls, "onToken", "(Ljava/lang/String;)V");
    jstring   jtok = env->NewStringUTF(token.c_str());
    env->CallVoidMethod(ic->javaObj, mid, jtok);
    env->DeleteLocalRef(jtok);
    env->DeleteLocalRef(cls);
    ic->jvm->DetachCurrentThread();
}

static void fireOnComplete(InferenceContext* ic, const std::string& fullText) {
    JNIEnv* env;
    ic->jvm->AttachCurrentThread(&env, nullptr);
    jclass    cls   = env->GetObjectClass(ic->javaObj);
    jmethodID mid   = env->GetMethodID(cls, "onComplete", "(Ljava/lang/String;)V");
    jstring   jtext = env->NewStringUTF(fullText.c_str());
    env->CallVoidMethod(ic->javaObj, mid, jtext);
    env->DeleteLocalRef(jtext);
    env->DeleteLocalRef(cls);
    ic->jvm->DetachCurrentThread();
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_pocketive_llamandroid_LlamaAndroid_nativeLoadModel(
        JNIEnv* env, jobject obj,
        jstring modelPath, jint contextSize, jint threads) {

    const char* path = env->GetStringUTFChars(modelPath, nullptr);

    llama_backend_init();

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
        JNIEnv* env, jobject obj,
        jlong handle, jstring prompt, jint maxTokens) {

    InferenceContext* ic = (InferenceContext*) handle;
    if (!ic) return;

    // ── FIX: Recreate the llama_context before each inference ────────────────
    // llama_kv_cache_clear was removed in recent master, and llama_memory_seq_rm
    // with seq_id=-1 does not fully reset internal position state.
    // Recreating the context is the only API-stable way to guarantee a clean
    // slate. The model weights stay loaded in RAM — only the small KV buffer
    // (~few MB for 2048 ctx) is freed and reallocated, which takes ~1-2ms.
    llama_free(ic->ctx);
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = ic->n_ctx;
    cparams.n_threads = ic->n_threads;
    ic->ctx = llama_init_from_model(ic->model, cparams);
    if (!ic->ctx) {
        fireOnComplete(ic, "[context reset failed]");
        return;
    }

    const char* promptStr = env->GetStringUTFChars(prompt, nullptr);
    std::string promptCpp(promptStr);
    env->ReleaseStringUTFChars(prompt, promptStr);

    const llama_vocab* vocab = llama_model_get_vocab(ic->model);

    // Tokenize with auto-sizing buffer
    int bufSize = (int)promptCpp.size() + 128;
    std::vector<llama_token> tokens(bufSize);
    int nTokens = llama_tokenize(
        vocab, promptCpp.c_str(), (int)promptCpp.size(),
        tokens.data(), (int)tokens.size(), true, false);
    if (nTokens < 0) {
        tokens.resize(-nTokens + 1);
        nTokens = llama_tokenize(
            vocab, promptCpp.c_str(), (int)promptCpp.size(),
            tokens.data(), (int)tokens.size(), true, false);
    }
    if (nTokens < 0) { fireOnComplete(ic, "[tokenization failed]"); return; }
    tokens.resize(nTokens);

    // Decode prompt
    llama_batch batch = llama_batch_get_one(tokens.data(), nTokens);
    if (llama_decode(ic->ctx, batch) != 0) {
        fireOnComplete(ic, "[decode failed]");
        return;
    }

    llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    std::string fullOutput;
    std::string tokenBatch;
    int batchCount = 0;
    const int BATCH_SIZE = 6;

    for (int i = 0; i < (int)maxTokens; i++) {
        llama_token id = llama_sampler_sample(sampler, ic->ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) break;

        char buf[256];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, false);
        if (n < 0) break;

        std::string piece(buf, n);
        fullOutput += piece;
        tokenBatch += piece;
        batchCount++;

        if (batchCount >= BATCH_SIZE) {
            fireOnToken(ic, tokenBatch);
            tokenBatch.clear();
            batchCount = 0;
        }

        llama_sampler_accept(sampler, id);

        llama_batch next = llama_batch_get_one(&id, 1);
        if (llama_decode(ic->ctx, next) != 0) break;
    }

    llama_sampler_free(sampler);
    if (!tokenBatch.empty()) fireOnToken(ic, tokenBatch);
    fireOnComplete(ic, fullOutput);
}

extern "C" JNIEXPORT void JNICALL
Java_com_pocketive_llamandroid_LlamaAndroid_nativeFreeModel(
        JNIEnv* env, jobject obj, jlong handle) {

    InferenceContext* ic = (InferenceContext*) handle;
    if (!ic) return;

    llama_free(ic->ctx);
    llama_model_free(ic->model);
    llama_backend_free();
    env->DeleteGlobalRef(ic->javaObj);
    delete ic;
}
