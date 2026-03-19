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
    ic->model   = model;
    ic->ctx     = ctx;
    ic->javaObj = env->NewGlobalRef(obj);
    env->GetJavaVM(&ic->jvm);

    return (jlong) ic;
}

extern "C" JNIEXPORT void JNICALL
Java_com_pocketive_llamandroid_LlamaAndroid_nativeInfer(
        JNIEnv* env, jobject obj,
        jlong handle, jstring prompt, jint maxTokens) {

    InferenceContext* ic = (InferenceContext*) handle;
    if (!ic) return;

    const char* promptStr = env->GetStringUTFChars(prompt, nullptr);
    std::string promptCpp(promptStr);
    env->ReleaseStringUTFChars(prompt, promptStr);

    const llama_vocab* vocab = llama_model_get_vocab(ic->model);

    // ── FIX: Fully reset context state before each inference ────────────────
    // llama_kv_cache_clear is the correct, stable API for wiping the KV cache.
    // The old llama_memory_seq_rm approach only removes sequences but leaves
    // the context's internal "n_past" position counter dirty, causing the
    // second decode call to write into already-used slots → crash/OOM.
    llama_kv_cache_clear(ic->ctx);

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

    // ── FIX: Create sampler fresh each call, free at end ────────────────────
    // Reusing a sampler across calls carries stale repetition-penalty state,
    // which can cause degenerate output or illegal memory access on 2nd call.
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

        // ── FIX: Accept the sampled token into sampler state ────────────────
        // Without this, greedy sampler doesn't update its internal last-token
        // state, which breaks repetition avoidance on longer outputs.
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
