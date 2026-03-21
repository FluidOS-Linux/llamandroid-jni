#include <jni.h>
#include <string>
#include <vector>
#include "llama.h"
#include "ggml.h"

// ── Backend init guard ────────────────────────────────────────────────────────
// llama_backend_init() must be called exactly once per process.
// Calling it multiple times (e.g. load → free → load) or calling
// llama_backend_free() mid-session corrupts global GGML state.
static bool gBackendInitialized = false;

struct InferenceContext {
    llama_model*   model;
    llama_context* ctx;
    JavaVM*        jvm;
    jobject        javaObj;   // GlobalRef to the LlamaAndroid Java object
    uint32_t       n_ctx;
    uint32_t       n_threads;
};

// ── CRITICAL FIX: do NOT use AttachCurrentThread / DetachCurrentThread here ──
//
// These functions are called from nativeInfer(), which runs on the Java
// ExecutorService thread.  That thread is already a JVM-managed Java thread —
// it is already "attached".  Calling DetachCurrentThread() on an already-
// attached Java thread permanently detaches it from the JVM.  Any subsequent
// JNI call from that thread (the very next token) causes a SIGSEGV crash.
//
// The correct fix: receive the JNIEnv* that the JVM already handed us in
// nativeInfer() and pass it straight through.  No attach/detach needed.

static void fireOnToken(JNIEnv* env, InferenceContext* ic, const std::string& token) {
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

// ── nativeLoadModel ───────────────────────────────────────────────────────────

extern "C" JNIEXPORT jlong JNICALL
Java_com_pocketive_llamandroid_LlamaAndroid_nativeLoadModel(
        JNIEnv* env, jobject obj,
        jstring modelPath, jint contextSize, jint threads) {

    const char* path = env->GetStringUTFChars(modelPath, nullptr);

    // Only ever initialise the backend once per process lifetime.
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

// ── nativeInfer ──────────────────────────────────────────────────────────────

extern "C" JNIEXPORT void JNICALL
Java_com_pocketive_llamandroid_LlamaAndroid_nativeInfer(
        JNIEnv* env, jobject /*obj*/,
        jlong handle, jstring prompt, jint maxTokens) {

    InferenceContext* ic = (InferenceContext*) handle;
    if (!ic) return;

    // Recreate the context before each inference for a clean KV-cache slate.
    // The model weights stay in RAM; only the small KV buffer (~few MB) is
    // freed and reallocated (~1–2 ms).
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

    // Tokenise with auto-sizing buffer.
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
    if (nTokens < 0) { fireOnComplete(env, ic, "[tokenization failed]"); return; }
    tokens.resize(nTokens);

    // Decode the prompt.
    llama_batch batch = llama_batch_get_one(tokens.data(), nTokens);
    if (llama_decode(ic->ctx, batch) != 0) {
        fireOnComplete(env, ic, "[decode failed]");
        return;
    }

    llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));


    std::string fullOutput;
    std::string tokenBatch;
    int batchCount  = 0;
    const int BATCH_SIZE = 6;

    for (int i = 0; i < (int)maxTokens; i++) {
        llama_token id = llama_sampler_sample(sampler, ic->ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) break;

        char buf[256];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, false);
        if (n <= 0) break;

        std::string piece(buf, n);
        fullOutput  += piece;
        tokenBatch  += piece;
        batchCount++;

        if (batchCount >= BATCH_SIZE) {
            // env is valid here — this IS the Java executor thread, already attached.
            fireOnToken(env, ic, tokenBatch);
            tokenBatch.clear();
            batchCount = 0;
        }

        llama_sampler_accept(sampler, id);

        llama_batch next = llama_batch_get_one(&id, 1);
        if (llama_decode(ic->ctx, next) != 0) break;
    }

    llama_sampler_free(sampler);

    if (!tokenBatch.empty()) {
        fireOnToken(env, ic, tokenBatch);
    }
    fireOnComplete(env, ic, fullOutput);
}

// ── nativeFreeModel ──────────────────────────────────────────────────────────

extern "C" JNIEXPORT void JNICALL
Java_com_pocketive_llamandroid_LlamaAndroid_nativeFreeModel(
        JNIEnv* env, jobject /*obj*/, jlong handle) {

    InferenceContext* ic = (InferenceContext*) handle;
    if (!ic) return;

    llama_free(ic->ctx);
    llama_model_free(ic->model);

    // DO NOT call llama_backend_free() here.
    // It tears down global GGML state (thread pools, memory arenas).
    // Calling it between load/free/reload cycles leaves the backend in a
    // half-freed state and causes a crash on the next nativeLoadModel call.
    // The backend will be cleaned up automatically when the process exits.

    env->DeleteGlobalRef(ic->javaObj);
    delete ic;
}
