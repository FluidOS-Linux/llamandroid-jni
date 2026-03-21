#include <jni.h>
#include <string>
#include <vector>
#include "llama.h"
#include "ggml.h"

// ── Backend init guard ────────────────────────────────────────────────────────
// llama_backend_init() must be called exactly once per process.
static bool gBackendInitialized = false;

struct InferenceContext {
    llama_model*   model;
    llama_context* ctx;
    JavaVM*        jvm;
    jobject        javaObj;
    uint32_t       n_ctx;
    uint32_t       n_threads;
};

// ── Fire helpers ─────────────────────────────────────────────────────────────
// IMPORTANT: do NOT call AttachCurrentThread / DetachCurrentThread here.
// nativeInfer runs on a Java ExecutorService thread which is already attached.
// Detaching it permanently breaks all subsequent JNI calls from that thread.

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
        jlong handle, jstring prompt, jint maxTokens, jstring stopStr) {

    InferenceContext* ic = (InferenceContext*) handle;
    if (!ic) return;

    // Extract stop string (empty string = no stop)
    const char* stopChars = env->GetStringUTFChars(stopStr, nullptr);
    std::string stopString(stopChars);
    env->ReleaseStringUTFChars(stopStr, stopChars);

    // Recreate context for a clean KV-cache slate before each inference.
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

    // Sampler: temperature → top-p → dist (prevents greedy repetition loops)
    llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    std::string fullOutput;
    std::string tokenBatch;
    int batchCount = 0;
    const int BATCH_SIZE = 6;
    bool hitStop = false;

    for (int i = 0; i < (int)maxTokens; i++) {
        llama_token id = llama_sampler_sample(sampler, ic->ctx, -1);

        // Model-native end-of-generation token
        if (llama_vocab_is_eog(vocab, id)) break;

        char buf[256];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, false);
        if (n <= 0) break;

        std::string piece(buf, n);
        fullOutput += piece;
        tokenBatch += piece;
        batchCount++;

        // ── Stop string check ─────────────────────────────────────────────
        // This is the standard way every LLM runtime (ollama, llama-server,
        // llmcpp) halts generation at a custom delimiter.  Different model
        // families use different stop tokens:
        //   Qwen:    <|im_end|>
        //   Llama3:  <|eot_id|>
        //   Mistral: </s>
        //   Gemma:   <end_of_turn>
        //   Phi:     <|end|>
        // Passing "" disables the check entirely.
        if (!stopString.empty() &&
            fullOutput.size() >= stopString.size() &&
            fullOutput.compare(fullOutput.size() - stopString.size(),
                               stopString.size(), stopString) == 0) {

            // Trim the stop string from both output buffers before breaking.
            fullOutput.erase(fullOutput.size() - stopString.size());
            if (tokenBatch.size() >= stopString.size()) {
                tokenBatch.erase(tokenBatch.size() - stopString.size());
            } else {
                tokenBatch.clear();
            }
            hitStop = true;
            break;
        }

        // Fire token batch every BATCH_SIZE pieces for smooth streaming.
        if (batchCount >= BATCH_SIZE) {
            fireOnToken(env, ic, tokenBatch);
            tokenBatch.clear();
            batchCount = 0;
        }

        llama_sampler_accept(sampler, id);

        llama_batch next = llama_batch_get_one(&id, 1);
        if (llama_decode(ic->ctx, next) != 0) break;
    }

    llama_sampler_free(sampler);

    // Fire any remaining buffered tokens.
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

    // DO NOT call llama_backend_free() here — it tears down global GGML state
    // and causes crashes on the next nativeLoadModel call.
    // It runs automatically when the process exits.

    env->DeleteGlobalRef(ic->javaObj);
    delete ic;
}
