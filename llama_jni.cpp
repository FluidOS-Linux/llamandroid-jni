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

// Do NOT call AttachCurrentThread / DetachCurrentThread here.
// nativeInfer runs on a Java ExecutorService thread which is already attached.

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

    const char* stopChars = env->GetStringUTFChars(stopStr, nullptr);
    std::string stopString(stopChars);
    env->ReleaseStringUTFChars(stopStr, stopChars);

    // Recreate context for a clean KV-cache slate.
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
        tokens.data(), (int)tokens.size(), true, false);
    if (nTokens < 0) {
        tokens.resize(-nTokens + 1);
        nTokens = llama_tokenize(
            vocab, promptCpp.c_str(), (int)promptCpp.size(),
            tokens.data(), (int)tokens.size(), true, false);
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

    // ── Holdback buffer ───────────────────────────────────────────────────────
    // This is the same approach used by ollama, llama-server, and every serious
    // LLM streaming runtime.
    //
    // The problem: a stop string like "<|im_end|>" arrives as multiple token
    // pieces across multiple decode steps. If we fire OnToken immediately for
    // every piece, some pieces of the stop string will have already been sent
    // to the UI before we realize a stop string was forming.
    //
    // The solution: maintain a "holdback" buffer that always retains the last
    // N characters where N = stopString.size(). We only release characters from
    // the front of the holdback to OnToken once we're certain they are not part
    // of a forming stop string. If the stop string completes in the holdback,
    // we discard it entirely and break. If generation ends normally, we release
    // whatever clean content remains in the holdback.
    //
    // This guarantees: the stop string NEVER appears in any OnToken call, and
    // OnComplete always receives perfectly clean output.

    const size_t holdSize = stopString.empty() ? 0 : stopString.size();

    std::string fullOutput;  // complete accumulated output (pre-strip)
    std::string holdback;    // characters held back pending stop string check
    const int BATCH_SIZE = 6;
    int batchCount = 0;
    std::string tokenBatch;  // batched chars ready to fire via OnToken

    for (int i = 0; i < (int)maxTokens; i++) {
        llama_token id = llama_sampler_sample(sampler, ic->ctx, -1);

        // Native EOG token — stop immediately.
        if (llama_vocab_is_eog(vocab, id)) break;

        char buf[256];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, false);
        if (n <= 0) break;

        std::string piece(buf, n);
        fullOutput += piece;
        holdback   += piece;

        // ── Check if stop string has fully formed in the holdback ─────────
        if (!stopString.empty()) {
            size_t stopPos = holdback.find(stopString);
            if (stopPos != std::string::npos) {
                // Stop string found. Release everything before it as clean
                // output, discard the stop string and everything after.
                std::string safeChunk = holdback.substr(0, stopPos);
                tokenBatch += safeChunk;
                if (!tokenBatch.empty()) {
                    fireOnToken(env, ic, tokenBatch);
                    tokenBatch.clear();
                }
                // Trim fullOutput to match what we actually sent.
                size_t fullStopPos = fullOutput.find(stopString);
                if (fullStopPos != std::string::npos) {
                    fullOutput = fullOutput.substr(0, fullStopPos);
                }
                break;
            }

            // No complete stop string yet. Release safe characters from the
            // front of holdback — only keep the last holdSize chars back.
            if (holdback.size() > holdSize) {
                std::string safe = holdback.substr(0, holdback.size() - holdSize);
                holdback = holdback.substr(holdback.size() - holdSize);
                tokenBatch += safe;
            }
        } else {
            // No stop string — everything is safe to release immediately.
            tokenBatch += piece;
        }

        // Fire OnToken in batches for smooth streaming.
        if ((int)tokenBatch.size() >= BATCH_SIZE) {
            fireOnToken(env, ic, tokenBatch);
            tokenBatch.clear();
            batchCount = 0;
        }

        llama_sampler_accept(sampler, id);

        llama_batch next = llama_batch_get_one(&id, 1);
        if (llama_decode(ic->ctx, next) != 0) break;
    }

    llama_sampler_free(sampler);

    // ── Flush remaining safe content ──────────────────────────────────────
    // If generation ended normally (EOG or maxTokens), release whatever is
    // left in the holdback. It contains no stop string (we would have broken
    // out of the loop already if it did), so it's safe to send.
    if (!stopString.empty() && !holdback.empty()) {
        // Final check: strip any partial stop string prefix at the very end.
        // e.g. output ends with "<|im_end" — trim that partial match.
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

    // OnComplete fires with the same clean output that was streamed.
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

    // Do NOT call llama_backend_free() here.
    env->DeleteGlobalRef(ic->javaObj);
    delete ic;
}
