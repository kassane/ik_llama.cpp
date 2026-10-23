#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

ggml_cgraph * llm_build_context::build_flint() {
    ggml_cgraph * gf = new_graph_custom();

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    struct ggml_tensor * inp_pos     = build_inp_pos();
    struct ggml_tensor * inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;
    struct ggml_tensor * KQ_mask     = build_inp_KQ_mask();
    struct ggml_tensor * KQ_mask_swa = build_inp_KQ_mask_swa();

    for (int il = 0; il < n_layer; ++il) {
        const bool is_swa = hparams.swa_layers[il];
        const int64_t n_embd_head = hparams.n_embd_head_v(il);
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(il));
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        auto KQ_mask_l = is_swa ? KQ_mask_swa : KQ_mask;

        cur = build_std_attention(gf, model.layers[il].attn_norm, inpL,
                inp_pos, il == n_layer - 1 ? inp_out_ids : nullptr, nullptr, KQ_mask_l,
                model.layers[il].attn_sinks, nullptr, 1.0f/sqrtf(float(n_embd_head)), 0.0f,
                is_swa ? hparams.n_swa : 0, il, hparams.has_rope(il), false, true);

        auto ffn_inp = cur;

        if (il < (int) hparams.n_layer_dense_lead) {
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true);
            cb(cur, "ffn_out", il);
        } else {
            cur = llm_build_std_moe_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_gate_inp,  nullptr,
                    model.layers[il].ffn_up_exps,   nullptr,
                    model.layers[il].ffn_gate_exps, nullptr,
                    model.layers[il].ffn_down_exps, nullptr,
                    model.layers[il].ffn_exp_probs_b,
                    model.layers[il].ffn_up_shexp,   nullptr,
                    model.layers[il].ffn_gate_shexp, nullptr,
                    model.layers[il].ffn_down_shexp, nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU,
                    hparams.expert_weights_norm,
                    hparams.expert_weights_scale != 1.0f,
                    hparams.expert_weights_scale,
                    (llm_expert_gating_func_type) hparams.expert_gating_func,
                    LLM_FFN_SILU, cb, il, gf, true,
                    model.layers[il].ffn_up_gate_exps);
            cb(cur, "ffn_out", il);
        }

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;

    cur = build_output(lctx, ctx0, cur, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}
