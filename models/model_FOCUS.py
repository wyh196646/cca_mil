# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import os
import warnings

import torch
import torch.nn as nn
from torch.nn import functional as F

from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize


logger = logging.getLogger(__name__)


class TextEncoder(nn.Module):
    def __init__(self, conch_model):
        super().__init__()
        self.text = conch_model.text
        self.transformer = conch_model.text.transformer
        self.positional_embedding = conch_model.text.positional_embedding
        self.ln_final = conch_model.text.ln_final
        self.text_projection = conch_model.text.text_projection
        self.cls_emb = getattr(conch_model.text, "cls_emb", None)
        self.dtype = next(conch_model.parameters()).dtype

    def forward(self, prompts, tokenized_prompts):
        cast_dtype = self.transformer.get_cast_dtype()
        x = prompts.to(cast_dtype)
        tokenized_prompts = tokenized_prompts.to(x.device)
        seq_len = x.shape[1]
        attn_mask = self.text.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask.to(device=x.device, dtype=cast_dtype)

        if self.cls_emb is not None:
            seq_len += 1
            x = torch.cat([x, self.text._repeat(self.cls_emb, x.shape[0]).to(cast_dtype)], dim=1)
            cls_mask = self.text.build_cls_mask(tokenized_prompts, cast_dtype)
            attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]
        else:
            attn_mask = attn_mask[:seq_len, :seq_len] if attn_mask is not None else None

        x = x + self.positional_embedding[:seq_len].to(cast_dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)

        if self.cls_emb is not None:
            pooled = self.ln_final(x[:, -1])
        else:
            x = self.ln_final(x)
            pooled = x[torch.arange(x.shape[0], device=x.device), tokenized_prompts.argmax(dim=-1)]

        if self.text_projection is not None:
            pooled = pooled @ self.text_projection
        return pooled


class PromptLearner(nn.Module):
    def __init__(self, classnames, conch_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 16
        ctx_init = ""
        dtype = next(conch_model.parameters()).dtype
        ctx_dim = conch_model.text.ln_final.weight.shape[0]
        self.tokenizer = get_tokenizer()

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(self.tokenizer, [ctx_init])
            with torch.no_grad():
                embedding = conch_model.text.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)

        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [name for name in classnames]
        tokenized_prompts = tokenize(self.tokenizer, prompts)
        if getattr(conch_model.text, "cls_emb", None) is not None:
            tokenized_prompts = tokenized_prompts[:, :-1]

        with torch.no_grad():
            embedding = conch_model.text.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.name_lens = [
            len(self.tokenizer.encode(name, max_length=127, truncation=True))
            for name in classnames
        ]
        self.class_token_position = "end"

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat([prefix_i, class_i, ctx_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError
        return prompts


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _resolve_conch_ckpt(config):
    candidates = []
    configured = getattr(config, "conch_ckpt_path", None)
    if configured:
        candidates.append(configured)
    candidates.extend([
        "ckpts/conch.pth",
        "ckg/pytorch_model.bin",
    ])

    for path in candidates:
        if path and os.path.isfile(path):
            return path
    raise FileNotFoundError("CONCH checkpoint not found. Tried: {}".format(candidates))


class FOCUS(nn.Module):
    def __init__(self, config, num_classes=3):
        super(FOCUS, self).__init__()
        self.loss_ce = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.window_size = config.window_size
        self.sim_threshold = config.sim_threshold

        self.L = config.input_size
        self.D = 512
        self.L_max = config.max_context_length

        conch_model_cfg = "conch_ViT-B-16"
        conch_checkpoint_path = _resolve_conch_ckpt(config)
        conch_model, _ = create_model_from_pretrained(conch_model_cfg, conch_checkpoint_path)
        conch_model = conch_model.float().eval()

        self.feature_dim = conch_model.text.text_projection.shape[1]

        self.prompt_learner = PromptLearner(config.text_prompt, conch_model)
        self.text_encoder = TextEncoder(conch_model)

        self.feature_encoder = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.LayerNorm(self.D),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        num_heads = 8
        self.head_dim = self.feature_dim // num_heads
        self.q_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.k_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.v_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.o_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.num_heads = num_heads

        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def cross_attention(self, queries, keys, values, attention_mask=None):
        bsz, q_len, _ = queries.size()
        _, kv_len, _ = keys.size()

        query_states = self.q_proj(queries)
        key_states = self.k_proj(keys)
        value_states = self.v_proj(values)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.feature_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output

    def compute_patch_similarity(self, x, window_size):
        N, _ = x.shape
        x_norm = F.normalize(x, p=2, dim=-1)

        similarities = []
        selected_indices = []

        for i in range(0, N, window_size):
            window = x_norm[i : i + window_size]

            if len(window) < 2:
                selected_indices.append(torch.arange(i, min(i + window_size, N), device=x.device))
                continue

            window_sim = torch.mm(window, window.t())

            if window_sim.numel() > 1:
                threshold = window_sim.mean() + window_sim.std(unbiased=False)
            else:
                threshold = window_sim.mean()

            redundant = window_sim.mean(1) > threshold
            keep_indices = torch.where(~redundant)[0] + i

            if len(keep_indices) == 0:
                keep_indices = torch.tensor([i], device=x.device)

            selected_indices.append(keep_indices)
            similarities.append(window_sim)

        if not selected_indices:
            return [], torch.arange(N, device=x.device)

        return similarities, torch.cat(selected_indices)

    def adaptive_token_selection(self, features, text_features):
        N, _ = features.shape
        _, indices = self.compute_patch_similarity(features, self.window_size)

        if features.shape[-1] != text_features.shape[-1]:
            projection = nn.Linear(features.shape[-1], text_features.shape[-1], device=features.device)
            features_projected = projection(features)
        else:
            features_projected = features

        text_relevance = torch.matmul(features_projected, text_features.T).mean(-1)

        importance_mask = torch.zeros(N, device=features.device)
        importance_mask[indices] = text_relevance[indices]

        num_tokens = min(self.L_max, N)
        _, selected_indices = torch.topk(importance_mask, num_tokens)
        selected_indices, _ = torch.sort(selected_indices)

        selected_features = features[selected_indices]

        return selected_features, selected_indices

    def spatial_token_compression(self, features, text_features):
        chunk_size = 8
        compressed_chunks = []

        for i in range(0, features.shape[0], chunk_size):
            chunk = features[i : i + chunk_size]
            if len(chunk) == 1:
                compressed_chunks.append(chunk)
                continue

            chunk_norm = F.normalize(chunk, p=2, dim=-1)
            sim = F.cosine_similarity(chunk_norm[:-1], chunk_norm[1:], dim=-1)

            keep_mask = sim < self.sim_threshold
            kept_tokens = torch.cat([chunk[:1], chunk[1:][keep_mask]])
            compressed_chunks.append(kept_tokens)

        compressed_features = torch.cat(compressed_chunks)

        if len(compressed_features) > self.L_max:
            compressed_features = compressed_features[: self.L_max]

        return compressed_features

    def forward(self, x_s, x_l, label):
        prompts = self.prompt_learner()
        text_features = self.text_encoder(
            prompts,
            self.prompt_learner.tokenized_prompts,
        )[self.num_classes :]

        features = self.feature_encoder(x_l.float())

        selected_features, _ = self.adaptive_token_selection(features, text_features)
        compressed_features = self.spatial_token_compression(selected_features, text_features)

        compressed_features = compressed_features.unsqueeze(0)
        text_features = text_features.unsqueeze(0)

        attended_features = self.cross_attention(text_features, compressed_features, compressed_features)
        final_features = attended_features.mean(1)
        logits = self.classifier(final_features)

        loss = self.loss_ce(logits, label)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]

        return Y_prob, Y_hat, loss
