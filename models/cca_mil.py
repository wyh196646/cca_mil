# coding=utf-8
import os

import torch
import torch.nn as nn
from torch.nn import functional as F

from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize

from utils.concept_loader import load_concept_bank


DEFAULT_CONCEPT_PROMPT_TEMPLATES = (
    "CLASSNAME.",
    "a photomicrograph showing CLASSNAME.",
    "a photomicrograph of CLASSNAME.",
    "an image of CLASSNAME.",
    "an image showing CLASSNAME.",
    "an example of CLASSNAME.",
    "CLASSNAME is shown.",
    "this is CLASSNAME.",
    "there is CLASSNAME.",
    "a histopathological image showing CLASSNAME.",
    "a histopathological image of CLASSNAME.",
    "a histopathological photograph of CLASSNAME.",
    "a histopathological photograph showing CLASSNAME.",
    "shows CLASSNAME.",
    "presence of CLASSNAME.",
    "CLASSNAME is present.",
    "an H&E stained image of CLASSNAME.",
    "an H&E stained image showing CLASSNAME.",
    "an H&E image showing CLASSNAME.",
    "an H&E image of CLASSNAME.",
    "CLASSNAME, H&E stain.",
    "CLASSNAME, H&E.",
)


def _format_concept_prompt(template, concept_name):
    concept_name = concept_name.replace("_", " ")
    if "CLASSNAME" in template:
        return template.replace("CLASSNAME", concept_name)
    return "{} {}".format(template.rstrip(), concept_name).strip()


def _init_identity_linear(layer):
    nn.init.zeros_(layer.bias)
    if layer.weight.shape[0] == layer.weight.shape[1]:
        nn.init.eye_(layer.weight)
    else:
        nn.init.xavier_uniform_(layer.weight)


def _resolve_conch_ckpt(config):
    candidates = []
    configured = getattr(config, "conch_ckpt_path", None)
    if configured:
        candidates.append(configured)
    candidates.extend([
        "ckg/pytorch_model.bin",
        "ckpts/conch.pth",
    ])

    for path in candidates:
        if path and os.path.isfile(path):
            return path
    raise FileNotFoundError("CONCH checkpoint not found. Tried: {}".format(candidates))


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
    def __init__(self, concept_names, conch_model, prompt_templates=None, n_ctx=0):
        super().__init__()
        self.num_concepts = len(concept_names)
        self.prompt_templates = tuple(prompt_templates or DEFAULT_CONCEPT_PROMPT_TEMPLATES)
        self.num_templates = len(self.prompt_templates)
        n_ctx = int(n_ctx)
        if n_ctx < 0:
            raise ValueError("n_ctx must be non-negative, got {}".format(n_ctx))

        dtype = next(conch_model.parameters()).dtype
        ctx_dim = conch_model.text.ln_final.weight.shape[0]
        self.tokenizer = get_tokenizer()

        if n_ctx > 0:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.ctx = nn.Parameter(ctx_vectors)
        else:
            self.register_parameter("ctx", None)

        concept_names = [name.replace("_", " ") for name in concept_names]
        prompt_texts = []
        ctx_prefix = " ".join(["X"] * n_ctx)
        for concept_name in concept_names:
            for template in self.prompt_templates:
                prompt_text = _format_concept_prompt(template, concept_name)
                if n_ctx > 0:
                    prompt_text = "{} {}".format(ctx_prefix, prompt_text).strip()
                prompt_texts.append(prompt_text)
        self.prompt_texts = prompt_texts

        tokenized_prompts = tokenize(self.tokenizer, prompt_texts)
        if getattr(conch_model.text, "cls_emb", None) is not None:
            tokenized_prompts = tokenized_prompts[:, :-1]
        with torch.no_grad():
            embedding = conch_model.text.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])
        self.n_ctx = n_ctx
        self.register_buffer("tokenized_prompts", tokenized_prompts)

    def forward(self):
        if self.n_ctx == 0:
            return torch.cat([self.token_prefix, self.token_suffix], dim=1)

        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.num_concepts * self.num_templates, -1, -1)
        return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)


class VisualPrototypeEvidence(nn.Module):
    """Learnable visual anchors that softly aggregate patch evidence tokens."""

    def __init__(self, proto_num, proto_dim, tau=0.1, eps=1e-6):
        super().__init__()
        self.visual_prototypes = nn.Parameter(torch.randn(proto_num, proto_dim))
        nn.init.normal_(self.visual_prototypes, std=0.02)
        self.tau = tau
        self.eps = eps

    def forward(self, z):
        prototype = F.normalize(self.visual_prototypes, dim=-1)
        sim = torch.einsum("bnd,kd->bnk", z, prototype) / self.tau
        q = torch.softmax(sim, dim=-1)
        denom = q.sum(dim=1).unsqueeze(-1) + self.eps
        evidence = torch.einsum("bnk,bnd->bkd", q, z) / denom
        evidence = F.normalize(evidence, dim=-1)
        return evidence, q, prototype


class AdaptiveUnbalancedOT(nn.Module):
    """Entropy-regularized unbalanced Sinkhorn solver supporting K != M."""

    def __init__(self, ot_epsilon=0.05, sinkhorn_iter=50, uot_rho_a=0.5, uot_rho_b=0.5, eps=1e-6):
        super().__init__()
        self.ot_epsilon = ot_epsilon
        self.sinkhorn_iter = sinkhorn_iter
        self.uot_rho_a = uot_rho_a
        self.uot_rho_b = uot_rho_b
        self.eps = eps

    def forward(self, cost, a, b):
        cost = cost.clamp(min=0.0, max=2.0)
        kernel = torch.exp(-cost / self.ot_epsilon).clamp_min(self.eps)
        tau_a = self.uot_rho_a / (self.uot_rho_a + self.ot_epsilon)
        tau_b = self.uot_rho_b / (self.uot_rho_b + self.ot_epsilon)

        u = torch.ones_like(a)
        v = torch.ones_like(b)
        for _ in range(self.sinkhorn_iter):
            kv = torch.bmm(kernel, v.unsqueeze(-1)).squeeze(-1).clamp_min(self.eps)
            u = (a / kv).clamp_min(self.eps).pow(tau_a)
            ktu = torch.bmm(kernel.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1).clamp_min(self.eps)
            v = (b / ktu).clamp_min(self.eps).pow(tau_b)

        return u.unsqueeze(-1) * kernel * v.unsqueeze(1)


class DisCommonAOTAssignment(nn.Module):
    def __init__(
        self,
        visual_dim,
        concept_dim,
        align_dim,
        ot_epsilon=0.05,
        sinkhorn_iter=50,
        uot_rho_a=0.5,
        uot_rho_b=0.5,
        eps=1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.visual_proj = nn.Linear(visual_dim, align_dim)
        self.concept_align_proj = nn.Linear(concept_dim, align_dim)
        self.proto_marginal_net = nn.Sequential(nn.LayerNorm(align_dim), nn.Linear(align_dim, 1))
        self.concept_marginal_net = nn.Sequential(nn.LayerNorm(align_dim), nn.Linear(align_dim, 1))
        self.sinkhorn = AdaptiveUnbalancedOT(
            ot_epsilon=ot_epsilon,
            sinkhorn_iter=sinkhorn_iter,
            uot_rho_a=uot_rho_a,
            uot_rho_b=uot_rho_b,
            eps=eps,
        )
        _init_identity_linear(self.visual_proj)
        _init_identity_linear(self.concept_align_proj)

    def forward(self, visual_evidence, t_dis, t_com):
        if t_dis.size(0) == 0:
            raise ValueError("CCA_MIL requires at least one discriminative concept.")
        if t_com is None:
            t_com = t_dis.new_zeros(0, t_dis.size(-1))

        t_all = torch.cat([t_dis, t_com], dim=0)
        num_dis = t_dis.size(0)

        v = self.visual_proj(visual_evidence)
        v = F.normalize(v, dim=-1)
        t_all_proj = self.concept_align_proj(t_all)
        t_all_proj = F.normalize(t_all_proj, dim=-1)
        t_dis_proj = t_all_proj[:num_dis]
        t_com_proj = t_all_proj[num_dis:]

        sim = torch.einsum("bkd,md->bkm", v, t_all_proj)
        cost = (1.0 - sim).clamp(min=0.0, max=2.0)

        a = torch.softmax(self.proto_marginal_net(v).squeeze(-1), dim=-1)
        b = torch.softmax(self.concept_marginal_net(t_all_proj).squeeze(-1), dim=-1)
        b = b.unsqueeze(0).expand(v.size(0), -1)

        transport = self.sinkhorn(cost, a, b)
        p_dis = transport[:, :, :num_dis]
        p_com = transport[:, :, num_dis:]

        denom_dis = p_dis.sum(dim=1).unsqueeze(-1) + self.eps
        z_dis = torch.einsum("bkm,bkd->bmd", p_dis, v) / denom_dis
        z_dis = F.normalize(z_dis, dim=-1)

        if p_com.size(-1) > 0:
            denom_com = p_com.sum(dim=1).unsqueeze(-1) + self.eps
            z_com = torch.einsum("bkm,bkd->bmd", p_com, v) / denom_com
            z_com = F.normalize(z_com, dim=-1)
            com_score = p_com.sum(dim=-1)
        else:
            z_com = v.new_zeros(v.size(0), 0, v.size(-1))
            com_score = torch.zeros_like(p_dis.sum(dim=-1))

        dis_score = p_dis.sum(dim=-1)

        return {
            "Z_dis": z_dis,
            "Z_com": z_com,
            "transport": transport,
            "P_dis": p_dis,
            "P_com": p_com,
            "dis_score": dis_score,
            "com_score": com_score,
            "V": v,
            "T_dis_proj": t_dis_proj,
            "T_com_proj": t_com_proj,
        }


class CCA_MIL(nn.Module):
    def __init__(self, config, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        self.input_dim = config.input_size
        self.feature_dim = getattr(config, "feature_dim", 512)
        self.align_dim = getattr(config, "align_dim", self.feature_dim)
        self.dropout = getattr(config, "dropout", 0.0)
        self.eps = getattr(config, "eps", 1e-6)

        proto_num = getattr(config, "num_visual_prototypes", None)
        if proto_num is None:
            proto_num = getattr(config, "prototype_number", getattr(config, "cluster_k", 8))
        self.num_visual_prototypes = int(proto_num)
        if self.num_visual_prototypes < 1:
            raise ValueError("num_visual_prototypes must be positive, got {}".format(self.num_visual_prototypes))
        self.proto_tau = getattr(config, "proto_tau", 0.1)
        self.ot_epsilon = getattr(config, "ot_epsilon", 0.05)
        self.sinkhorn_iter = getattr(config, "sinkhorn_iter", 50)
        self.uot_rho_a = getattr(config, "uot_rho_a", 0.5)
        self.uot_rho_b = getattr(config, "uot_rho_b", 0.5)
        self.concept_pooling = getattr(config, "concept_pooling", "attention")
        self.lambda_contrast = getattr(config, "lambda_contrast", None)
        if self.lambda_contrast is None:
            self.lambda_contrast = getattr(config, "lambda_con", 0.0)
        self.lambda_con = self.lambda_contrast
        self.lambda_div = getattr(config, "lambda_div", 0.0) or 0.0
        self.contrast_tau = getattr(config, "contrast_tau", None)
        if self.contrast_tau is None:
            self.contrast_tau = getattr(config, "tau", 0.07)
        self.tau = self.contrast_tau
        self.max_train_patches = int(getattr(config, "max_train_patches", 0) or 0)
        self.max_eval_patches = int(getattr(config, "max_eval_patches", 0) or 0)
        self.concept_logit_weight = float(getattr(config, "concept_logit_weight", 0.0) or 0.0)
        self.concept_logit_tau = float(getattr(config, "concept_logit_tau", 1.0) or 1.0)
        self.store_explanations = getattr(config, "store_explanations", False)
        self.last_explanations = None
        self._concept_feature_cache = None

        concept_bank = load_concept_bank(
            config.concept_bank_path,
            class_names=config.class_names,
            common_weight=getattr(config, "common_concept_weight", 0.3),
        )
        if len(concept_bank) != self.num_classes:
            raise ValueError(
                "Concept bank has {} classes, but num_classes is {}".format(
                    len(concept_bank), self.num_classes
                )
            )

        concept_texts, concept_weights, concept_type_mask = self._build_concept_metadata(concept_bank)

        conch_model_cfg = "conch_ViT-B-16"
        conch_model, _ = create_model_from_pretrained(conch_model_cfg, _resolve_conch_ckpt(config))
        conch_model = conch_model.float().eval()

        prompt_templates = getattr(config, "concept_prompt_templates", DEFAULT_CONCEPT_PROMPT_TEMPLATES)
        prompt_template_count = int(getattr(config, "concept_prompt_template_count", 0) or 0)
        if prompt_template_count > 0:
            prompt_templates = tuple(prompt_templates[:prompt_template_count])
        prompt_n_ctx = getattr(config, "concept_prompt_n_ctx", 0)
        self.prompt_learner = PromptLearner(
            concept_texts,
            conch_model,
            prompt_templates=prompt_templates,
            n_ctx=prompt_n_ctx,
        )
        self.text_encoder = TextEncoder(conch_model)
        self.text_encoder.requires_grad_(False)
        if not getattr(config, "train_concept_prompt", False):
            self.prompt_learner.requires_grad_(False)

        self.register_buffer("concept_weights", torch.tensor(concept_weights, dtype=torch.float32))
        self.register_buffer("concept_type_mask", torch.tensor(concept_type_mask, dtype=torch.bool))

        self.patch_projector = nn.Sequential(
            nn.Linear(self.input_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        self.visual_evidence = VisualPrototypeEvidence(
            proto_num=self.num_visual_prototypes,
            proto_dim=self.feature_dim,
            tau=self.proto_tau,
            eps=self.eps,
        )
        self.assignment = DisCommonAOTAssignment(
            visual_dim=self.feature_dim,
            concept_dim=self.feature_dim,
            align_dim=self.align_dim,
            ot_epsilon=self.ot_epsilon,
            sinkhorn_iter=self.sinkhorn_iter,
            uot_rho_a=self.uot_rho_a,
            uot_rho_b=self.uot_rho_b,
            eps=self.eps,
        )
        self.concept_attn = nn.Sequential(nn.LayerNorm(self.align_dim), nn.Linear(self.align_dim, 1))
        self.concept_weight = nn.Parameter(torch.zeros(self.num_dis_concepts))
        self.classifier = nn.Linear(self.align_dim, self.num_classes)
        self.concept_logit_scale = nn.Parameter(torch.tensor(1.0))
        self.loss_ce = nn.CrossEntropyLoss()

    def train(self, mode=True):
        if mode:
            self._concept_feature_cache = None
        return super().train(mode)

    def _build_concept_metadata(self, concept_bank):
        self.class_names = [entry["class_name"] for entry in concept_bank]
        self.concept_slices = []

        concept_texts = []
        concept_weights = []
        concept_type_mask = []
        dis_indices = []
        com_indices = []
        dis_class_labels = []
        com_class_labels = []

        start = 0
        for class_id, entry in enumerate(concept_bank):
            count = len(entry["concepts"])
            end = start + count
            class_indices = list(range(start, end))
            self.concept_slices.append(slice(start, end))

            local_dis = [idx for idx, concept_type in enumerate(entry["types"]) if concept_type == "discriminative"]
            local_com = [idx for idx, concept_type in enumerate(entry["types"]) if concept_type == "common"]
            class_dis_indices = [start + idx for idx in local_dis] or class_indices
            class_com_indices = [start + idx for idx in local_com]

            dis_indices.extend(class_dis_indices)
            dis_class_labels.extend([class_id] * len(class_dis_indices))
            com_indices.extend(class_com_indices)
            com_class_labels.extend([class_id] * len(class_com_indices))

            concept_texts.extend(entry["concepts"])
            concept_weights.extend(entry["weights"])
            concept_type_mask.extend([concept_type == "discriminative" for concept_type in entry["types"]])
            start = end

        self.num_dis_concepts = len(dis_indices)
        self.num_com_concepts = len(com_indices)
        if self.num_dis_concepts == 0:
            raise ValueError("Concept bank must provide at least one usable concept.")

        self.register_buffer("dis_concept_indices", torch.tensor(dis_indices, dtype=torch.long))
        self.register_buffer("com_concept_indices", torch.tensor(com_indices, dtype=torch.long))
        self.register_buffer("dis_class_labels", torch.tensor(dis_class_labels, dtype=torch.long))
        self.register_buffer("com_class_labels", torch.tensor(com_class_labels, dtype=torch.long))
        return concept_texts, concept_weights, concept_type_mask

    def _encode_concepts(self):
        prompt_grad = any(p.requires_grad for p in self.prompt_learner.parameters())
        use_cache = (not self.training) or (not prompt_grad)
        if use_cache and self._concept_feature_cache is not None:
            return self._concept_feature_cache

        prompts = self.prompt_learner()
        text_grad = any(p.requires_grad for p in self.text_encoder.parameters())
        with torch.set_grad_enabled(text_grad or prompt_grad):
            prompt_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)

        prompt_features = F.normalize(prompt_features.float(), dim=-1)
        concept_features = prompt_features.view(
            self.prompt_learner.num_concepts,
            self.prompt_learner.num_templates,
            -1,
        ).mean(dim=1)
        concept_features = F.normalize(concept_features, dim=-1)
        if use_cache:
            self._concept_feature_cache = concept_features.detach()
            return self._concept_feature_cache
        return concept_features

    def _as_3d(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        elif x.dim() == 3:
            pass
        elif x.dim() == 4:
            x = x.reshape(x.size(0), x.size(1) * x.size(2), x.size(3))
        else:
            raise ValueError("Expected feature tensor with shape [N, D], [B, N, D], or [B, K1, K2, D], got {}".format(tuple(x.shape)))

        if x.size(1) == 0:
            raise ValueError("CCA_MIL received an empty WSI feature bag.")
        return x

    def _sample_patches(self, x):
        max_patches = self.max_train_patches if self.training else self.max_eval_patches
        if max_patches <= 0 or x.size(1) <= max_patches:
            return x

        if self.training:
            idx = torch.randperm(x.size(1), device=x.device)[:max_patches]
            idx, _ = torch.sort(idx)
        else:
            idx = torch.linspace(0, x.size(1) - 1, steps=max_patches, device=x.device).round().long()
        return x.index_select(1, idx)

    def _global_concept_bank(self, concept_features):
        dis_indices = self.dis_concept_indices.to(concept_features.device)
        com_indices = self.com_concept_indices.to(concept_features.device)
        t_dis = concept_features.index_select(0, dis_indices)
        if com_indices.numel() > 0:
            t_com = concept_features.index_select(0, com_indices)
        else:
            t_com = concept_features.new_zeros(0, concept_features.size(-1))
        return t_dis, t_com

    def _pool_discriminative_evidence(self, z_dis):
        if z_dis.size(1) == 0:
            return z_dis.new_zeros(z_dis.size(0), z_dis.size(-1))

        if self.concept_pooling == "mean":
            return z_dis.mean(dim=1)
        if self.concept_pooling == "learnable":
            weight = torch.softmax(self.concept_weight[: z_dis.size(1)], dim=0)
            return torch.sum(weight.view(1, -1, 1) * z_dis, dim=1)
        if self.concept_pooling == "attention":
            attn_logits = self.concept_attn(z_dis).squeeze(-1)
            attn = torch.softmax(attn_logits, dim=1)
            return torch.sum(attn.unsqueeze(-1) * z_dis, dim=1)

        raise ValueError("Unknown concept_pooling '{}'. Use mean, learnable, or attention.".format(self.concept_pooling))

    def _concept_guided_logits(self, z_dis, t_dis_proj, concept_dis_score):
        if self.concept_logit_weight <= 0:
            return z_dis.new_zeros(z_dis.size(0), self.num_classes)

        dis_labels = self.dis_class_labels.to(z_dis.device)
        sim = torch.sum(
            F.normalize(z_dis, dim=-1) * F.normalize(t_dis_proj.unsqueeze(0), dim=-1),
            dim=-1,
        ) / self.concept_logit_tau

        logits = []
        for class_id in range(self.num_classes):
            idx = (dis_labels == class_id).nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                logits.append(sim.new_zeros(sim.size(0)))
                continue
            score = sim.index_select(1, idx)
            mass = concept_dis_score.index_select(1, idx).clamp_min(self.eps)
            logits.append((score * mass).sum(dim=1) / mass.sum(dim=1).clamp_min(self.eps))

        logits = torch.stack(logits, dim=1)
        return self.concept_logit_scale.clamp(0.0, 10.0) * logits

    def _class_aware_contrastive_loss(self, z_dis, t_dis_proj, t_com_proj, labels, logits):
        if self.lambda_contrast <= 0:
            return logits.new_tensor(0.0)

        labels = labels.view(-1).long().to(z_dis.device)
        dis_labels = self.dis_class_labels.to(z_dis.device)
        com_labels = self.com_class_labels.to(z_dis.device)
        losses = []

        for batch_id in range(z_dis.size(0)):
            label = labels[batch_id]
            idx_pos = (dis_labels == label).nonzero(as_tuple=False).flatten()
            idx_neg_dis = (dis_labels != label).nonzero(as_tuple=False).flatten()

            if t_com_proj.size(0) == 0:
                idx_neg_com = torch.empty(0, device=z_dis.device, dtype=torch.long)
            elif com_labels.numel() == t_com_proj.size(0):
                idx_neg_com = (com_labels == label).nonzero(as_tuple=False).flatten()
            else:
                idx_neg_com = torch.arange(t_com_proj.size(0), device=z_dis.device)

            if idx_pos.numel() == 0:
                losses.append(logits.new_tensor(0.0))
                continue

            z_pos = F.normalize(z_dis[batch_id, idx_pos], dim=-1)
            t_pos = F.normalize(t_dis_proj[idx_pos], dim=-1)
            negatives = []
            if idx_neg_dis.numel() > 0:
                negatives.append(t_dis_proj[idx_neg_dis])
            if idx_neg_com.numel() > 0:
                negatives.append(t_com_proj[idx_neg_com])

            if not negatives:
                losses.append(logits.new_tensor(0.0))
                continue

            t_neg = F.normalize(torch.cat(negatives, dim=0), dim=-1)
            pos_logits = torch.sum(z_pos * t_pos, dim=-1, keepdim=True) / self.contrast_tau
            neg_logits = torch.matmul(z_pos, t_neg.transpose(0, 1)) / self.contrast_tau
            contrast_logits = torch.cat([pos_logits, neg_logits], dim=1)
            targets = torch.zeros(z_pos.size(0), device=z_dis.device, dtype=torch.long)
            losses.append(F.cross_entropy(contrast_logits, targets))

        return torch.stack(losses).mean() if losses else logits.new_tensor(0.0)

    def _diversity_loss(self, z_dis, labels, logits):
        if self.lambda_div <= 0:
            return logits.new_tensor(0.0)

        labels = labels.view(-1).long().to(z_dis.device)
        dis_labels = self.dis_class_labels.to(z_dis.device)
        losses = []

        for batch_id in range(z_dis.size(0)):
            label = labels[batch_id]
            idx = (dis_labels == label).nonzero(as_tuple=False).flatten()
            z_y = z_dis[batch_id, idx]
            if z_y.size(0) <= 1:
                losses.append(logits.new_tensor(0.0))
                continue

            z_y = F.normalize(z_y, dim=-1)
            gram = torch.matmul(z_y, z_y.transpose(0, 1))
            eye = torch.eye(z_y.size(0), device=z_y.device, dtype=z_y.dtype)
            losses.append(((gram * (1.0 - eye)) ** 2).sum() / (z_y.size(0) * (z_y.size(0) - 1) + self.eps))

        return torch.stack(losses).mean() if losses else logits.new_tensor(0.0)

    def _detach_for_explanations(self, outputs):
        keys = (
            "logits",
            "concept_logits",
            "y_prob",
            "y_hat",
            "patch_embed",
            "visual_evidence",
            "Z_dis",
            "Z_com",
            "z_dis_pool",
            "transport",
            "P_dis",
            "P_com",
            "patch_proto_assign",
            "patch_dis_score",
            "patch_com_score",
            "concept_dis_score",
            "concept_com_score",
            "T_dis_proj",
            "T_com_proj",
        )
        explanations = {"class_names": list(self.class_names)}
        for key in keys:
            value = outputs.get(key)
            if torch.is_tensor(value):
                explanations[key] = value.detach().cpu()
        explanations["dis_class_labels"] = self.dis_class_labels.detach().cpu()
        explanations["com_class_labels"] = self.com_class_labels.detach().cpu()
        return explanations

    def forward(self, x_s, x_l=None, label=None, legacy_return=True):
        if x_l is None:
            x_l = x_s

        x_l = self._as_3d(x_l).float()
        x_l = self._sample_patches(x_l)
        patch_embed = self.patch_projector(x_l)
        patch_embed = F.normalize(patch_embed, dim=-1)

        concept_features = self._encode_concepts()
        t_dis, t_com = self._global_concept_bank(concept_features)

        visual_evidence, patch_proto_assign, _ = self.visual_evidence(patch_embed)
        assign = self.assignment(visual_evidence, t_dis, t_com)

        z_dis = assign["Z_dis"]
        z_dis_pool = self._pool_discriminative_evidence(z_dis)
        concept_dis_score = assign["P_dis"].sum(dim=1)
        concept_logits = self._concept_guided_logits(
            z_dis,
            assign["T_dis_proj"],
            concept_dis_score,
        )
        logits = self.classifier(z_dis_pool) + self.concept_logit_weight * concept_logits
        y_prob = F.softmax(logits, dim=1)
        y_hat = torch.argmax(y_prob, dim=1)

        patch_dis_score = torch.einsum("bnk,bk->bn", patch_proto_assign, assign["dis_score"])
        patch_com_score = torch.einsum("bnk,bk->bn", patch_proto_assign, assign["com_score"])
        concept_com_score = assign["P_com"].sum(dim=1)

        outputs = {
            "logits": logits,
            "concept_logits": concept_logits,
            "y_prob": y_prob,
            "y_hat": y_hat,
            "patch_embed": patch_embed,
            "visual_evidence": visual_evidence,
            "Z_dis": z_dis,
            "Z_com": assign["Z_com"],
            "z_dis_pool": z_dis_pool,
            "transport": assign["transport"],
            "P_dis": assign["P_dis"],
            "P_com": assign["P_com"],
            "patch_proto_assign": patch_proto_assign,
            "patch_dis_score": patch_dis_score,
            "patch_com_score": patch_com_score,
            "concept_dis_score": concept_dis_score,
            "concept_com_score": concept_com_score,
            "T_dis_proj": assign["T_dis_proj"],
            "T_com_proj": assign["T_com_proj"],
        }

        loss_total = logits.new_tensor(0.0)
        if label is not None:
            target = label.view(-1).long().to(logits.device)
            loss_cls = self.loss_ce(logits, target)
            loss_contrast = self._class_aware_contrastive_loss(
                z_dis,
                assign["T_dis_proj"],
                assign["T_com_proj"],
                target,
                logits,
            )
            loss_div = self._diversity_loss(z_dis, target, logits)
            loss_total = loss_cls + self.lambda_contrast * loss_contrast + self.lambda_div * loss_div
            outputs.update({
                "loss_total": loss_total,
                "loss_cls": loss_cls,
                "loss_contrast": loss_contrast,
                "loss_div": loss_div,
            })

        if self.store_explanations:
            self.last_explanations = self._detach_for_explanations(outputs)

        if legacy_return:
            return y_prob, y_hat, loss_total
        return outputs
