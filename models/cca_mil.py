# coding=utf-8
import torch
import torch.nn as nn
from torch.nn import functional as F
import os

from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
from models.concept_guided_aggregator import ConceptGuidedAggregator
from utils.concept_loader import load_concept_bank


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
    def __init__(self, concept_names, conch_model):
        super().__init__()
        n_cls = len(concept_names)
        n_ctx = 16
        dtype = next(conch_model.parameters()).dtype
        ctx_dim = conch_model.text.ln_final.weight.shape[0]
        self.tokenizer = get_tokenizer()

        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

        concept_names = [name.replace("_", " ") for name in concept_names]
        tokenized_prompts = tokenize(self.tokenizer, concept_names)
        if getattr(conch_model.text, "cls_emb", None) is not None:
            tokenized_prompts = tokenized_prompts[:, :-1]
        with torch.no_grad():
            embedding = conch_model.text.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.register_buffer("tokenized_prompts", tokenized_prompts)

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)


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


class CCA_MIL(nn.Module):
    def __init__(self, config, num_classes=3):
        super(CCA_MIL, self).__init__()
        self.num_classes = num_classes
        self.input_dim = config.input_size
        self.feature_dim = getattr(config, "feature_dim", 512)
        self.cluster_k = config.cluster_k
        self.kmeans_iters = config.kmeans_iters
        self.normalize_kmeans = config.normalize_kmeans
        self.min_cluster_size = config.min_cluster_size
        self.top_r = config.selection_top_r
        self.alpha = config.concept_alpha
        self.lambda_con = config.lambda_con
        self.lambda_div = config.lambda_div
        self.tau = config.tau
        self.store_explanations = getattr(config, "store_explanations", False)
        self.last_explanations = None

        concept_bank = load_concept_bank(
            config.concept_bank_path,
            class_names=config.class_names,
            common_weight=config.common_concept_weight,
        )
        if len(concept_bank) != self.num_classes:
            raise ValueError(
                "Concept bank has {} classes, but num_classes is {}".format(
                    len(concept_bank), self.num_classes
                )
            )
        self.class_names = [entry["class_name"] for entry in concept_bank]
        self.concept_slices = []
        self.discriminative_indices = []
        concept_texts = []
        concept_weights = []
        concept_type_mask = []

        start = 0
        for entry in concept_bank:
            count = len(entry["concepts"])
            end = start + count
            self.concept_slices.append(slice(start, end))
            self.discriminative_indices.append([
                idx for idx, concept_type in enumerate(entry["types"])
                if concept_type == "discriminative"
            ])
            concept_texts.extend(entry["concepts"])
            concept_weights.extend(entry["weights"])
            concept_type_mask.extend([1 if t == "discriminative" else 0 for t in entry["types"]])
            start = end

        self.register_buffer("concept_weights", torch.tensor(concept_weights, dtype=torch.float32))
        self.register_buffer("concept_type_mask", torch.tensor(concept_type_mask, dtype=torch.bool))

        conch_model_cfg = "conch_ViT-B-16"
        conch_model, _ = create_model_from_pretrained(conch_model_cfg, _resolve_conch_ckpt(config))
        conch_model = conch_model.float().eval()

        self.prompt_learner = PromptLearner(concept_texts, conch_model)
        self.text_encoder = TextEncoder(conch_model)
        self.text_encoder.requires_grad_(False)
        if not getattr(config, "train_concept_prompt", False):
            self.prompt_learner.requires_grad_(False)

        self.patch_projector = nn.Linear(self.input_dim, self.feature_dim)
        self.text_projector = nn.Linear(self.feature_dim, self.feature_dim)
        self.global_projector = nn.Linear(self.feature_dim, self.feature_dim)
        _init_identity_linear(self.patch_projector)
        _init_identity_linear(self.text_projector)
        _init_identity_linear(self.global_projector)

        self.aggregator = ConceptGuidedAggregator(
            feature_dim=self.feature_dim,
            num_heads=getattr(config, "num_attention_heads", 8),
            dropout=getattr(config, "attn_dropout", 0.0),
        )
        self.logit_head = nn.Linear(self.feature_dim, 1)
        self.loss_ce = nn.CrossEntropyLoss()

    def _encode_concepts(self):
        prompts = self.prompt_learner()
        text_grad = any(p.requires_grad for p in self.text_encoder.parameters())
        prompt_grad = any(p.requires_grad for p in self.prompt_learner.parameters())
        with torch.set_grad_enabled(text_grad or prompt_grad):
            concept_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
        return self.text_projector(concept_features.float())

    def _as_2d(self, x):
        if x.dim() == 3 and x.size(0) == 1:
            x = x.squeeze(0)
        if x.dim() != 2:
            raise ValueError("Expected feature tensor with shape [N, D], got {}".format(tuple(x.shape)))
        return x

    def _run_kmeans(self, features):
        num_patches = features.size(0)
        if num_patches == 0:
            raise ValueError("CCA_MIL received an empty WSI feature bag.")

        k = min(self.cluster_k, num_patches)
        if k == 1:
            assignments = torch.zeros(num_patches, dtype=torch.long, device=features.device)
            return assignments, features.mean(dim=0, keepdim=True)

        cluster_features = F.normalize(features, dim=-1) if self.normalize_kmeans else features
        init_ids = torch.linspace(0, num_patches - 1, steps=k, device=features.device).long()

        with torch.no_grad():
            centers = cluster_features.index_select(0, init_ids).clone()
            assignments = torch.zeros(num_patches, dtype=torch.long, device=features.device)
            for _ in range(self.kmeans_iters):
                distances = torch.cdist(cluster_features, centers)
                assignments = distances.argmin(dim=1)
                new_centers = []
                for cluster_id in range(k):
                    mask = assignments == cluster_id
                    if mask.any():
                        new_centers.append(cluster_features[mask].mean(dim=0))
                    else:
                        new_centers.append(centers[cluster_id])
                centers = torch.stack(new_centers, dim=0)

            assignments = self._merge_small_clusters(cluster_features, assignments, centers)

        assignments, centers = self._compact_clusters(features, assignments)
        return assignments, centers

    def _merge_small_clusters(self, cluster_features, assignments, centers):
        if self.min_cluster_size <= 1 or centers.size(0) <= 1:
            return assignments

        counts = torch.bincount(assignments, minlength=centers.size(0))
        valid = counts >= self.min_cluster_size
        if valid.all() or not valid.any():
            return assignments

        valid_ids = torch.where(valid)[0]
        valid_centers = centers.index_select(0, valid_ids)
        merged = assignments.clone()
        for cluster_id in torch.where(~valid)[0]:
            patch_ids = torch.where(assignments == cluster_id)[0]
            if patch_ids.numel() == 0:
                continue
            distances = torch.cdist(cluster_features.index_select(0, patch_ids), valid_centers)
            nearest = valid_ids.index_select(0, distances.argmin(dim=1))
            merged[patch_ids] = nearest
        return merged

    def _compact_clusters(self, features, assignments):
        unique_ids = torch.unique(assignments, sorted=True)
        remapped = torch.empty_like(assignments)
        centers = []
        for new_id, old_id in enumerate(unique_ids):
            mask = assignments == old_id
            remapped[mask] = new_id
            centers.append(features[mask].mean(dim=0))
        return remapped, torch.stack(centers, dim=0)

    def _select_patches_for_class(self, features, centers, assignments, class_concepts, alignment):
        assigned_concepts = alignment.argmax(dim=1)
        selected_indices = []
        scores_by_index = []

        norm_features = F.normalize(features, dim=-1)
        norm_centers = F.normalize(centers, dim=-1)
        norm_concepts = F.normalize(class_concepts, dim=-1)

        for cluster_id in range(centers.size(0)):
            patch_ids = torch.where(assignments == cluster_id)[0]
            if patch_ids.numel() == 0:
                continue

            concept_id = assigned_concepts[cluster_id]
            concept_scores = norm_features.index_select(0, patch_ids) @ norm_concepts[concept_id]
            cluster_scores = norm_features.index_select(0, patch_ids) @ norm_centers[cluster_id]
            scores = self.alpha * concept_scores + (1.0 - self.alpha) * cluster_scores

            keep = min(self.top_r, patch_ids.numel())
            top_local = torch.topk(scores, keep).indices
            kept_patch_ids = patch_ids.index_select(0, top_local)
            selected_indices.append(kept_patch_ids)
            scores_by_index.append(scores.index_select(0, top_local))

        selected_indices = torch.cat(selected_indices, dim=0)
        selection_scores = torch.cat(scores_by_index, dim=0)
        return selected_indices, selection_scores, assigned_concepts

    def _contrastive_loss(self, features, concept_features, label):
        prototypes = []
        for class_slice in self.concept_slices:
            class_concepts = concept_features[class_slice]
            class_weights = self.concept_weights[class_slice].to(class_concepts.device).unsqueeze(-1)
            prototype = (class_concepts * class_weights).sum(dim=0) / class_weights.sum().clamp_min(1e-6)
            prototypes.append(prototype)

        prototypes = F.normalize(torch.stack(prototypes, dim=0), dim=-1)
        slide_feature = self.global_projector(features.mean(dim=0, keepdim=True))
        slide_feature = F.normalize(slide_feature, dim=-1)
        logits = slide_feature @ prototypes.t()
        logits = logits / self.tau
        return F.cross_entropy(logits, label)

    def _diversity_loss(self, alignments):
        losses = []
        for class_id, alignment in enumerate(alignments):
            local_ids = self.discriminative_indices[class_id]
            if len(local_ids) < 2 or alignment.size(0) < 2:
                continue

            ids = torch.tensor(local_ids, dtype=torch.long, device=alignment.device)
            evidence = F.softmax(alignment.index_select(1, ids).transpose(0, 1), dim=-1)
            evidence = F.normalize(evidence, dim=-1)
            sim = evidence @ evidence.t()
            off_diag = sim[~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)]
            losses.append(off_diag.mean())

        if not losses:
            return alignments[0].new_tensor(0.0)
        return torch.stack(losses).mean()

    def _build_explanations(self, alignments, selected_indices, selection_scores, assigned_concepts):
        explanations = []
        for class_id, alignment in enumerate(alignments):
            explanations.append({
                "class_name": self.class_names[class_id],
                "alignment": alignment.detach().cpu(),
                "concept_evidence": alignment.max(dim=0).values.detach().cpu(),
                "selected_indices": selected_indices[class_id].detach().cpu(),
                "selection_scores": selection_scores[class_id].detach().cpu(),
                "assigned_concepts": assigned_concepts[class_id].detach().cpu(),
            })
        return explanations

    def forward(self, x_s, x_l, label):
        del x_s
        x_l = self._as_2d(x_l)
        features = self.patch_projector(x_l.float())
        concept_features = self._encode_concepts()

        assignments, centers = self._run_kmeans(features)
        logits = []
        alignments = []
        selected_indices_by_class = []
        selection_scores_by_class = []
        assigned_concepts_by_class = []

        norm_centers = F.normalize(centers, dim=-1)

        for class_id, class_slice in enumerate(self.concept_slices):
            class_concepts = concept_features[class_slice]
            class_weights = self.concept_weights[class_slice].to(features.device)
            alignment = norm_centers @ F.normalize(class_concepts, dim=-1).t()

            selected_indices, selection_scores, assigned_concepts = self._select_patches_for_class(
                features, centers, assignments, class_concepts, alignment
            )
            selected_features = features.index_select(0, selected_indices)
            class_feature, _, _ = self.aggregator(class_concepts, selected_features, class_weights)

            logits.append(self.logit_head(class_feature).squeeze(-1))
            alignments.append(alignment)
            selected_indices_by_class.append(selected_indices)
            selection_scores_by_class.append(selection_scores)
            assigned_concepts_by_class.append(assigned_concepts)

        logits = torch.stack(logits, dim=0).unsqueeze(0)
        loss_cls = self.loss_ce(logits, label)
        loss_con = self._contrastive_loss(features, concept_features, label)
        loss_div = self._diversity_loss(alignments)
        loss = loss_cls + self.lambda_con * loss_con + self.lambda_div * loss_div

        if self.store_explanations:
            self.last_explanations = self._build_explanations(
                alignments,
                selected_indices_by_class,
                selection_scores_by_class,
                assigned_concepts_by_class,
            )

        y_prob = F.softmax(logits, dim=1)
        y_hat = torch.topk(y_prob, 1, dim=1)[1]
        return y_prob, y_hat, loss
