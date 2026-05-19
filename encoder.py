from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
import torchhd


class HDlm:
    def __init__(
        self,
        feature_dim: int,
        HD_DIM: int = 10000,
        device: str = "cpu",
        HD_cls=torchhd.MAPTensor,
    ) -> None:
        self.HD_DIM = HD_DIM
        self.feature_dim = feature_dim
        self.device = torch.device(device)
        self.HD_cls = HD_cls
        self.encoder = torchhd.MAPTensor.random(feature_dim, HD_DIM, device=self.device)

    @torch.no_grad()
    def encode(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected 2D tensor [N, D], got shape {tuple(x.shape)}")
        if x.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Feature dim mismatch: got {x.shape[-1]}, expected {self.feature_dim}"
            )

        x = x.to(self.device)
        hv = x @ self.encoder.type_as(x)
        if normalize:
            hv = hv.normalize()
        return hv

    @torch.no_grad()
    def bundle_encoded(self, encoded_hv: torch.Tensor) -> torch.Tensor:
        if encoded_hv.ndim != 2:
            raise ValueError(
                f"Expected 2D encoded tensor [N, HD_DIM], got shape {tuple(encoded_hv.shape)}"
            )

        encoded_hv = self.HD_cls(encoded_hv)
        bundled = self.HD_cls.multibundle(encoded_hv)
        return F.normalize(bundled, dim=-1)

    @torch.no_grad()
    def segment_bundle(self, segment_features: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(segment_features, normalize=True)
        return self.bundle_encoded(encoded)


@torch.no_grad()
def encode_index_features_hdc(
    index_features: torch.Tensor,
    hdc_encoder: HDlm,
    chunk_size: int = 2048,
    output_device: str = "cpu",
) -> torch.Tensor:
    chunks = []
    for start_idx in range(0, index_features.shape[0], chunk_size):
        end_idx = min(start_idx + chunk_size, index_features.shape[0])
        hv_chunk = hdc_encoder.encode(index_features[start_idx:end_idx], normalize=True)
        chunks.append(hv_chunk.to(output_device))
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def get_hd_similarity_matrix(query_hv: torch.Tensor, index_hv: torch.Tensor, device: torch.device = None) -> torch.Tensor:
    """Returns a similarity matrix using torchhd.hamming_similarity."""
    # Chunk over queries and indices to avoid OOM
    query_chunk_size = 32
    index_chunk_size = 4096

    if device is None:
        device = query_hv.device

    output_rows = []
    for q_start in range(0, query_hv.shape[0], query_chunk_size):
        q_end = min(q_start + query_chunk_size, query_hv.shape[0])
        q_chunk = query_hv[q_start:q_end].to(device)
        row_chunks = []

        for i_start in range(0, index_hv.shape[0], index_chunk_size):
            i_end = min(i_start + index_chunk_size, index_hv.shape[0])
            i_chunk = index_hv[i_start:i_end].to(device)
            
            if not isinstance(i_chunk, torchhd.MAPTensor) and hasattr(torchhd, 'MAPTensor'):
                i_chunk = torchhd.MAPTensor(i_chunk)

            chunk_sims = []
            for q_vec in q_chunk:
                if not isinstance(q_vec, torchhd.MAPTensor) and hasattr(torchhd, 'MAPTensor'):
                    q_vec = torchhd.MAPTensor(q_vec)
                
                if hasattr(torchhd, 'hamming_similarity'):
                    sims = torchhd.hamming_similarity(q_vec, i_chunk)
                else:
                    sims = torchhd.cosine_similarity(q_vec, i_chunk)
                chunk_sims.append(sims.reshape(-1))
            row_chunks.append(torch.stack(chunk_sims, dim=0).cpu())
            
        output_rows.append(torch.cat(row_chunks, dim=-1))
    return torch.cat(output_rows, dim=0)


import torch.nn as nn

INTENT_MAP = {
    "negation": "NEGATION",
    "addition": "EXPLICIT_ATTRIBUTE",
    "direct_addressing": "EXPLICIT_ATTRIBUTE",
    "compare_change": "EXPLICIT_ATTRIBUTE",
    "spatial_relations_background": "SCENE",
    "viewpoint": "SCENE",
    "comparative_statement": "RELATIVE_MODIFICATION",
    "cardinality": "RELATIVE_MODIFICATION"
}

coarse_classes = sorted(list(set(INTENT_MAP.values())))

class HDClassifier(nn.Module):
    def __init__(self, aspect_dim: int, hd_dim: int = 10000, num_classes: int = 4, seed: int = None, device='cpu'):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.device = device
        self.hd_proj = torchhd.MAPTensor.random(aspect_dim, hd_dim).to(device)
        self.hd_proto = torchhd.MAPTensor.empty(num_classes, hd_dim).zero_().to(device)
    
    def train_proto(self, aspect_vecs, labels):
        aspect_vecs = aspect_vecs.to(self.device)
        labels = labels.to(self.device)
        hd_vecs = (aspect_vecs @ self.hd_proj).normalize()
        for i in range(len(self.hd_proto)):
            class_vecs = hd_vecs[labels[:, i] == 1.0]
            if len(class_vecs) > 0:
                self.hd_proto[i] += torchhd.multibundle(class_vecs)
        self.hd_proto = self.hd_proto.normalize()
    
    def forward(self, aspect_vecs):
        aspect_vecs = aspect_vecs.to(self.device)
        hd_vecs = (aspect_vecs @ self.hd_proj).normalize()
        return torchhd.hamming_similarity(hd_vecs, self.hd_proto)

def apply_threshold_predictions(logits, threshold_offset=20.0):
    """Apply threshold to predictions."""
    mean_sim = logits.float().mean(dim=1, keepdim=True)
    preds = (logits >= mean_sim + threshold_offset).float()
    
    return preds
