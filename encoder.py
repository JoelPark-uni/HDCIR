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
            hv = F.normalize(hv, dim=-1)
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
