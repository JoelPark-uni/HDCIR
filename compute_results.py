import json
import os
import pickle
import sys
from pathlib import Path
from typing import List, Dict, Union, Optional

import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import clip
import tqdm
from encoder import get_hd_similarity_matrix, encode_index_features_hdc


import clip
import torchhd

# Allow importing hdc_splice helpers (resolve_model_tag, default_db_path) without
# duplicating the tag-resolution table here. hdc_splice.py lives one directory up.
_HDCIR_ROOT = Path(__file__).resolve().parent.parent
if str(_HDCIR_ROOT) not in sys.path:
    sys.path.insert(0, str(_HDCIR_ROOT))


# Re-export the canonical loader from hdc_splice so there's a single source of truth
# for the DB pickle schema. `variant='targetpad'` matches the experimentally winning
# `targetpad_centered` recipe (see splice_vit16_analysis.md).
from hdc_splice import load_db_image_means as _load_db_image_means


def load_db_image_means(db_emb_path, expected_clip_model=None, variant='targetpad'):
    return _load_db_image_means(db_emb_path, expected_clip_model=expected_clip_model, variant=variant)

def build_splice_base(clip_model, device, hd_dim=10000, vocab_path="/workspace/joel/HDCIR/SpLiCE/data/vocab/laion.txt", num_concepts=10000):
    import os
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocab file not found at {vocab_path}")

    # 1. Vocab 읽기
    with open(vocab_path, "r") as f:
        all_lines = [line.strip() for line in f.readlines() if line.strip()]
        basis_names = all_lines[-num_concepts:] if len(all_lines) >= num_concepts else all_lines

    # 2. CLIP 텍스트 인코딩
    print(f"Encoding {len(basis_names)} concepts for SpLiCE basis...")
    concepts = []
    
    # tokenizer 가져오기
    if hasattr(clip_model, 'tokenizer'):
        tokenizer = clip_model.tokenizer
    else:
        import clip
        tokenizer = lambda texts: clip.tokenize(texts, truncate=True)

    with torch.no_grad():
        batch_size = 256
        for i in range(0, len(basis_names), batch_size):
            batch_texts = basis_names[i:i+batch_size]
            tokens = tokenizer(batch_texts).to(device)
            embeddings = clip_model.encode_text(tokens)
            concepts.append(embeddings)

    concepts = torch.cat(concepts, dim=0).float()
    
    # 3. SpLiCE 정규화 로직
    concepts = torch.nn.functional.normalize(concepts, p=2, dim=1)
    basis_mean = concepts.mean(dim=0)
    concepts = concepts - basis_mean
    basis_embeddings = torch.nn.functional.normalize(concepts, p=2, dim=1)
    
    clip_dim = basis_embeddings.shape[1]
    encoder = torchhd.MAPTensor.random(clip_dim, hd_dim).to(device)
    basis_hd = torchhd.MAPTensor(basis_embeddings @ encoder)
    basis_hd = torchhd.normalize(basis_hd)
    
    return basis_hd, basis_embeddings, basis_mean, encoder, basis_names

def splice_decompose(query_clip, basis_hd, basis_mean, encoder, top_k=100, num_iterations=3, lambda_sparse=0.25, lr=0.1, temp=1.0, batch_size=256):
    all_weights, all_indices, all_query_hd = [], [], []
    for i in range(0, len(query_clip), batch_size):
        batch_clip = query_clip[i:i+batch_size]
        
        # batch_clip = torch.nn.functional.normalize(batch_clip, p=2, dim=1)
        batch_clip = batch_clip - basis_mean  # Centering with the same mean used for the basis (modality gap mitigation)
        batch_clip = torch.nn.functional.normalize(batch_clip, p=2, dim=1)  # Re-normalize after centering

        query_hd = torchhd.MAPTensor(batch_clip @ encoder)
        query_hd = torchhd.normalize(query_hd)
        
        similarities = torchhd.hamming_similarity(query_hd, basis_hd)
        current_k = min(top_k, basis_hd.shape[0])
        
        values, indices = torch.topk(similarities, current_k, dim=-1)
        HD_DIM = basis_hd.shape[1]
        
        weights = values.clone() / HD_DIM
        weights = torch.softmax(weights / temp, dim=-1)
        
        for _ in range(num_iterations):
            rel_hvs = basis_hd[indices]
            req = torch.bmm(weights.unsqueeze(1), rel_hvs).squeeze(1)
            req = torch.nn.functional.normalize(req, p=2, dim=-1)
            
            residual = query_hd - req
            grad = -torch.bmm(rel_hvs, residual.unsqueeze(-1)).squeeze(-1)
            
            weights = weights - lr * grad
            weights = torch.clamp(weights - (lr * lambda_sparse), min=0.0)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)
            weights = torch.softmax(weights / temp, dim=-1)
            
        all_weights.append(weights.cpu())
        all_indices.append(indices.cpu())
        all_query_hd.append(query_hd.cpu())
        
    return torch.cat(all_weights, 0).to(query_clip.device), \
           torch.cat(all_indices, 0).to(query_clip.device), \
           torch.cat(all_query_hd, 0).to(query_clip.device)

@torch.no_grad()
def fiq(
    device: torch.device,
    predicted_features: torch.Tensor,
    positive_features: torch.Tensor,
    negative_features: torch.Tensor,
    target_names: List,
    index_features: torch.Tensor,
    index_names: List,
    split: str='val',
    **kwargs
) -> Dict[str, float]:
    """
    Compute the retrieval metrics on the Fashion-IQ validation set fiven the dataset, pseudo tokens and reference names.
    Computes Recall@10 and Recall@50.
    """
    if hdc_index_features is not None:
        hd_encoder = kwargs.get('hd_encoder')
        predicted_hv = encode_index_features_hdc(predicted_features, hd_encoder, output_device=device)
        similarities = get_hd_similarity_matrix(predicted_hv, hdc_index_features.to(device)).cpu()
    else:
        # Move the features to the device
        index_features = torch.nn.functional.normalize(index_features).to(device)
        predicted_features = torch.nn.functional.normalize(predicted_features).to(device)

    # Compute the distances
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

            # Check if the target names are in the top 10 and top 50
            labels = torch.tensor(
                sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1)
            )
            assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

            metrics = {
                'Recall@1': (torch.sum(labels[:, :1]) / len(labels)).item() * 100,
                'Recall@5': (torch.sum(labels[:, :5]) / len(labels)).item() * 100,
                'Recall@10': (torch.sum(labels[:, :10]) / len(labels)).item() * 100,
                'Recall@50': (torch.sum(labels[:, :50]) / len(labels)).item() * 100,
            }

            if sweep_enabled:
                prefix = f'pos{pos_w:g}_neg{neg_w:g}_'
                for metric_name, metric_value in metrics.items():
                    output_metrics[prefix + metric_name] = metric_value
            else:
                output_metrics.update(metrics)

    return output_metrics
    

@torch.no_grad()
def cirr(
    device: torch.device, 
    predicted_features: torch.Tensor,
    positive_features: torch.Tensor,
    negative_features: torch.Tensor, 
    reference_names: List, 
    targets: Union[np.ndarray,List], 
    target_names: List, 
    index_features: torch.Tensor, 
    index_names: List, 
    query_ids: Union[np.ndarray,List],
    preload_dict: Dict[str, Union[str, None]],
    split: str='val',    
    hdc_index_features: Optional[torch.Tensor] = None,
    **kwargs
) -> Dict[str, float]:
    """
    Compute the retrieval metrics on the CIRR validation set given the dataset, pseudo tokens and the reference names.
    Computes Recall@1, 5, 10 and 50. If given a test set, will generate submittable file.
    """   
    # Put on device.
    index_features = index_features.to(device)
    predicted_features = predicted_features.to(device)

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    if distances.ndim == 3:
        # If there are multiple features per instance, we average.
        distances = distances.mean(dim=1)
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    resize = len(sorted_index_names) if split == 'test' else len(target_names)
    reference_mask = torch.tensor(sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(resize, -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0], sorted_index_names.shape[1] - 1)
    
    # Compute the subset predictions and ground-truth labels
    targets = np.array(targets)
    group_mask = (sorted_index_names[..., None] == targets[:, None, :]).sum(-1).astype(bool)

    if split == 'test':
        sorted_group_names = sorted_index_names[group_mask].reshape(sorted_index_names.shape[0], -1)
        pairid_to_retrieved_images, pairid_to_group_retrieved_images = {}, {}
        for pair_id, prediction in zip(query_ids, sorted_index_names):
            pairid_to_retrieved_images[str(int(pair_id))] = prediction[:50].tolist()
        for pair_id, prediction in zip(query_ids, sorted_group_names):
            pairid_to_group_retrieved_images[str(int(pair_id))] = prediction[:3].tolist()            

        submission = {'version': 'rc2', 'metric': 'recall'}
        group_submission = {'version': 'rc2', 'metric': 'recall_subset'}

        submission.update(pairid_to_retrieved_images)
        group_submission.update(pairid_to_group_retrieved_images)

        submissions_folder_path = os.path.join(os.getcwd(), 'data', 'test_submissions', 'cirr')
        os.makedirs(submissions_folder_path, exist_ok=True)

        with open(os.path.join(submissions_folder_path, preload_dict['test']), 'w') as file:
            json.dump(submission, file, sort_keys=True)
        with open(os.path.join(submissions_folder_path, f"subset_{preload_dict['test']}"), 'w') as file:
            json.dump(group_submission, file, sort_keys=True)                        
        return None
            
    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(sorted_index_names == np.repeat(np.array(target_names), len(index_names) - 1).reshape(len(target_names), -1))    
    group_labels = labels[group_mask].reshape(labels.shape[0], -1)

    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    assert torch.equal(torch.sum(group_labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    output_metrics = {f'recall@{key}': (torch.sum(labels[:, :key]) / len(labels)).item() * 100 for key in [1, 5, 10, 50]}
    output_metrics.update({f'group_recall@{key}': (torch.sum(group_labels[:, :key]) / len(group_labels)).item() * 100 for key in [1, 2, 3]})

    return output_metrics


@torch.no_grad()
def circo(
    device: torch.device, 
    predicted_features: torch.Tensor, 
    positive_features: torch.Tensor,
    negative_features: torch.Tensor,
    targets: Union[np.ndarray,List], 
    target_names: List, 
    index_features: torch.Tensor, 
    index_names: List,
    query_ids: Union[np.ndarray,List],
    preload_dict: Dict[str, Union[str, None]],
    split: str='val',
    hdc_index_features: Optional[torch.Tensor] = None,
    **kwargs
) -> Dict[str, float]:
    """
    Compute the retrieval metrics on the CIRCO validation set given the pseudo tokens and the reference names.
    Computes mAP@5, 10, 25 and 50. If test-split, generates submittable file.
    """
    if hdc_index_features is None:
        # Load the model
        # Put on device.
        index_features = index_features.to(device)
        predicted_features = predicted_features.to(device)
    
    ### Compute Test Submission in case of test split.
    if split == 'test':
        print('Generating test submission file!')
        similarity = predicted_features @ index_features.T
        if similarity.ndim == 3:
            # If there are multiple features per instance, we average.
            similarity = similarity.mean(dim=1)                    
        sorted_indices = torch.topk(similarity, dim=-1, k=50).indices.cpu()
        sorted_index_names = np.array(index_names)[sorted_indices]
        # Return prediction dict to submit.
        queryid_to_retrieved_images = {
            query_id: query_sorted_names[:50].tolist() for (query_id, query_sorted_names) in zip(query_ids, sorted_index_names)            
        }
        
        submissions_folder_path = os.path.join(os.getcwd(), 'data', 'test_submissions', 'circo')
        os.makedirs(submissions_folder_path, exist_ok=True)
        with open(os.path.join(submissions_folder_path, preload_dict['test']), 'w') as file:
            json.dump(queryid_to_retrieved_images, file, sort_keys=True)        
        return None
    
    ### Directly compute metrics when using validation split.
    retrievals = [5, 10, 25, 50]
    recalls = {key: [] for key in retrievals}
    maps = {key: [] for key in retrievals}
    
    if hdc_index_features is not None:
        hd_encoder = kwargs.get('hd_encoder')
        if predicted_features.ndim == 3:
            b, k, d = predicted_features.shape
            predicted_hv = encode_index_features_hdc(predicted_features.view(-1, d), hd_encoder, output_device=device)
            all_similarities = get_hd_similarity_matrix(predicted_hv, hdc_index_features.to(device))
            all_similarities = all_similarities.view(b, k, -1).mean(dim=1).cpu()
        else:
            predicted_hv = encode_index_features_hdc(predicted_features, hd_encoder, output_device=device)
            all_similarities = get_hd_similarity_matrix(predicted_hv, hdc_index_features.to(device)).cpu()
        
    for idx, (predicted_feature, target_name, sub_targets) in enumerate(tqdm.tqdm(zip(predicted_features, target_names, targets), total=len(predicted_features), desc='Computing Metric.')):
        sub_targets = np.array(sub_targets)[np.array(sub_targets) != '']  # remove trailing empty strings added for collate_fn
        if hdc_index_features is not None:
            similarity = all_similarities[idx]
        else:
            similarity = predicted_feature @ index_features.T
            if similarity.ndim == 2:
                # If there are multiple features per instance, we average.
                similarity = similarity.mean(dim=0)
            similarity = similarity.cpu()
        sorted_indices = torch.topk(similarity, dim=-1, k=50).indices.cpu()
        sorted_index_names = np.array(index_names)[sorted_indices]
        map_labels = torch.tensor(np.isin(sorted_index_names, sub_targets), dtype=torch.uint8)
        precisions = torch.cumsum(map_labels, dim=0) * map_labels  # Consider only positions corresponding to GTs
        precisions = precisions / torch.arange(1, map_labels.shape[0] + 1)  # Compute precision for each position

        for key in retrievals:
            maps[key].append(float(torch.sum(precisions[:key]) / min(len(sub_targets), key)))

        assert target_name == sub_targets[0], f"Target name not in GTs {target_name} {sub_targets}"
        single_gt_labels = torch.tensor(sorted_index_names == target_name)
        
        for key in retrievals:
            recalls[key].append(float(torch.sum(single_gt_labels[:key])))

    output_metrics = {f'mAP@{key}': np.mean(item) * 100 for key, item in maps.items()}
    output_metrics.update({f'recall@{key}': np.mean(item) * 100 for key, item in recalls.items()})
    return output_metrics


@torch.no_grad()
def genecis(
    device: torch.device, 
    predicted_features: torch.Tensor, 
    index_features: torch.Tensor, 
    index_ranks: List,
    topk: List[int] = [1, 2, 3],
    **kwargs    
) -> Dict[str, float]:
    
    predicted_features = torch.nn.functional.normalize(predicted_features.float(), dim=-1).to(device)
    index_features = torch.nn.functional.normalize(index_features.float(), dim=-1).to(device)
    
    # Compute similarity
    if predicted_features.ndim == 3:
        similarities = predicted_features.bmm(index_features.permute(0,2,1)).mean(dim=1)
    else:
        similarities = (predicted_features[:, None, :] * index_features).sum(dim=-1)

    # # Sort the similarities in ascending order (closest example is the predicted sample)
    _, sort_idxs = similarities.sort(dim=-1, descending=True)                   # B x N

    # Compute recall at K
    if isinstance(index_ranks, list):
        index_ranks = torch.stack(index_ranks)
    index_ranks = index_ranks.to(device)
    
    output_metrics = {f'R@{k}': get_recall(sort_idxs[:, :k], index_ranks) * 100 for k in topk}

    return output_metrics


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    
    
def get_recall(indices, targets): #recall --> wether next item in session is within top K recommended items or not
    """
    Code adapted from: https://github.com/hungthanhpham94/GRU4REC-pytorch/blob/master/lib/metric.py
    Calculates the recall score for the given predictions and targets
    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B) or (BxN): torch.LongTensor. actual target indices.
    Returns:
        recall (float): the recall score
    """

    if len(targets.size()) == 1:
        # One hot label branch
        targets = targets.view(-1, 1).expand_as(indices)
        hits = (targets == indices).nonzero()
        if len(hits) == 0: return 0
        n_hits = (targets == indices).nonzero()[:, :-1].size(0)
        recall = float(n_hits) / targets.size(0)
        return recall
    else:        
        # Multi hot label branch
        recall = []
        for preds, gt in zip(indices, targets):            
            max_val = torch.max(torch.cat([preds, gt])).int().item()
            preds_binary = torch.zeros((max_val + 1,), device=preds.device, dtype=torch.float32).scatter_(0, preds, 1)
            gt_binary = torch.zeros((max_val + 1,), device=gt.device, dtype=torch.float32).scatter_(0, gt.long(), 1)
            success = (preds_binary * gt_binary).sum() > 0
            recall.append(int(success))        
        return torch.Tensor(recall).float().mean()
    






def apply_splice_modifications(
    i, reference_img_features, basis_hd, basis_mean, encoder, basis_names,
    scene_captions, negation_captions, scene_features, negation_features, reference_names,
    image_mean=None,
):
    # `image_mean` (DB-derived, targetpad_centered recipe) centers image queries;
    # `basis_mean` (concept text mean) keeps centering text-side. Fall back to
    # basis_mean if image_mean wasn't supplied (preserves pre-integration behaviour).
    image_center = image_mean if image_mean is not None else basis_mean

    ref_clip = reference_img_features[i:i+1]
    ref_weights, ref_indices, orig_ref_hd = splice_decompose(
        ref_clip, basis_hd, image_center, encoder, top_k=100, num_iterations=3, temp=0.001, batch_size=1
    )
    # print("Reference Image:", reference_names[i])
    # print("Top contributing basis vectors before modification:")
    # for j in range(min(10, ref_indices.shape[1])):
    #     basis_name = basis_names[ref_indices[0, j].item()]
    #     weight = ref_weights[0, j].item()
    #     print(f"  {j+1}. {basis_name} (weight: {weight:.4f})")

    # Get the raw original HD vector before normalization
    ref_clip_centered = ref_clip - image_center
    ref_clip_centered = torch.nn.functional.normalize(ref_clip_centered, p=2, dim=1)
    original_ref_raw_hd = torchhd.MAPTensor(ref_clip_centered @ encoder) # [1, HD_DIM]

    delta_hd = torch.zeros_like(original_ref_raw_hd)
    mod_flag = False

    c_rel_hvs = basis_hd[ref_indices[0]] # [100, HD_DIM]

    scene_feats = scene_features[i] if scene_captions[i] else None
    if scene_feats is not None:
        scene_feats = torch.nn.functional.normalize(scene_feats - basis_mean, p=2, dim=-1) # Center and normalize scene features the same way as the basis
        scene_hds = torchhd.MAPTensor(scene_feats @ encoder) # [K, HD_DIM]
        scene_hds = torchhd.normalize(scene_hds)
        sims_with_scene = torchhd.hamming_similarity(scene_hds, c_rel_hvs) # [K, 100]
        most_sim_idxs = sims_with_scene.argmax(dim=-1) # [K]
        for k in range(scene_hds.shape[0]):
            sim_idx = most_sim_idxs[k]
            replaced_basis_name = basis_names[ref_indices[0, sim_idx].item()]
            # print(f"Query {i} (Scene[{k}]): Replacing '{replaced_basis_name}' with scene HD via Delta")
            orig_weight = ref_weights[0, sim_idx].item()
            delta_hd -= orig_weight * c_rel_hvs[sim_idx].unsqueeze(0)
            delta_hd += orig_weight * scene_hds[k].unsqueeze(0)
        mod_flag = True

    neg_feats = negation_features[i] if negation_captions[i] else None
    if neg_feats is not None:
        neg_feats = torch.nn.functional.normalize(neg_feats - basis_mean, p=2, dim=-1) # Center and normalize negation features the same way as the basis
        neg_hds = torchhd.MAPTensor(neg_feats @ encoder) # [K, HD_DIM]
        neg_hds = torchhd.normalize(neg_hds)
        sims_with_neg = torchhd.hamming_similarity(neg_hds, c_rel_hvs) # [K, 100]
        most_sim_idxs = sims_with_neg.argmax(dim=-1) # [K]
        for k in range(neg_hds.shape[0]):
            sim_idx = most_sim_idxs[k]
            zeroed_basis_name = basis_names[ref_indices[0, sim_idx].item()]
            # print(f"Query {i} (Negation[{k}]): Zeroing out '{zeroed_basis_name}' via Delta subtraction")
            orig_weight = ref_weights[0, sim_idx].item()
            delta_hd -= orig_weight * c_rel_hvs[sim_idx].unsqueeze(0)
        mod_flag = True

    return mod_flag, delta_hd

@torch.no_grad()
def intent_experiment(
    device: torch.device, 
    predicted_features: torch.Tensor, 
    targets: Union[np.ndarray,List], 
    target_names: List, 
    reference_names: List,
    index_features: torch.Tensor, 
    index_names: List,
    query_ids: Union[np.ndarray,List],
    preload_dict: Dict[str, Union[str, None]],
    instructions: List[str],
    coarse_labels: List[set],
    semantic_aspects_list: List[Union[str, List[str]]],
    scene_captions: List[str],
    negation_captions: List[str],
    clip_model: clip.model.CLIP,
    split: str='val',
    hdc_index_features: Optional[torch.Tensor] = None,
    **kwargs
) -> Dict[str, float]:
    """
    Compute the retrieval metrics with intent-based weights:
    similarity = w1*sim(predicted, index) + w2*sim(instruction, index) + w3*sim(reference, index)
    w1, w2, w3 depend on the intent.
    Need `clip_model` to encode `instructions` (which are relative captions).
    """
    index_features = index_features.to(device)
    predicted_features = predicted_features.to(device)

    # 1. Get instruction features
    if hasattr(clip_model, 'tokenizer'):
        tokenized_instructions = clip_model.tokenizer(instructions, context_length=77).to(device)
    else:
        tokenized_instructions = clip.tokenize(instructions, context_length=77, truncate=True).to(device)
    
    instruction_features = clip_model.encode_text(tokenized_instructions)
    instruction_features = torch.nn.functional.normalize(instruction_features.float(), dim=-1).to(device)
    
    # 1.5 Encode Scene and Negation concepts.
    # Each caption may bundle multiple sub-concepts separated by commas; encode each
    # part independently so apply_splice_modifications can replace/remove every match.
    # Returns a list aligned to queries: each entry is a [K_i, D] tensor, or None when empty.
    def get_text_embeds_split(captions):
        result = []
        for cap in captions:
            if not cap or not cap.strip():
                result.append(None)
                continue
            # parts = cap.strip()
            parts = [p.strip() for p in cap.split(',') if p.strip()]
            if not parts:
                result.append(None)
                continue
            if hasattr(clip_model, 'tokenizer'):
                toks = clip_model.tokenizer(parts, context_length=77).to(device)
            else:
                toks = clip.tokenize(parts, context_length=77, truncate=True).to(device)
            embeds = clip_model.encode_text(toks)
            embeds = torch.nn.functional.normalize(embeds.float(), dim=-1).to(device)
            result.append(embeds)
        return result

    scene_features = get_text_embeds_split(scene_captions)
    negation_features = get_text_embeds_split(negation_captions)
    
    args_obj = kwargs.get('args', None)
    use_splice = getattr(args_obj, 'use_splice', False)

    # SPLiCE Integration:
    if use_splice:
        vocab_path = getattr(args_obj, 'splice_vocab_path', '/workspace/joel/HDCIR/SpLiCE/data/vocab/laion.txt')
        basis_hd, basis_embeddings, basis_mean, encoder, basis_names = build_splice_base(clip_model, device, hd_dim=10000, vocab_path=vocab_path)

        # 1.6 Project index_features to HD space (no need to decompose the whole DB)
        index_hd = torchhd.MAPTensor(index_features @ encoder)
        index_hd = torchhd.normalize(index_hd)

        # Load the DB-derived image_mean (targetpad variant) that hdc_splice produced.
        # main.py is responsible for ensuring the pickle exists when --preprocess-type targetpad.
        image_mean = None
        db_emb_path = getattr(args_obj, 'db_emb_path', None)
        if db_emb_path is None:
            from hdc_splice import resolve_model_tag
            tag = resolve_model_tag(getattr(args_obj, 'clip', 'ViT-B-32'))
            db_emb_path = os.path.join('precomputed', f'db_embeddings_{tag}.pkl')
        try:
            image_mean = load_db_image_means(
                db_emb_path,
                expected_clip_model=getattr(args_obj, 'clip', None),
                variant='targetpad',
            ).to(device)
            print(f"[SPLiCE] Loaded image_mean from {db_emb_path} (variant=targetpad).")
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"[SPLiCE] WARNING: {e}\n[SPLiCE] Falling back to basis_mean (text mean) for image centering.")
    
    # 2. Get reference image features
    index_name_to_idx = {name: i for i, name in enumerate(index_names)}
    ref_indices_list = [index_name_to_idx[name] for name in reference_names]
    reference_img_features = index_features[ref_indices_list].to(device)
    
    # splice weights
    intent_weights = {
        'negation': (1.0, 0.0, 0.0, 1.0),
        'addition': (1.0, 0.4, 0.0, 0.0), 
        'direct_addressing': (1.0, 0.3, 0.0, 0.5), 
        'compare_change': (1.0, 0.2, 0.0, 1.0), 
        'spatial_relations_background': (1.0, 0.1, 0.0, 0.0), 
        'viewpoint': (1.0, 0.25, 0.0, 1.0), 
        'comparative_statement': (1.0, 0.25, 0.0, 0.0), 
        'cardinality': (1.0, 0.1, 0.0, 0.0), 
        'default': (1.0, 0.0, 0.0, 0.0)
    }

    # w3 weights
    # intent_weights = {
    #     'negation': (1.0, 0.0, 0.5, 0.0),
    #     'addition': (1.0, 0.3, 0.0, 0.0), 
    #     'direct_addressing': (1.0, 0.3, 0.05, 0.0), 
    #     'compare_change': (1.0, 0.2, 0.1, 0.0), 
    #     'spatial_relations_background': (1.0, 0.1, 0.0, 0.0), 
    #     'viewpoint': (1.0, 0.25, 0.5, 0.0), 
    #     'comparative_statement': (1.0, 0.25, 0.1, 0.0), 
    #     'cardinality': (1.0, 0.1, 0.0, 0.0), 
    #     'default': (1.0, 0.0, 0.0, 0.0)
    # }

    # Ratio-preserving scaling
    for key in intent_weights:
        w1, w2, w3, w4 = intent_weights[key]
        new_w2 = w2 * (w1+w3+w4)
        intent_weights[key] = (w1, new_w2, w3, w4)

    active_intents = getattr(kwargs.get('args', {}), 'active_intents', [])
    cache_path = getattr(args_obj, 'cache_path', None) if args_obj is not None else None

    similarities = []
    # Per-query similarity components (for weight tuning / caching). Stored on CPU
    # so a long sweep doesn't pin GPU memory; hd_sim is zeros when mod_flag is False
    # so the cached tensor stacks cleanly and w4*0 contributes nothing downstream.
    cache_pred_sims = []
    cache_inst_sims = []
    cache_ref_sims = []
    cache_hd_sims = []
    cache_mod_flags = []
    cache_query_labels = []

    for i in range(len(predicted_features)):
        labels = []
        if i < len(semantic_aspects_list):
            aspects = semantic_aspects_list[i]
            if isinstance(aspects, list):
                labels = aspects
            elif isinstance(aspects, str):
                labels = [aspects]
                
        w1, w2, w3, w4 = 0.0, 0.0, 0.0, 0.0
        count = 0
        for lbl in labels:
            if lbl in intent_weights and lbl in active_intents:
                w1 += intent_weights[lbl][0]
                w2 += intent_weights[lbl][1]
                w3 += intent_weights[lbl][2]
                w4 += intent_weights[lbl][3]
                count += 1
        if count == 0:
            w1, w2, w3, w4 = intent_weights['default']
        else:
            w1 /= count
            w2 /= count
            w3 /= count
            w4 /= count
        
            

        # print(f"Query {i}: Intents {labels}, Weights: w1={w1:.2f}, w2={w2:.2f}, w3={w3:.2f}, w4={w4:.2f}")

        mod_flag = False
        if use_splice and (scene_captions[i] or negation_captions[i]):
            mod_flag, delta_hd = apply_splice_modifications(
                i, reference_img_features, basis_hd, basis_mean, encoder, basis_names,
                scene_captions, negation_captions, scene_features, negation_features, reference_names,
                image_mean=image_mean,
            )

        pred_sim = (predicted_features[i:i+1] @ index_features.T).squeeze(0)
        inst_sim = (instruction_features[i:i+1] @ index_features.T).squeeze(0)
        ref_sim = (reference_img_features[i:i+1] @ index_features.T).squeeze(0)

        sim = w1 * pred_sim + w2 * inst_sim + w3 * ref_sim

        if mod_flag:
            orig_ref_hv = index_hd[ref_indices_list[i]].unsqueeze(0)
            mod_ref_raw_hd = torchhd.bundle(orig_ref_hv, delta_hd)
            # mod_ref_raw_hd = orig_ref_hv + delta_hd
            mod_ref_hd = torchhd.normalize(mod_ref_raw_hd)

            hd_sim = torchhd.cosine_similarity(mod_ref_hd, index_hd).squeeze(0)

            # Add the hd similarity logic into existing sum. (We scale it appropriately via w4)
            sim = sim + (w4 * hd_sim)
        else:
            hd_sim = torch.zeros_like(pred_sim)

        similarities.append(sim)

        if cache_path:
            # float16 halves disk usage; precision is fine for top-K ranking sweeps.
            cache_pred_sims.append(pred_sim.detach().to('cpu', dtype=torch.float16))
            cache_inst_sims.append(inst_sim.detach().to('cpu', dtype=torch.float16))
            cache_ref_sims.append(ref_sim.detach().to('cpu', dtype=torch.float16))
            cache_hd_sims.append(hd_sim.detach().to('cpu', dtype=torch.float16))
            cache_mod_flags.append(bool(mod_flag))
            cache_query_labels.append(list(labels))

    if cache_path:
        cache = {
            'pred_sims': torch.stack(cache_pred_sims),
            'inst_sims': torch.stack(cache_inst_sims),
            'ref_sims': torch.stack(cache_ref_sims),
            'hd_sims': torch.stack(cache_hd_sims),
            'mod_flags': torch.tensor(cache_mod_flags, dtype=torch.bool),
            'query_labels': cache_query_labels,
            'target_names': list(target_names),
            'targets': [list(t) for t in targets],
            'index_names': list(index_names),
            'active_intents_at_capture': list(active_intents),
            'intent_weights_at_capture': dict(intent_weights),
        }
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        torch.save(cache, cache_path)
        print(f"[CACHE] Saved per-query similarity components for {len(cache_pred_sims)} queries to {cache_path}")

    all_similarities = torch.stack(similarities, dim=0).cpu()
    
    if split == 'test':
        print('Generating test submission file!')
        sorted_indices = torch.topk(all_similarities, dim=-1, k=50).indices.cpu()
        sorted_index_names = np.array(index_names)[sorted_indices]
        queryid_to_retrieved_images = {
            query_id: query_sorted_names[:50].tolist() for (query_id, query_sorted_names) in zip(query_ids, sorted_index_names)            
        }
        
        submissions_folder_path = os.path.join(os.getcwd(), 'data', 'test_submissions', 'circo')
        os.makedirs(submissions_folder_path, exist_ok=True)
        with open(os.path.join(submissions_folder_path, preload_dict['test']), 'w') as file:
            json.dump(queryid_to_retrieved_images, file, sort_keys=True)
        return None
    
    # Compute metrics like circo
    retrievals = [5, 10, 25, 50]
    recalls = {key: [] for key in retrievals}
    maps = {key: [] for key in retrievals}
    import tqdm
    for idx, (target_name, sub_targets) in enumerate(tqdm.tqdm(zip(target_names, targets), total=len(target_names), desc='Computing Metric.')):
        sub_targets = np.array(sub_targets)[np.array(sub_targets) != '']
        similarity = all_similarities[idx]
        sorted_indices = torch.topk(similarity, dim=-1, k=50).indices.cpu()
        sorted_index_names = np.array(index_names)[sorted_indices]
        map_labels = torch.tensor(np.isin(sorted_index_names, sub_targets), dtype=torch.uint8)
        precisions = torch.cumsum(map_labels, dim=0) * map_labels
        precisions = precisions / torch.arange(1, map_labels.shape[0] + 1)

        for key in retrievals:
            maps[key].append(float(torch.sum(precisions[:key]) / min(len(sub_targets), key)))

        assert target_name == sub_targets[0], f"Target name not in GTs {target_name} {sub_targets}"
        single_gt_labels = torch.tensor(sorted_index_names == target_name)
        
        for key in retrievals:
            recalls[key].append(float(torch.sum(single_gt_labels[:key])))

    output_metrics = {f'mAP@{key}': np.mean(item) * 100 for key, item in maps.items()}
    output_metrics.update({f'recall@{key}': np.mean(item) * 100 for key, item in recalls.items()})
    return output_metrics






# ########## TO UPDATE, CURRENTLY BROKEN ############
# @torch.no_grad()
# def fiq_generate_val_predictions(clip_model: clip.model.CLIP, blip_model:callable,query_dataset: torch.utils.data.Dataset, dress_type: str, preload_dict: Dict[str, Union[str,None]]) -> Tuple[torch.Tensor, List[str]]:
#     """
#     Generates features predictions for the validation set of Fashion IQ.
#     """    

#     # Create data loader
#     relative_val_loader = torch.utils.data.DataLoader(dataset=query_dataset, batch_size=32, num_workers=10,
#                                      pin_memory=False, collate_fn=data_utils.collate_fn, shuffle=False)
    
#     predicted_features_list = []
#     target_names_list = []
    
#     # Compute features
#     for batch in tqdm.tqdm(relative_val_loader):
#         reference_names = batch['reference_name']
#         target_names = batch['target_name']
#         relative_captions = batch['relative_captions']

#         flattened_captions = np.array(relative_captions).T.flatten().tolist()
#         input_captions = [
#             f"{flattened_captions[i].strip('.?, ')} and {flattened_captions[i + 1].strip('.?, ')}" for
#             i in range(0, len(flattened_captions), 2)]
#         orig_input_captions = input_captions
#         input_captions = [f"a photo of $ that {in_cap}" for in_cap in input_captions]
        

#         blip_image = batch['blip_ref_img'].to(device)
#         captions = []
#         for i in range(blip_image.size(0)):
#             img = blip_image[i].unsqueeze(0)
#             caption = blip_model.generate({'image': img, "prompt": prompts.fiq_blip2_prompt(dress_type)})
#             captions.append(caption[0])

#         modified_captions = []
#         for i in range(len(captions)):
#             instruction = orig_input_captions[i]
#             img_caption = captions[i]
#             final_prompt = prompts.fiq_fashion_prompt(dress_type) + '\n' + "Image Content: " + img_caption
#             final_prompt = final_prompt + '\n' + 'Instruction: '+ instruction
#             resp = openai_api.openai_completion(final_prompt)

#             resp = resp.split('\n')
#             print(resp)
#             ## extract edited description
#             description = ""
#             for line in resp:
                    
#                 if line.startswith('Edited Description:'):
#                     description = line.split(':')[1].strip()
#                     modified_captions.append(description)
#                     break
#             if description == "":
#                 modified_captions.append(orig_input_captions[i]) 
            
#         predicted_features = torch.nn.functional.normalize(
#             clip_model.encode_text(clip.tokenize(modified_captions,truncate=True).to(device)))
#         #predicted_features = torch.nn.functional.normalize((torch.nn.functional.normalize(text_features) + torch.nn.functional.normalize(text_features_reversed)) / 2)
#         # predicted_features = torch.nn.functional.normalize((text_features + text_features_reversed) / 2)

#         predicted_features_list.append(predicted_features)
#         target_names_list.extend(target_names)

#     predicted_features = torch.vstack(predicted_features_list)
#     return predicted_features, target_names_list

# @torch.no_grad()
# def fiq_compute_val_metrics(query_dataset: torch.utils.data.Dataset, clip_model: clip.model.CLIP, blip_model:callable, index_features: torch.Tensor,
#                             index_names: List[str], dress_type:str, preload_dict: Dict[str, Union[str,None]]) \
#         -> Dict[str, float]:
#     """
#     Compute the retrieval metrics on the FashionIQ validation set given the dataset, pseudo tokens and the reference names
#     """

#     # Generate the predicted features
#     predicted_features, target_names = fiq_generate_val_predictions(
#         clip_model, blip_model,query_dataset, dress_type, preload_dict=preload_dict)

#     # Move the features to the device
#     index_features = index_features.to(device)
#     predicted_features = predicted_features.to(device)

#     # Normalize the features
#     index_features = torch.nn.functional.normalize(index_features.float())

#     # Compute the distances
#     distances = 1 - predicted_features @ index_features.T
#     sorted_indices = torch.argsort(distances, dim=-1).cpu()
#     sorted_index_names = np.array(index_names)[sorted_indices]

#     # Check if the target names are in the top 10 and top 50
#     labels = torch.tensor(
#         sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
#     assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

#     # Compute the metrics
#     recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
#     recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
#     recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
#     recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100


#     return {'fiq_recall_at1': recall_at1,
#             'fiq_recall_at5': recall_at5,
#             'fiq_recall_at10': recall_at10,
#             'fiq_recall_at50': recall_at50}

# @torch.no_grad()
# def fiq(args: argparse.Namespace, dataset_path: str, dress_type: str, clip_model_name: str, blip_model:callable,
#                        preprocess: callable,blip_transform:callable, preload_dict: Dict[str, Union[str,None]]) -> Dict[str, float]:
#     """
#     Compute the retrieval metrics on the FashionIQ validation set given the pseudo tokens and the reference names
#     """
#     # Load the model
#     clip_model, _ = clip.load(clip_model_name, device=utils.device, jit=False)
#     clip_model = clip_model.float().eval().requires_grad_(False)

#     # Extract the index features

#     index_features, index_names = utils.extract_image_features(args, classic_val_dataset, clip_model, preload=preload_dict['img_features'])
#     #index_features, index_names = extract_image_captions(classic_val_dataset, blip_model)

#     # Define the relative dataset
#     query_dataset = datasets.FashionIQDataset(dataset_path, 'val', [dress_type], 'relative', preprocess,blip_transform=blip_transform)

#     return fiq_compute_val_metrics(query_dataset, clip_model, blip_model,index_features, index_names,
#                                    dress_type,preload_dict=preload_dict)
# ################################################3


