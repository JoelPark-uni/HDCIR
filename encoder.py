import torch
import torch.nn.functional as F
import torchhd
import numpy as np
from PIL import Image
from transformers import AutoProcessor

device = "cuda:1"
model_id = "xtuner/llava-phi-3-mini-hf"
processor = AutoProcessor.from_pretrained(model_id,)
input_embeddings = torch.load("./input_embeddings.pt").to(device)

hd_cls = torchhd.MAPTensor

def seed_everything(seed):
    import random
    import numpy as np
    import torch
    import os

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

class HDlm:
    def __init__(self, feature_num=20, feature_dim=3072, Hfeature_dim=9500, HD_DIM=10000, HD_cls=torchhd.MAPTensor, device='cuda:0'):
        self.HD_DIM = HD_DIM
        self.HD_cls = HD_cls
        self.feature_num = feature_num
        self.device = device
        if feature_dim is not None:
            self.encoder = torchhd.MAPTensor.random(feature_dim, HD_DIM, device=device)
        else:
            self.encoder = None
        
        self.Hfeature_dim = Hfeature_dim
        self.rand_indices = [torch.randperm(HD_DIM) for _ in range(feature_num)]

    def encode(self, x):
        # N, D = x.shape
        if self.encoder is None:
            return x
        
        x = x @ self.encoder.type(x.dtype)

        return x.normalize()

    def bundle(self, x):
        # x: (N, HD_DIM)
        x = self.encode(x)
        out = self.HD_cls.multibundle(x).normalize()
        return out

    def correlative(self, x, importance=None):
        # x: (N, HD_DIM)
        N, HD_DIM = x.shape
        x = self.encode(x)
        importance = importance if importance is not None else [1/self.feature_num for i in range(min(N, self.feature_num))]
        importance = [int(self.HD_DIM * importance[i]) for i in range(min(N, self.feature_num))]

        out = torch.ones_like(x)

        # feature마다 선택된 차원만 원래 값으로 교체
        for i in range(min(N, self.feature_num)): # 10 이하로
            idx = self.rand_indices[i][:importance[i]]
            # idx = self.Hfeature_index[i]

            out[i, idx] = x[i, idx]

        out = self.HD_cls(out)
        out = torchhd.multibind(out).normalize()
        return out

    
    def masked_similarity(self, x, y):
        # x, y: (B, HD_DIM)
        assert x.shape == y.shape, f"Input shapes {x.shape} and {y.shape} do not match"
        B, HD_DIM = x.shape
        assert HD_DIM == self.HD_DIM, f"Input HD dimension {HD_DIM} does not match model HD dimension {self.HD_DIM}"

        similarity = torchhd.hamming_similarity(x[:, self.shared_index], y[:, self.shared_index]) / len(self.shared_index)
        return similarity


class VocabHV:
    def __init__(self, special_tokens, split_tokens, input_embeddings, dim=10000):
        self.vocab_size = input_embeddings.shape[0]
        self.special_tokens = special_tokens
        self.split_tokens = split_tokens
        self.input_embeddings = input_embeddings
        self.dim = dim
        self.hv_map = torchhd.MAPTensor.random(input_embeddings.shape[1], dim, device=input_embeddings.device)
        self.hv_random = torchhd.MAPTensor.random(self.vocab_size, dim, device=input_embeddings.device)

    def encode(self, token_ids):
        mask = torch.tensor([token_id not in self.special_tokens for token_id in token_ids])
        token_ids = token_ids[mask]
        # find split positions
        split_positions = [i for i, token_id in enumerate(token_ids) if token_id in self.split_tokens]
        # split token_ids into segments
        segments = []
        prev_pos = 0
        for pos in split_positions:
            segments.append(token_ids[prev_pos:pos])
            prev_pos = pos + 1
        segments.append(token_ids[prev_pos:])
        # encode each segment
        segment_hvs = []
        for segment in segments:
            if len(segment) == 0:
                continue
            embs = self.input_embeddings[segment]
            hvs = F.normalize((embs @ self.hv_map))
            segment_hv = torchhd.MAPTensor.multibundle(hvs).normalize()
            segment_hvs.append(segment_hv)
        segment_hvs = torch.stack(segment_hvs)
        hv = torchhd.MAPTensor.multibundle(segment_hvs).normalize()
        return hv

    def encode_random(self, token_ids):
        mask = torch.tensor([token_id not in self.special_tokens for token_id in token_ids])
        token_ids = token_ids[mask]
        hvs = self.hv_random[token_ids]
        hv = torchhd.MAPTensor.multibundle(hvs)
        hv = hv.normalize()
        return hv




def encode_data(data_dict, importance=None, match_db=False):
    data_hvs = []
    for q, v in data_dict.items():
        if type(v) != str:
            v = str(v)
        token_ids = processor.tokenizer(v, return_tensors="pt", add_special_tokens=False)['input_ids'][0]
        if match_db:
            # Find the closest token in the database
            token_hv = vocab_hv.encode(token_ids).to(device)
            sims = torchhd.hamming_similarity(item_memory, token_hv)
            best_idx = torch.argmax(sims).item()
            data_hvs.append(item_memory[best_idx])
        else:
            data_hvs.append(vocab_hv.encode(token_ids))

    if type(item_memory) == list:
        item_memory.extend(data_hvs)
    data_hvs = torch.stack(data_hvs).to(device)
    data_hv = hdlm.correlative(data_hvs, importance=importance)
    return data_hv

def encode_all(data_dict, importance=None):
    result_dict = {}
    for i, (k, v) in enumerate(data_dict.items()):
        # if k == 'shape_rep':
        #     continue
        if i % 100 == 0:
            print(f"Encoding {i+1}/{len(data_dict)}: {k}")
        data_hv = encode_data(v, importance=importance)
        result_dict[k] = data_hv
    return result_dict




item_memory = []
importance = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
# importance = [0.2 - i*0.02 for i in range(10)] # Gradually decrease importance
result_dict = encode_all(data_dict, importance=importance)
item_memory = torch.stack(item_memory).to(device)

vocab_size = len(processor.tokenizer)
special_tokens = list(processor.tokenizer.get_added_vocab().values())
split_tokens = [322, 1919, 29871] # "and", ",", " "
vocab_hv = VocabHV(special_tokens, split_tokens, input_embeddings, dim=10000)
hdlm = HDlm(feature_num=7, feature_dim=None, Hfeature_dim=1500, HD_DIM=10000, device=device)