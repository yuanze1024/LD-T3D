"""
See https://github.com/SeanLee97/AnglE for source code
"""
from angle_emb import AnglE
import torch
from torch.nn import functional as F

import sys
sys.path.append('')
from feature_extractors import FeatureExtractor

class UaeEmbeddingEncoder(FeatureExtractor):
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').to(self.device)

    @torch.no_grad()
    def encode_text(self, caption_list):
        vecs = self.model.encode(caption_list, to_numpy=False)
        vecs = F.normalize(vecs, p=2, dim=1)
        return vecs
    
    def encode_query(self, caption_list):
        return self.encode_text(caption_list)

    def encode_3D(self, pc_tensor):
        raise NotImplementedError
    
    def encode_image(self, img_list):
        raise NotImplementedError