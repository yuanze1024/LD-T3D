"""
See https://github.com/openai/CLIP for source code
"""
import torch
import clip

import sys
sys.path.append('')
from feature_extractors import FeatureExtractor

class ClipEmbeddingEncoder(FeatureExtractor):
    def __init__(self, **kwargs):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-L/14@336px", device=self.device)

    @torch.no_grad()
    def encode_image(self, img_list):
        image_input = torch.stack([self.preprocess(image).unsqueeze(0).to(self.device) for image in img_list]).squeeze(1)
        embeddings = self.model.encode_image(image_input)
        return embeddings / embeddings.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_text(self, input_text):
        text = clip.tokenize(input_text).to(self.device)
        text_features = self.model.encode_text(text)
        return text_features / text_features.norm(dim=-1, keepdim=True)

    def encode_query(self, queries):
        return self.encode_text(queries)

    def encode_3D(self, pc_tensor):
        raise NotImplementedError