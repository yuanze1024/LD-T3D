"""
See https://github.com/microsoft/unilm/tree/master/e5 for source code
"""
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

import sys
sys.path.append('')
from feature_extractors import FeatureExtractor

class E5EmbeddingEncoder(FeatureExtractor):

    def __init__(self, **kwargs):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        repo_name = "intfloat/e5-large-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(repo_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(repo_name, trust_remote_code=True).to(self.device)

    @torch.no_grad()
    def encode_text(self, caption_list):
        caption_list = [f"query:{caption}" for caption in caption_list if not caption.startswith("query:")] # to align the training process of e5
        batch_dict = self.tokenizer(caption_list, max_length=512, padding=True, truncation=True, return_tensors='pt')
        for k in batch_dict.keys():
            batch_dict[k] = batch_dict[k].to(self.device)
        outputs = self.model(**batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def encode_query(self, caption_list):
        return self.encode_text(caption_list)

    def encode_3D(self, pc_tensor):
        raise NotImplementedError
    
    def encode_image(self, img_list):
        raise NotImplementedError


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
