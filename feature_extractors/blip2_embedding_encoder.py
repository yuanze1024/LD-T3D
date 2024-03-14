"""
See https://github.com/salesforce/LAVIS/tree/main/projects/blip2 for source code
"""
import torch
from torch.hub import set_dir
from lavis.models import load_model_and_preprocess
import sys
sys.path.append('')
from feature_extractors import FeatureExtractor
import os

class Blip2EmbeddingEncoder(FeatureExtractor):
    def __init__(self, cache_dir, **kwargs):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir # set tokenizer('bert-base-uncased') cache dir
        set_dir(cache_dir) # set BLIP2 model cache dir
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=self.device)

    @torch.no_grad()
    def encode_image(self, img_tensor_list):
        image_input_list = img_tensor_list.squeeze(1).to(self.device)
        sample = {}
        sample["image"] = image_input_list
        embeddings = self.model.extract_features(sample, mode="image")['image_embeds_proj'] # (bs, 32, 256)
        # take the first query of the 32 queries
        embeddings = embeddings[:, 0, :] # (bs, 256)
        assert embeddings.shape[0] == img_tensor_list.shape[0], "The number of embeddings should be the same as the number of source"
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings

    @torch.no_grad()
    def encode_text(self, caption_list):
        text_input_list = [self.txt_processors["eval"](caption) for caption in caption_list]
        sample = {}
        sample["text_input"] = text_input_list
        embeddings = self.model.extract_features(sample, mode="text")['text_embeds_proj'][:, 0, :] # (bs, 256)
        return embeddings / embeddings.norm(dim=-1, keepdim=True)

    def encode_query(self, caption_list):
        return self.encode_text(caption_list)
    
    def encode_3D(self, pc_tensor):
        raise NotImplementedError
    
    def get_img_transform(self):
        def inner(img):
            img = img.convert("RGB") # BLIP2 only accepts 3-channel RGB images instead of RGBA
            return self.vis_processors["eval"](img)
        return inner