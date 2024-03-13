
import os
import importlib
import torch
from tqdm import tqdm
from collections.abc import Sequence
import argparse
import yaml
from torch.utils.data import DataLoader

from utils.dataset import get_dataset, get_rel_dataset
from utils.metrics import calculate_metrics
from utils.logger import get_logger

logger = get_logger()

def predict(xb, xq, source_id_list, federated_dataset) -> Sequence[Sequence[str]]:
    """Predict the most related 3D models for each sub-dataset using Cosine Similarity.

    Args:
        xb: The embeddings of all 3D models, organized in the order of source_id_list.
        xq: The embeddings of all sub-datasets, where each row represents an embedding of the textual query.
        source_id_list: A list of ALL the source_id involved in the federated_dataset.
        federated_dataset: A dict, whose key is the query_id, and value is the sub-dataset, which is a list of source_id.
    
    Returns:
        A dict of predictions, whose key is the query_id, and value is the list of source_id, sorted in reverse order by relevance DESC.
    """
    result = {}
    source_to_id = {source_id: i for i, source_id in enumerate(source_id_list)}
    xb = xb.to(xq.device)
    sim = xq @ xb.T # (nq, nb)
    for i, query_id in tqdm(enumerate(federated_dataset.keys()), desc="searching..."):
        related_id = [source_to_id[target_id] for target_id in federated_dataset[query_id]] # find all the index of 3D models in a certain sub-dataset
        related_id = torch.tensor(related_id).to(sim.device)
        related_ord = sim[i, related_id].argsort(descending=True)
        result[query_id] = [source_id_list[related_id[j]] for j in related_ord.tolist()]
    return result

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the performance of different feature extractors.")
    parser.add_argument("--option", type=str, default="Uni3D", help="The feature extractor to be evaluated.")
    parser.add_argument("--cross_modal", type=str, default="text_image_3D", help="The modalities that are used, seperated by '_'. \
                        The default value is '', representing features extracted using the 'encode' method in the feature extractor. \
                        If not '', use the modalities included. E.g., 'text_image' or 'text_image_3D'.")
    parser.add_argument("--op", type=str, default="add", choices=["concat", "add"], help="The operation to be used to fuse different base embeddings.")
    parser.add_argument("--angles", nargs="+", default=["all"], choices=["all", "diag_below", "diag_above", "right", "left", "back", "front", "above", "below"], help="The angles to be used to extract the embeddings if modalities including 'Image'.")
    parser.add_argument("--batch_size", type=int, default=100, help="The batch size to be used to extract the embeddings.")
    args = parser.parse_args()
    logger.info(args)
    return args

def get_embedding(option, modality, source_id_list, encode_fn, cache_dir, angle=None, batch_size=128, img_transform=None):
    save_path = f'data/objaverse_{option}_{modality + (("_" + str(angle)) if angle is not None else "")}_embeddings.pt'
    if os.path.exists(save_path):
        return torch.load(save_path)
    else:
        embeddings = []
        dataset = get_dataset(modality, source_id_list, cache_dir=cache_dir, angle=angle, img_transform=img_transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=batch_size//4)
        for batch in tqdm(dataloader, desc=f"Extracting {modality + (('_' + str(angle)) if angle is not None else '')} embeddings..."):
            embeddings.append(encode_fn(batch))
        embedding = torch.cat(embeddings, dim=0).cpu()
        torch.save(embedding, save_path)
        return embedding

def get_yaml():
    file_yaml = 'config/config.yaml'
    with open(file=file_yaml, mode='r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)
    logger.info(f"Loaded config from {file_yaml}")
    logger.info(yaml_data)
    return yaml_data

def parse_rel_dataset(rel_dataset):
    federated_dataset_path, source_id_list_path, gt_dict_path, source_to_caption_path = "data/federated_dataset.pt", "data/source_id_list.pt", "data/gt_dict.pt", "data/source_to_caption.pt"
    if os.path.exists(federated_dataset_path) and os.path.exists(source_id_list_path) and os.path.exists(gt_dict_path) and os.path.exists(source_to_caption_path):
        return torch.load(federated_dataset_path), torch.load(source_id_list_path), torch.load(gt_dict_path), torch.load(source_to_caption_path)
    else:
        federated_dataset, source_id_list, gt_dict, source_to_caption = {}, set(), {}, {}
        for query_id in range(len(rel_dataset['query_id'])):
            federated_dataset[query_id] = rel_dataset['target_ids'][query_id]
            gt_dict[query_id] = rel_dataset['GT_ids'][query_id]
            source_to_caption[query_id] = rel_dataset['caption'][query_id]
        
        for value in federated_dataset.values():
            source_id_list.update(value)
        source_id_list = list(source_id_list)
        source_id_list.sort()
        torch.save(federated_dataset, federated_dataset_path)
        torch.save(source_id_list, source_id_list_path)
        torch.save(gt_dict, gt_dict_path)
        torch.save(source_to_caption, source_to_caption_path)
        return federated_dataset, source_id_list, gt_dict, source_to_caption

def evaluate(args, config):
    option = args.option.lower()
    angles = args.angles
    if "all" in angles:
        angles = ["diag_below", "diag_above", "right", "left", "back", "front", "above", "below"]
    op = args.op
    batch_size = args.batch_size
    cross_modal = args.cross_modal

    cache_dir = config['general']['cache_dir']

    rel_dataset = get_rel_dataset(cache_dir)
    federated_dataset, source_id_list, gt_dict, source_to_caption = parse_rel_dataset(rel_dataset)

    logger.info(f"Initializing {option}...")
    module = importlib.import_module(f"feature_extractors.{option}_embedding_encoder")
    encoder = getattr(module, f"{option.capitalize()}EmbeddingEncoder")(cache_dir, method_config=config.pop(option, None))
    
    # encode base embeddings
    embeddings = []
    if "text" in cross_modal:
        logger.info(f"Extracting text embeddings using {option}...")
        embeddings.append(get_embedding(option, "text", source_id_list, encoder.encode_text, cache_dir=cache_dir, batch_size=1024))
    if "image" in cross_modal:
        img_transform = encoder.get_img_transform()
        for angle in angles:
            logger.info(f"Extracting image embeddings using {option} at angle {angle}...")
            embeddings.append(get_embedding(option, "image", source_id_list, encoder.encode_image, cache_dir=cache_dir, angle=angle, batch_size=batch_size, img_transform=img_transform))
    if "3D" in cross_modal:
        logger.info(f"Extracting 3D embeddings using {option}...")
        embeddings.append(get_embedding(option, "3D", source_id_list, encoder.encode_3D, cache_dir=cache_dir, batch_size=batch_size))
        
    ## fuse base embeddings
    if len(embeddings) > 1:
        if op == "concat":
            embeddings = torch.cat(embeddings, dim=-1)
        elif op == "add":
            embeddings = sum(embeddings)
        else:
            raise ValueError(f"Unsupported operation: {op}")
        embeddings /= embeddings.norm(dim=-1, keepdim=True)
    else:
        embeddings = embeddings[0]

    # encode query embeddings
    logger.info(f"Encoding query embeddings using {option}...")
    xq = encoder.encode_query([source_to_caption[k] for k in federated_dataset.keys()])
    if op == "concat":
        xq = xq.repeat(1, embeddings.shape[-1] // xq.shape[-1]) # repeat to be aligned with the xb
        xq /= xq.norm(dim=-1, keepdim=True)
    
    pred_dict = predict(embeddings, xq, source_id_list, federated_dataset)

    # # test val split
    # import random
    # random.seed(2)
    # query_id_list = random.sample(list(gt_dict.keys()), 200)
    # new_pred_dict = {}
    # for query_id in query_id_list:
    #     new_pred_dict[query_id] = pred_dict[query_id]
    # pred_dict = new_pred_dict

    logger.info(f"option: {option}, all:") # TODO: 把config和结果持久化
    log_message = "\tmAP: {},\tmnDCG: {},\tmFT: {},\tmST: {}".format(
        calculate_metrics(pred_dict, gt_dict, metric="mAP"),
        calculate_metrics(pred_dict, gt_dict, metric="nDCG"),
        calculate_metrics(pred_dict, gt_dict, metric="FT"),
        calculate_metrics(pred_dict, gt_dict, metric="ST")
    )
    logger.info(log_message)


def main():
    args = parse_args()
    config = get_yaml()
    evaluate(args, config)

if __name__ == '__main__':
    main()