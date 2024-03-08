
import os
import json
import importlib
import torch
import pandas as pd
from tqdm import tqdm
from collections.abc import Sequence
import argparse
import yaml
from torch.utils.data import DataLoader

from utils.dataset import get_dataset
from utils.metrics import calculate_metrics


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
    parser.add_argument("--option", type=str, required=True, help="The feature extractor to be evaluated.")
    parser.add_argument("--cross_modal", type=str, default="", help="The modalities that are used, seperated by '_'. \
                        The default value is '', representing features extracted using the 'encode' method in the feature extractor. \
                        If not '', use the modalities included. E.g., 'text_image' or 'text_image_3D'.")
    parser.add_argument("--op", type=str, default="concat", choices=["concat", "add"], help="The operation to be used to fuse different embeddings.")
    parser.add_argument("--angles", nargs="+", default=[4], help="The angles to be used to extract the embeddings if modalities including 'Image'.")
    parser.add_argument("--batch_size", type=int, default=100, help="The batch size to be used to extract the embeddings.")
    parser.add_argument("--text_data_path", type=str, help="The path to caption csv.")
    parser.add_argument("--image_data_path", type=str, help="The path to image folder.")
    parser.add_argument("--3D_data_path", type=str, help="The path to point cloud npy folder.")
    return parser.parse_args()

def get_embedding(option, modality, source_id_list, data_path, encode_fn, angle=None, batch_size=128):
    save_path = f'data/objaverse_{option}_{modality + ("_" + str(angle)) if angle is not None else ""}_embeddings.pt'
    if os.path.exists(save_path):
        return torch.load(save_path)
    else:
        embeddings = []
        text_dataset = get_dataset(modality, source_id_list, data_path=data_path, angle=angle)
        dataloader = DataLoader(text_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=batch_size//4)
        for batch in tqdm(dataloader, desc=f"Extracting {modality} embeddings..."):
            embeddings.append(encode_fn(batch))
        embedding = torch.cat(embeddings, dim=0).cpu()
        torch.save(embedding, save_path)
        return embedding

def read_gt(result_path):
    """
    get a dict，key is source_id，value is GT source_id list
    """
    result = {}
    for file in sorted(os.listdir(result_path)):
        if file.endswith(".json"):
            with open(os.path.join(result_path, file), 'r') as f:
                data = json.load(f)
                assert type(data) == list
                for dct in data:
                    if dct['result']:
                        result[dct['query_id']] = result.get(dct['query_id'], []) + [dct['target_id']]
    return result

def read_queries(query_path):
    # get queries
    df = pd.read_csv(query_path, sep='\t')
    df.columns = ['id', 'source_id', 'caption', 'cn_caption', 'difficulty']
    df = df.set_index('source_id')
    source_to_caption = {}
    source_list = df.index.tolist()
    for query_id in source_list:
        source_to_caption[query_id] = df.at[query_id, 'caption']
    return source_to_caption
    
def get_yaml():
    file_yaml = 'config/config.yaml'
    rf = open(file=file_yaml, mode='r', encoding='utf-8')
    crf = rf.read()
    rf.close()  # 关闭文件
    yaml_data = yaml.load(stream=crf, Loader=yaml.FullLoader)
    return yaml_data

def evaluate(args, config):
    # parse args
    option = args.option
    angles = args.angles
    angles.sort()
    op = args.op
    batch_size = args.batch_size
    cross_modal = args.cross_modal

    # parse more configs
    text_data_path = config['data']['text_data_path']
    image_data_path = config['data']['image_data_folder_path']
    _3D_data_path = config['data']['3D_data_folder_path']
    federated_dataset_path = config['data']['federated_dataset_path']
    gt_result_path = config['data']['gt_result_folder_path']
    query_path = config['data']['query_path']

    # get federated dataset
    federated_dataset = torch.load(federated_dataset_path)

    # get all the source_id_list involved
    source_id_list = set()
    for value in federated_dataset.values():
        source_id_list.update(value)
    source_id_list = list(source_id_list)
    source_id_list.sort()
    
    # read Ground Truth
    gt_dict = read_gt(gt_result_path)
    
    # read queries
    source_to_caption = read_queries(query_path)

    # instantiate the feature extractor
    module = importlib.import_module(f"feature_extractors.{option}_embedding_encoder")
    encoder = getattr(module, f"{option.capitalize()}EmbeddingEncoder")(config.pop(option))
    
    # encode the embeddings
    embeddings = []
    if "text" in cross_modal:
        embeddings.append(get_embedding(option, "text", source_id_list, text_data_path, encoder.encode_text, batch_size=1024))
    if "image" in cross_modal:
        for angle in angles:
            embeddings.append(get_embedding(option, "image", source_id_list, image_data_path, encoder.encode_image, angle=angle, batch_size=batch_size))
    if "3D" in cross_modal:
        embeddings.append(get_embedding(option, "3D", source_id_list, _3D_data_path, encoder.encode_3D, batch_size=batch_size))
        
    ## fuse the embeddings
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
    xq = encoder.encode_query([source_to_caption[k] for k in federated_dataset.keys()])
    if op == "concat":
        xq = xq.repeat(1, embeddings.shape[-1] // xq.shape[-1]) # repeat to be aligned with the xb
        xq /= xq.norm(dim=-1, keepdim=True)
    
    pred_dict = predict(embeddings, xq, source_id_list, federated_dataset)

    # # 测试val集
    # import random
    # random.seed(2)
    # query_id_list = random.sample(list(gt_dict.keys()), 200)
    # new_pred_dict = {}
    # for query_id in query_id_list:
    #     new_pred_dict[query_id] = pred_dict[query_id]
    # pred_dict = new_pred_dict

    print(f"option: {option}, all:") # TODO: 把config和结果持久化
    print("\tmAP: ", calculate_metrics(pred_dict, gt_dict, metric="mAP"), end=", ")
    print("\tnDCG: ", calculate_metrics(pred_dict, gt_dict, metric="nDCG"), end=", ")
    print("\tFT: ", calculate_metrics(pred_dict, gt_dict, metric="FT"), end=", ")
    print("\tST: ", calculate_metrics(pred_dict, gt_dict, metric="ST"))


def main():
    args = parse_args()
    config = get_yaml()
    evaluate(args, config)

if __name__ == '__main__':
    main()