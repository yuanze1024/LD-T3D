import numpy as np
from collections.abc import Mapping, Sequence

def NDCG(prediction: Sequence[str], GT: Sequence[str]):
    """Calculate the normalized discounted cumulative gain (nDCG) for a single query.
    """
    k = len(prediction)
    log2_table = np.log2(np.arange(2, len(prediction) + 2))

    def dcg_at_n(rel, n):
        rel = np.asfarray(rel)[:n]
        dcg = np.sum(np.divide(rel, log2_table[:rel.shape[0]]))
        return dcg

    idcg = dcg_at_n(np.ones(len(GT)), n=k)
    # If the corresponding source_id in prediction is not in GT, then rel is 0, otherwise 1.
    rel = np.array([1 if prediction[i] in GT else 0 for i in range(len(prediction))])
    dcg = dcg_at_n(rel, n=k)
    return 0 if idcg == 0 else dcg / idcg

def mNDCG(predictions: Mapping[str, Sequence[str]], GTs: Mapping[str, Sequence[str]]):
    """Calculate the mean NDCG for all sub-datasets in 'predictions'.

    Args:
        predictions: Key is the query_id, and value is the list, sorted in reverse order by relevance, of sourcE_id of retrieved 3D models.
        GTs: Key is the query_id, and value is the list of sourcE_id of ground truth 3D models.

    Returns:
        The mean NDCG for all sub-datasets in 'predictions'.
    """
    NDCG_list = []
    for query_id in predictions.keys():
        NDCG_list.append(NDCG(predictions[query_id], GTs[query_id]))
    return sum(NDCG_list) / len(NDCG_list)

def AP(prediction: Sequence[str], GT: Sequence[str]):
    """Calculate the average precision (AP) for a single query.

    We follow the google landmark challenge's AP definition, but change the K to len(GT) for a fair comparison because the size of a sub-dataset varies.
    See [here](https://github.com/tensorflow/models/blob/94583313e0d452e116405cc03e5867394dfcda92/research/delf/delf/python/datasets/google_landmarks_dataset/metrics.py#L119) for the original implementation.
    """
    ap = 0.
    num_correct = 0
    GT = set(GT)
    for i in range(len(prediction)):
        if prediction[i] in GT:
            num_correct += 1
            ap += num_correct / (i + 1)
    return ap / len(GT)

def mAP(predictions: Mapping[str, Sequence[str]], GTs: Mapping[str, Sequence[str]]):
    """Calculate the mean average precision (mAP) for all sub-datasets in 'predictions'.

    Args:
        predictions: Key is the query_id, and value is the list, sorted in reverse order by relevance, of sourcE_id of retrieved 3D models.
        GTs: Key is the query_id, and value is the list of sourcE_id of ground truth 3D models.

    Returns:
        The mAP for all sub-datasets in 'predictions'.
    """
    AP_list = []
    for query_id in predictions.keys():
        AP_list.append(AP(predictions[query_id], GTs[query_id]))
    return sum(AP_list) / len(AP_list)

def FT(predictions: Mapping[str, Sequence[str]], GTs: Mapping[str, Sequence[str]]):
    """FT is short for First Tier. 
    
    It represents the recall@K, where K is the number of objects that should be retrieved, which are GTs.
    """
    recall_list = []
    for query_id in predictions.keys():
        prediction = predictions[query_id][:len(GTs[query_id])]
        recall_list.append(len(set(prediction).intersection(set(GTs[query_id]))) / len(GTs[query_id]))
    return sum(recall_list) / len(recall_list)

def ST(predictions: Mapping[str, Sequence[str]], GTs: Mapping[str, Sequence[str]]):
    """ST is short for Second Tier. 

    It is basically recall@2K, where K is the number of objects that should be retrieved, which are GTs. 
    Note that 2K won't exceed the length of predictions.
    """
    recall_list = []
    for query_id in predictions.keys():
        prediction = predictions[query_id][:min(2 * len(GTs[query_id]), len(predictions[query_id]))]
        recall_list.append(len(set(prediction).intersection(set(GTs[query_id]))) / len(GTs[query_id]))
    return sum(recall_list) / len(recall_list)

def calculate_metrics(predictions, GTs, metric="mAP"):
    if metric == "mAP":
        result = mAP(predictions, GTs)
    elif metric == "nDCG":
        result = mNDCG(predictions, GTs)
    elif metric == "FT":
        result = FT(predictions, GTs)
    elif metric == "ST":
        result = ST(predictions, GTs)
    else:
        raise NotImplementedError
    return result