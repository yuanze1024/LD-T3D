from datasets import load_dataset


class Map:
    def __init__(self, id_list, obj_list) -> None:
        self.map = {}
        for i, obj in zip(id_list, obj_list):
            self.map[i] = obj
    
    def __getitem__(self, key):
        return self.map[key]

def get_dataset(modality, id_list, cache_dir, angle=None, img_transform=None):
    """
    Return a list of objects in the same order corresponding to the given id_list.
    """
    repo = "VAST-AI/LD-T3D"

    if modality == "3D": # return [torch.Tensor]
        dataset = load_dataset(repo, name="pc_npy", split="base", cache_dir=cache_dir)
        dataset.set_format("torch")
        obj_list = dataset['pc'] 
    elif modality == "image": # return [torch.Tensor]
        angles = ["diag_below", "diag_above", "right", "left", "back", "front", "above", "below"]
        assert angle in angles, f"Unsupported angle: {angle}, supported angles: {angles}"
        # dataset is a list of PIL.Image.Image
        dataset = load_dataset(repo, name=f"rendered_imgs_{angle}", split="base", cache_dir=cache_dir) 

        def transform(example):
            return {'image': img_transform(example['image'][0])}
        # transform image into torch.Tensor (and perhaps other operations) according to img_transform
        dataset = dataset.with_transform(transform) 

        obj_list = dataset['image'] 
    elif modality == "text": # return [caption]
        data_files = {"captions": "Cap3D_automated_Objaverse_no3Dword.csv"}
        dataset = load_dataset("tiange/Cap3D", data_files=data_files, names=["source_id", "caption"], header=None, split='captions', cache_dir=cache_dir)
        obj_list = dataset['caption'] 
    else:
        raise ValueError(f"Unsupported modality: {modality}")
    
    map = Map(dataset['source_id'], obj_list)
    return [map[i] for i in id_list]

def get_rel_dataset(cache_dir):
    """
    Return a list of dictionary of relations.

    Example:
    ```csv
    query_id, target_ids, GT_ids, caption
    abc, [bcd,efg], [bcd], caption
    ```
    """
    repo = "VAST-AI/LD-T3D"
    dataset = load_dataset(repo, name="relations", split="full", cache_dir=cache_dir)
    return dataset

if __name__ == '__main__':
    # from torch.utils.data import DataLoader
    # import torch
    
    # federated_dataset = torch.load("data/federated_dataset.pt")
    # source_id_list = set()
    # for value in federated_dataset.values():
    #     source_id_list.update(value)
    # source_id_list = list(source_id_list)
    # source_id_list.sort()

    # # test 3D
    # ds = get_dataset("3D", source_id_list)

    # # test image
    # transform = transforms.Compose([
    #     transforms.ToTensor()
    # ])
    # ds = get_dataset("image", source_id_list, "diag_below", img_transform=transform)
    # dataloader = DataLoader(ds, batch_size=1)

    # # test text
    # ds = get_dataset("text", source_id_list)

    # print(source_id_list[0])
    # dataloader = DataLoader(ds, batch_size=1)
    # for batch in dataloader:
    #     print(batch) 
    #     break

    get_rel_dataset()