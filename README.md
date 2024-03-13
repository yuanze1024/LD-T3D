# LD-T3D: A Large-scale and Diverse Benchmark for Text-based 3D Model Retrieval


## Benchmark

## Online Demo


## Installation
### All in One
We provide a image built from scratch.
### From Scratch
If you want to use some customized settings or you fail to pull our image, you can: 
1. Get the [Nvidia official pytorch image: 23.08](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-08.html#rel-23-08). 
```shell
docker pull nvcr.io/nvidia/pytorch:23.08
```
2. Enter a container, and install requirements
```shell
apt update
apt install libwebp-dev -y
git clone https://github.com/yuanze1024/LD-T3D.git
cd LD-T3D
pip install -r requirements.txt
pip install --force-reinstall pillow
pip install --no-deps salesforce-lavis==1.0.2
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
pip install git+https://github.com/openai/CLIP.git
```
**[Note]**: Try to change the CUDA_ARCH in [Pointnet](https://github.com/erikwijmans/Pointnet2_PyTorch/blob/b5ceb6d9ca0467ea34beb81023f96ee82228f626/pointnet2_ops_lib/setup.py#L19) if you are facing problems compiling it. For example, **8.0** for A100.



## Config
Set your huggingface cache_dir in [config/config.yaml](config/config.yaml).
**[Note]**: Make sure you set the ***general.cache_dir*** correctly, which means the dir where you put the downloaded pretrained checkpoints.

## Evaluation
We put the parameters that are more likely to be adjusted at run time in **args** for parsing, and the configurations that are less likely to be changed once determined in [config/config.yaml](config/config.yaml).

The methods' checkpoints will be downloaded automaticlly the first time you use a certain method.
```shell
# E5
python eval.py --option e5 --cross_modal text --batch_size 1024

# Uni3D
python eval.py --option Uni3D
```


## Eval Custom Method
Note that we only support dual-stream architecture by now, which means the embeddings of queries and multimodal features must be encoded seperately.

You can refer to encoders in `feature_extractors` and achieve your own method which inherits the base class `FeatureExtractor` in `feature_extractors/__init__.py`.


## Citation
~~~bib
our arxiv
~~~