# LD-T3D: A Large-scale and Diverse Benchmark for Text-based 3D Model Retrieval


## Benchmark

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th rowspan="2">Features</th>
<th colspan="4">All</th>
<th colspan="2">Easy</th>
<th colspan="2">Medium</th>
<th colspan="2">Hard</th>
</tr>
<tr>
<th>mAP</th>
<th>mNDCG</th>
<th>mFT</th>
<th>mST</th>
<th>mAP</th>
<th>mFT</th>
<th>mAP</th>
<th>mFT</th>
<th>mAP</th>
<th>mFT</th>
</tr>
</thead>
<tbody align="center">
<tr>
<td rowspan="1"><a href="https://huggingface.co/intfloat/e5-large-v2">E5</a></td>
<td>Text</td>
<td>64.5</td>
<td>82.1</td>
<td>58.4</td>
<td>79.6</td>
<td>69.4</td>
<td>64.9</td>
<td>62.4</td>
<td>55.9</td>
<td>61.7</td>
<td>54.4</td>
</tr>
<tr>
<td><a href="https://github.com/SeanLee97/AnglE">AnglE</a></td>
<td>Text</td>
<td>66.8</td>
<td>83.4</td>
<td>60.8</td>
<td>81.1</td>
<td>71.3</td>
<td>66.8</td>
<td>65.7</td>
<td>59.0</td>
<td>63.3</td>
<td>56.6</td>
</tr>
<tr>
<td rowspan="3"><a href="https://github.com/openai/CLIP">CLIP</a></td>
<td>Text</td>
<td>59.5</td>
<td>79.7</td>
<td>53.6</td>
<td>74.2</td>
<td>59.6</td>
<td>54.8</td>
<td>59.1</td>
<td>52.3</td>
<td>59.9</td>
<td>53.8</td>
</tr>
<tr>
<td>Image</td>
<td>66.1</td>
<td>82.4</td>
<td>59.8</td>
<td>79.9</td>
<td>73.8</td>
<td>68.2</td>
<td>66.1</td>
<td>59.1</td>
<td>58.2</td>
<td>51.7</td>
</tr>
<tr>
<td>Text & Image</td>
<td>69.7</td>
<td>85.1</td>
<td>63.0</td>
<td>83.2</td>
<td>74.1</td>
<td>68.3</td>
<td>69.4</td>
<td>62.7</td>
<td>65.5</td>
<td>57.6</td>
</tr>
<tr>
<td rowspan="3"><a href="https://github.com/salesforce/LAVIS/tree/main/projects/blip2">BLIP2</a></td>
<td>Text</td>
<td>56.5</td>
<td>77.3</td>
<td>50.3</td>
<td>71.8</td>
<td>58.4</td>
<td>52.4</td>
<td>53.7</td>
<td>47.6</td>
<td>57.8</td>
<td>51.3</td>
</tr>
<tr>
<td>Image</td>
<td>68.6</td>
<td>84.1</td>
<td>62.5</td>
<td>81.7</td>
<td>74.5</td>
<td>69.4</td>
<td>68.2</td>
<td>61.9</td>
<td>63.1</td>
<td>56.1</td>
</tr>
<tr>
<td>Text & Image</td>
<td>70.0</td>
<td>84.9</td>
<td>63.6</td>
<td>83.2</td>
<td>75.0</td>
<td>69.6</td>
<td>69.1</td>
<td>62.4</td>
<td>66.0</td>
<td>58.7</td>
</tr>
<tr>
<td rowspan="3"><a href="https://github.com/Colin97/OpenShape_code">Openshape</a></td>
<td>3D</td>
<td>51.9</td>
<td>73.1</td>
<td>46.5</td>
<td>67.0</td>
<td>63.6</td>
<td>58.8</td>
<td>50.8</td>
<td>45.8</td>
<td>40.9</td>
<td>34.5</td>
</tr>
<tr>
<td>3D & Image</td>

<td>70.2</td>
<td>85.1</td>
<td>63.7</td>
<td>82.6</td>
<td>76.9</td>
<td>71.5</td>
<td>70.0</td>
<td>62.7</td>
<td>63.5</td>
<td>56.7</td>
</tr>
<tr>
<td>3D & Image & Text</td>
<td>74.3</td>
<td>87.8</td>
<td>67.0</td>
<td>86.1</td>
<td>78.4</td>
<td>72.4</td>
<td>74.5</td>
<td>66.7</td>
<td>69.9</td>
<td>61.6</td>
</tr>
<tr>
<td rowspan="3"><a href="https://github.com/baaivision/Uni3D">Uni3D</a></td>
<td>3D</td>
<td>66.8</td>
<td>82.5</td>
<td>60.5</td>
<td>80.3</td>
<td>76.8</td>
<td>72.0</td>
<td>64.5</td>
<td>58.3</td>
<td>59.0</td>
<td>51.0</td>
</tr>
<tr>
<td>3D & Image</td>
<td>75.0</td>
<td>87.9</td>
<td>68.3</td>
<td>86.8</td>
<td>81.0</td>
<td>75.7</td>
<td>74.4</td>
<td>67.5</td>
<td>69.6</td>
<td>61.8</td>
</tr>
<tr>
<td>3D & Image & Text</td>
<td>77.1</td>
<td>89.3</td>
<td>70.0</td>
<td>88.3</td>
<td>81.4</td>
<td>75.8</td>
<td>76.8</td>
<td>69.1</td>
<td>73.0</td>
<td>65.1</td>
</tr>
</tbody>
</table>

## Online Demo
All our experiments are conducted undeer the GPU A100. You can try our [online demo](https://huggingface.co/spaces/VAST-AI/LD-T3D) for visualization of ***Uni3D***'s retrieval results.

## Installation
**[Note]**: The installation steps is not necessary to use our dataset, which you can easily use in [HF Dataset](https://huggingface.co/datasets/VAST-AI/LD-T3D). The docker image may be quite heavy because it involves all the requirements of retrieval methods mentioned in our benchmark. If you only want to try one of those methods, you can refer to their official code repo.
### All in One
We provide a built image [yuanze1024/LD-T3D](https://hub.docker.com/repository/docker/yuanze1024/ld-t3d/general).

You can use it by:
```
docker pull yuanze1024/ld-t3d:v1
```
### From Scratch
If you fail to pull our image, you can build from the Dockerfile:
```shell
git clone https://github.com/yuanze1024/LD-T3D.git
cd LD-T3D
docker build -t ld-t3d .
```
**[Note]**: Change the TORCH_CUDA_ARCH_LIST in [Dockerfile](Dockerfile) for compilation, e.g., **8.0** for A100, and **8.6** for 3090.


## Config
Set your huggingface cache_dir in [config/config.yaml](config/config.yaml).
**[Note]**: Make sure you set the ***general.cache_dir*** correctly, which means the dir where you put the downloaded pretrained checkpoints.

## Evaluation
The methods' checkpoints will be downloaded automaticlly the first time you use a certain method.
```shell
# E5
python eval.py --option e5 --cross_modal text --batch_size 1024
# AnglE
python eval.py --option angle --cross_modal text --batch_size 1024
# CLIP
python eval.py --option clip --cross_modal image --angles diag_above --batch_size 256
# BLIP2
python eval.py --option blip2 --cross_modal image --angles diag_above --batch_size 256
# Openshape
python eval.py --option Openshape --cross_modal text_image_3D --op add --angles diag_above --batch_size 256
# Uni3D
python eval.py --option Uni3D --cross_modal text_image_3D --op add --angles diag_above --batch_size 256
```


## Eval Custom Method
Note that we only support dual-stream architecture by now, which means the embeddings of queries and multimodal features must be encoded seperately.

You can refer to encoders in `feature_extractors` and achieve your own method which inherits the base class `FeatureExtractor` in `feature_extractors/__init__.py`. BTW, if you want to use image modality, you also need to implement a `get_img_transform` function.
## Citation
~~~bib
not published yet
~~~