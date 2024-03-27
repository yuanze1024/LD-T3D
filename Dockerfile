FROM nvcr.io/nvidia/pytorch:23.08-py3

LABEL maintainer="yuanze"
LABEL email="yuanze1024@gmail.com"

WORKDIR /code

# Install webp support
RUN apt update && apt install libwebp-dev -y

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

RUN pip install --no-deps salesforce-lavis==1.0.2

RUN pip install git+https://github.com/openai/CLIP.git

# Change this to fit your GPU, e.g., 8.0 for A100, and 8.6 for 3090.
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"

# Install Pointnet2_PyTorch
RUN git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git

# replace the setup.py file to fix the error
COPY ./change_setup.bak /code/Pointnet2_PyTorch/pointnet2_ops_lib/setup.py

RUN cd Pointnet2_PyTorch/pointnet2_ops_lib \
    && pip install .
