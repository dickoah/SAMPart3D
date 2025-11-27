### Installation

We test our model on a 24G RTX4090 GPU with Python 3.10, CUDA 12.1 and Pytorch 2.1.0.
We also support RTX5090 GPU with Python 3.11, CUDA 12.8 and PyTorch 2.7.0.

1. Install basic modules: torch and packages in requirements.txt
    ```bash
    conda create -n sampart3d python=3.11
    conda activate sampart3d
    pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128
    pip install -r requirements.txt
    ```

    **Note for RTX5090 (CUDA 12.8)**: If you encounter build issues with torch-scatter and tiny-cuda-nn, use these commands:
    ```bash
    pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
    pip install --no-build-isolation git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
    ```

2. Install modules for PTv3-object
    ```bash
    cd libs/pointops
    python setup.py install
    cd ../..

    # spconv (SparseUNet)
    # refer https://github.com/traveller59/spconv
    pip install spconv-cu120  # choose version match your local cuda
    ```

    Following [README](https://github.com/Dao-AILab/flash-attention) in Flash Attention repo and install Flash Attention for PTv3-object.


3. Install modules for acceleration (necessary in current version of code)
    ```bash
    pip install ninja
    pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

    # using GPU-based HDBSCAN clustering algorithm
    # refer https://docs.rapids.ai/install
    pip install --extra-index-url=https://pypi.nvidia.com cudf-cu11==24.6.* cuml-cu11==24.6.*
    ```

    **Important**: RAPIDS packages (cudf, cuml) require `numpy<2.0`. The `requirements.txt` pins numpy to `<2.0` to maintain compatibility. If you see a warning about numpy version conflicts with opencv-python, this is expected and safe - the pinned version in requirements.txt takes precedence.

    **Note**: For CUDA 12.8 with RTX5090, use `--no-build-isolation` flag if you encounter build errors:
    ```bash
    pip install --no-build-isolation git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
    ```
