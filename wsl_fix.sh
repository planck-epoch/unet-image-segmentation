# https://medium.com/swlh/how-to-install-the-nvidia-cuda-toolkit-11-in-wsl2-88292cf4ab77


conda install tensorflow-gpu -c anaconda

sudo apt-get --yes install cuda-toolkit-11-8
sudo apt-get install --yes --no-install-recommends libcudnn8 libcudnn8-dev
sudo apt-get install --yes --no-install-recommends libnvinfer8 libnvinfer-dev libnvinfer-plugin8

# libcuda.so not found
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH