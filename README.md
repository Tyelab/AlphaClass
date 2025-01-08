# AlphaClass
AlphaClass is a 2-D behavioral estimation method that uses keypoint detection to estimate behaviors in a single frame.

# Installation
AlphaClass runs on a Windows OS platform and uses a conda environment.
We have provide a yaml file with the requirements so you can easily create a conda environment.
```
conda env create --file alphaclass/AlphaClass_env.yml
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install albumentations==1.3.0 kornia==0.5.8 opencv-contrib-python==4.6.0.66 opencv-python==4.5.5.64 imutils==0.5.4
```

This version was tested on Windows 10 Pro with a GPU (Quadro RTX 4000), that had CUDA version 12.5 installed.



