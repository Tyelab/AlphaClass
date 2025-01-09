# AlphaClass
## About
AlphaClass is a 2-D behavioral estimation method that primarily uses keypoint detection to estimate behaviors directly from single images.
First, AlphaClass receives an image as an input, such as a single frame from a video of behaving mice. 
Next, AlphaClass extracts features from these single frames via a downsampling approach using convolutional neural networks (CNN). 
Following this step, AlphaClass performs non-max suppression to remove any faulty detections or low-confidence detections. 
This allows AlphaClass to accurately regress the likely locations of behavioral labels in novel video data. 
Importantly, AlphaClass does not perform frame-by-frame tracking, but rather provides behavioral estimations on a frame-to-frame basis.

## General useage
Users first begin by defining areas of behavior by placing a point (circle) at the location of a particular behavior, such as placing a point on a grooming mouse, or placing a point at the joint between a mouse’s tail and another mouse’s nose to label chasing behaviors. Importantly, multiple behaviors can be labeled at a time, such as labeling two mice in a scene that are fighting, while simultaneously labeling a third mouse that is rearing. AlphaClass receives this training data and performs feature detection and keypoint estimation to estimate the likely location of those same user-defined behaviors in untrained frames of video. As a result, AlphaClass bypasses the pose estimation phase and directly provides behavioral labels from single frames of video, thereby providing a more direct method to estimate difficult-to-detect behaviors.


# Installation
AlphaClass runs on a Windows OS platform and uses a conda environment.
We have provide a yaml file with the requirements so you can easily create a conda environment.
```
conda env create --file alphaclass/AlphaClass_env.yml
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install albumentations==1.3.0 kornia==0.5.8 opencv-contrib-python==4.6.0.66 opencv-python==4.5.5.64 imutils==0.5.4
```

AlphaClass has been tested on Windows 10 Pro with a GPU (Quadro RTX 4000), that had CUDA version 12.5 installed.


# Labeling data
AlphaClass is a supervised classification method, which means it must have labeled data on which to train a model.  This amounts to a two step process. First, the user must extract frames from the video showing the behavior to label.  Then a point must be placed at the approximate location of the behavior in the image.  The image name, image size, training label, and label location is put into a csv format.  You will use a single csv for training AlphaClass; the csv can include multiple labels.

We have provided a jupyter notebook to extract individual frames from a video and save them as single images.  See the labeling_tools/extract_frames/.  

