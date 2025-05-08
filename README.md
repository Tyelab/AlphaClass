# AlphaClass
## About
AlphaClass is a 2-D behavioral estimation method that primarily uses keypoint detection to estimate behaviors directly from single images using a YOLO detection model.
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
cd to the AlphaClass folder (usually in Documents\GitHub\AlphaClass).
```
conda env create --file alphaclass\AlphaClass_env.yml -y
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install albumentations==1.3.0 kornia==0.5.8 opencv-contrib-python==4.6.0.66 opencv-python==4.5.5.64 imutils==0.5.4
```

AlphaClass has been tested on Windows 10 Pro with a GPU (Quadro RTX 4000), that had CUDA version 12.5 installed.


# Labeling data
AlphaClass is a supervised classification method, which means it must have labeled data on which to train a model.  Creating the training data is a two step process *Extraction* and *Annotation*. 

**Extraction** First, the user must extract frames from the video showing the behavior to label and sorts them into separate folders for each behavior.  
For example, if I extract 1000 frames, I would then divide them into folders for Rearing, Grooming, Huddle.  
It is a good idea to include negative examples, that is, example images that do not show the labeled behaviors.  However, do not include the label "Nothing" or "None". 
We have provided a jupyter notebook to extract individual frames from a video and save them as single images.  See the labeling_tools/extract_frames/.  

**Annotation** Next, a point must be placed at the approximate location of the behavior in the image.  The image name, image size, training label, and label location is put into a csv format.  You will use a single csv for training AlphaClass (the csv can include multiple labels).  We recommend using the website www.makesense.ai to help with annotation. 
1.	Go to www.makesense.ai
2.	Click on **Get Started**
3.	Upload your video frames
4.	Click on **Object Detection**
5.	Create your behavior labels.  Put one label for each sub behavior, such as Rearing, Huddle, Grooming, Nothing
6.	Click on **Start Project** and begin labeling using the ‘Point’ label type
7.	When finished, click on **Actions** → Export annotations → single CSV file
8.	Rename the CSV file: labels.csv
9.	Create a folder with two subfolders and label the subfolders: images and labels 
  -	In the images folder, upload your video frames; keep all images in a single folder.  Do not use the subfolder structure you used for annotation.
  -	In the labels folder, upload the exported CSV file from makesense.ai and change the excel filename to "labels.csv".

In the end, you should have a folder of extracted images saved in images, and a folder called labels containing a single file named labels.csv.
```
Training_Data/
└── images
    ├── image1.jpg
    ├── image2.jpg
    ├── image3.jpg
    ├── image4.jpg
    ├── image5.jpg
    ├── image6.jpg
└── labels
    ├── labels.csv
```

Here is an example of the labels.csv.  Note that image4 contains multiple labels. 

|-----------|------|------|------------|----|-----|
| :--- |     :----: | ---: | :--- | :----: | ---: |
| Grooming  | 	236|	236 | image1.jpg |960|  540  |
| Rearing|	283|	252|	image3.jpg|	960|	540|
| Rearing|	275|	266|	image4.jpg|	960|	540|
| Huddle|	275|	266|	image4.jpg|	960|	540|
|-----------|------|------|-------|----|------|

Note, the image2.jpg, image5.jpg and image6.jpg are examples where there is not grooming, rearing or huddle behavior occuring.  These remain in your image directory but you do not need to label these images, as it is understood they represent no behavior.

## Training the model
After you have set up your training_data directory with the appropriate examples of labels and images directories, and added labels.csv to your labels directory, you are ready to train the model.  Update the file standard.json to point to your training_data folder in `labeled_data_path`, correct the image size of your frames as needed in `image_training_width` and `image_training_height`, set the batch size as needed for your GPU and any other features of the training model.  Then run:
```
cd alphaclass\src
python train.py --options standard.json
```
Results for the trained model will be saved in the Results directory.	Every time you run command: `python train.py --options standard.json`, whether it be successful or not, it will create folder called run0,1,..,.n When successful, each run folder contains 5 files:
1.	`augmentation.json` contains the parameters for augmenting the dataset during training.
2.	`metrics.txt` (can load onto jupyter notebook and plot) 
	each column is `epoch #, train loss, test loss, learning rate/over time`
3.	The output of the best model, e.g. `resnet.best.pt`
4.	The output of the most recent model, e.g. `resnet.last.pt`
5.	The parameters file `options.json` contains location of where results are saved (`exp_path`), which video it was run on (`streams`), what model weights were used for inference (`weights_path`), the original training data folder (`labeled_data_path`), and other parameters used for inference.


## Tracking/running inference on all frames
Let's assume you have successfully trained a model and results are output to this folder: Results\run0.  In the following examples, you might need to edit the file paths to match your outputs.

To track the labels across all video frames, you would run
```
cd alphaclass\src
python video_inference.py --options ..\..\Results\run0\options.json --streams <<Full_Path_to_Video>> --weights_path ..\..\Results\run0\resnet.best.pt
```
where `<<Full_Path_to_Video>>` is the full path to the video you want to track.  
The results are saved to the `Results\run0\video_inference_runs\run0`.  

You can visualize the output using the plot function, which will add colored points to your video on frames containg given behavior labels.
```
python plot.py --options ..\..\Results\run0\video_inference_runs\run0\video_inference_options.json --streams <<Full_Path_to_Video>>
```

## Organization of output h5 file
The main output of the video inference is a data.h5 file.  This file contains outputs for each frame and its specific form depends on the number of labels and how many overlapping labels are present in your dataset.  
1. `points`
2. `time`
3. `confs`

We have provided an h5_to_csv.py file to convert this data to a more easily useable format for viewing detected labels across videos.  [add function]


## To add
- citation: [COMING SOON] please reference Caroline's paper and Aneesh Bal contributions as primary author.
- reference ultralytics yolov4 too?
- Bug Fix: why aren't predictions showing up in the right place?  (scaling problem somewhere)
- Demo: add link to demo dataset; add notes on using demo dataset to get started.






