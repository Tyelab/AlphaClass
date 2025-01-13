# Extracting frames

Open the Jupyter Notebook in extract_frames/GetThoseFrames_forALPHACLASS.ipynb
 
Update the path to the GetThoseFrames Python code.
Click get those frames.
In the code, change the video path to the folder where you videos are and the output folder to the folder where you want the frames to go the videos should be in mp4 format.

In the demo example, you will sift through frames and divide them into folders for
- Attending (500)  The alone mouse is attending to the port.
- Nothing (50)  
- Huddle (200)  The social mice have formed a huddle around the milkshake port
- Climbing (300)  The alone mouse is climbing
- OneFriendEat (200)  ??


# Annotating training data

Although you can annotate in whatever way works for you, we recommend using the website [www.makesense.ai](http://www.makesense.ai/).  
These instructions describe how to use that website for annotation, and assume you have extracted images from your videos for each behavior you wish to label.

1. Click on **Get Started**
2. Upload your video frames
3. Click on **Object Detection**
4. Create your behavior labels
4a. Put one label for each sub behavior
4b. In the demo example, you will need labels for each of these behaviors: Attending, huddle, climbing, OneFriendEat
5. Click on **Start Project** and begin labeling using **Point** option.
6. When finished, click on **Actions** → Export annotations → single CSV file
6a. Name the CSV file: labels.csv
7. Create a folder with two subfolders and label the subfolders: **images** and **labels**
7a. In the images folder, upload your video frames
7b. In the labels folder, upload the exported CSV file from makesense.ai and change the excel file to *labels (from step 8 below)
8. Move the folder containing the subfolders to the AlphaClass folder


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

Here is an example of how the labels.csv is organized.  Note that in this example, image4 contains multiple labels. 

|-----------|------|------|------------|----|-----|
| :--- |     :----: | ---: | :--- | :----: | ---: |
| Grooming  | 	236|	236 | image1.jpg |960|  540  |
| Nothing|	236|	236|	image2.jpg|	960|	540|
| Rearing|	283|	252|	image3.jpg|	960|	540|
| Rearing|	275|	266|	image4.jpg|	960|	540|
| Huddle|	275|	266|	image4.jpg|	960|	540|
| Nothing|	231|	413|	image5.jpg|	960|	540|
| Nothing|	236|	236|	image6.jpg|	960|	540|
|-----------|------|------|-------|----|------|
