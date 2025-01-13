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
    a. Put one label for each sub behavior
    b. In the demo example, you will need labels for each of these behaviors: Attending, huddle, climbing, OneFriendEat
5. Click on **Start Project** and begin labeling
    a. Point
6. When finished, click on **Actions** → Export annotations → single CSV file
    a. name the CSV file: labels
7. Create a folder with two subfolders and label the subfolders: **images** and **labels**
    a. In the images folder, upload your video frames
    b. In the labels folder, upload the exported CSV file from makesense.ai and change the excel file to *labels (from step 8 below)
8. Move the folder containing the subfolders to the AlphaClass folder


