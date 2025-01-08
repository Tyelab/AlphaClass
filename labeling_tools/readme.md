# Extracting frames

Open Jupyter Notebook in extract_frames/GetThoseFrames_forALPHACLASS
 
the path to the GetThoseFrames Python code
Click get those frames
In the code, change the video path to the folder where you videos are and the output folder to the folder where you want the frames to go the videos should be in mp4 format
Sift through frames and divide them into folders for
- Attending (500)
- Nothing (50)
- Huddle (200)
- Climbing (300)
- OneFriendEat (200)


# Annotation
 
1.	www.makesense.ai
2.	Click on Get Started
3.	Upload your video frames
4.	Click on Object Detection
5.	Create your behavior labels 
	1.	Put one for each sub behavior
	2.	Attending, huddle, climbing, OneFriendEat
6.	Click on Start Project and begin labeling Point
7.	When finished, click on Actions → Export annotations → single CSV file 
	1.	name the CSV file: labels
8.	Create a folder with two subfolders and label the subfolders: images and labels 
	1.	In the images folder, upload your video frames
	2.	In the labels folder, upload the exported CSV file from makesense.ai and change the excel file to labels (from step 8 below)
9.	Move the folder containing the subfolders to the AlphaClass folder on Goat
