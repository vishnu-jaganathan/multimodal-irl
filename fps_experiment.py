from fer import FER
from fer import Video
import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt
import time

def record_video(savename):
	
	vid = cv2.VideoCapture(0)
	# Check if camera opened successfully
	if (vid.isOpened() == False): 
		print("Unable to read camera feed")

	# Default resolutions of the frame are obtained.The default resolutions are system dependent.
	# We convert the resolutions from float to integer.
	frame_width = int(vid.get(3))
	frame_height = int(vid.get(4))
	fps_in_saved_video = 2

	out = cv2.VideoWriter(savename + '.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps_in_saved_video, (frame_width, frame_height))
	out2 = cv2.VideoWriter(savename + '-full' + '.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 60, (frame_width, frame_height))

	frame_count = 1
	while True:
		ret, frame = vid.read()
		if ret == True:
			out2.write(frame)
			if frame_count % 30 == 0: # save an image every 30 frames
				out.write(frame)
			frame_count += 1
			img = cv2.resize(frame, (frame_width // 2, frame_height // 2))
			cv2.imshow('frame', img)
			# quit by pressing q
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		else:
			break

	vid.release()
	out.release()
	out2.release()
	cv2.destroyAllWindows()

def process_emotions(filename):
	emotion_detector = FER(mtcnn=True)
	input_video = Video(filename + '.avi')

	# The Analyze() function will run analysis on every frame of the input video. 
	# It will create a rectangular box around every image and show the emotion values next to that.
	# Finally, the method will publish a new video that will have a box around the face of the human with live emotion values.
	start = time.time()
	processing_data = input_video.analyze(emotion_detector, display=False)
	print("Time elapsed for analysis:", time.time() - start)

	# We will now convert the analysed information into a dataframe.
	# This will help us import the data as a .CSV file to perform analysis over it later
	vid_df = input_video.to_pandas(processing_data)
	vid_df = input_video.get_first_face(vid_df)
	vid_df = input_video.get_emotions(vid_df)

	# Plotting the emotions against time in the video
	pltfig = vid_df.plot(figsize=(20, 8), fontsize=16).get_figure()
	plt.show()

	# Extract which emotion was prominent in the video
	num_frames = len(vid_df)
	angry = sum(vid_df.angry) / num_frames
	disgust = sum(vid_df.disgust) / num_frames
	fear = sum(vid_df.fear) / num_frames
	happy = sum(vid_df.happy) / num_frames
	sad = sum(vid_df.sad) / num_frames
	surprise = sum(vid_df.surprise) / num_frames
	neutral = sum(vid_df.neutral) / num_frames

	emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
	emotions_values = [angry, disgust, fear, happy, sad, surprise, neutral]

	score_comparisons = pd.DataFrame(emotions, columns = ['Human Emotions'])
	score_comparisons['Emotion Value from the Video'] = emotions_values
	return score_comparisons

name = "exp"
record_video(name)
score_base = process_emotions(name)
print(score_base)
score_full = process_emotions(name + "-full")
print(score_full)
