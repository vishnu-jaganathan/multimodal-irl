from fer import FER
from fer import Video
import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt

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

	frame_count = 1
	while True:
		ret, frame = vid.read()
		if ret == True:
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
	cv2.destroyAllWindows()

def process_emotions(filename):
	emotion_detector = FER(mtcnn=True)
	input_video = Video(filename + '.avi')

	# The Analyze() function will run analysis on every frame of the input video. 
	# It will create a rectangular box around every image and show the emotion values next to that.
	# Finally, the method will publish a new video that will have a box around the face of the human with live emotion values.
	processing_data = input_video.analyze(emotion_detector, display=False)

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
	return score_comparisons, emotions_values

def extract_score(emotions):
	emotion_to_idx = {'Angry':0, 'Disgust':1, 'Fear':2, 'Happy':3, 'Sad':4, 'Surprise':5, 'Neutral':6}
	negative = sum(emotions[0:3]) + emotions[4]
	positive = emotions[3]
	if positive > 2 * negative:
		return 1
	if negative > 3 * positive:
		return -1
	if emotions[emotion_to_idx["Neutral"]] > 0.6: # threshold hyperparameter
		return 0
	if abs(1.5 * positive - negative) < 0.1:
		return 0
	return 1 if positive > negative else -1 # scoring system subject to change, simple positive negative

name = "output"
record_video(name)
scores, emotions = process_emotions(name)
print(scores, emotions)
score = extract_score(emotions)
print(score)