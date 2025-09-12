# Script to sort PNG files in left/right folders and create MP4 videos from each using OpenCV
import os
import cv2

def make_video_from_folder(folder, output_file, fps=30):
	# Get sorted list of PNG files
	files = [f for f in os.listdir(folder) if f.endswith('.png')]
	files.sort(key=lambda x: int(os.path.splitext(x)[0]))
	if not files:
		print(f"No PNG files found in {folder}")
		return
	# Read first image to get frame size
	first_frame = cv2.imread(os.path.join(folder, files[0]))
	height, width, layers = first_frame.shape
	fourcc = cv2.VideoWriter.fourcc(*'mp4v')
	video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
	for file in files:
		img = cv2.imread(os.path.join(folder, file))
		if img is None:
			print(f"Warning: Could not read {file}")
			continue
		video.write(img)
	video.release()
	print(f"Video saved to {output_file}")

if __name__ == "__main__":
	left_folder = "videos/left"
	right_folder = "videos/right"
	make_video_from_folder(left_folder, "videos/left.mp4")
	make_video_from_folder(right_folder, "videos/right.mp4")
# ...existing code...
