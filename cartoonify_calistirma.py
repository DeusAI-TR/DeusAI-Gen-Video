import os
from cartoonify import Cartoonify

video_path = os.path.join(os.getcwd(), "C:/Users/alpnn/Desktop/GPU/video.mp4")
output_video_path = os.path.join(os.getcwd(), "cartoon.mp4")
style = "Paprika"

stylizer = Cartoonify(video_path, output_video_path, style)
stylizer.run()