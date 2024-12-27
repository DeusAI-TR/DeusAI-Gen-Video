from anime import AnimeGANv2Processor

processor = AnimeGANv2Processor()

video_path = "C:/Users/alpnn/Desktop/GPU/video.mp4"
frames_folder = "frames"
output_frames_folder = "output_frames"
output_video_path = "anime.mp4"
frames_per_second = 24

processor.frame_frame(video_path, frames_folder, frames_per_second)
processor.process_frames(frames_folder, output_frames_folder)
processor.video_olustur(video_path, output_frames_folder, output_video_path)
processor.clean_up(frames_folder, output_frames_folder)