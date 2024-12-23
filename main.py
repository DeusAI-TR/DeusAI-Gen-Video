import os
from cartoonify import Cartoonify
from DeepFace import DuyguAnalizi
from anime import AnimeGANv2Processor


islem = int(input("(Cartoonify: 1, Anime: 2, DeepFace: 3)\nAlgoritma Seçiminizi Giriniz: "))

if islem == 1:
    video_path = os.path.join(os.getcwd(), "kızgın8.mp4")
    output_video_path = os.path.join(os.getcwd(), "cartoonify.mp4")
    style = "Paprika"

    stylizer = Cartoonify(video_path, output_video_path, style)
    stylizer.run()
    

elif islem == 2:
    processor = AnimeGANv2Processor()

    video_path = "Video Yolu Girilecek"
    frames_folder = "frames"
    output_frames_folder = "output_frames"
    output_video_path = "anime.mp4"
    frames_per_second = 24

    processor.frame_frame(video_path, frames_folder, frames_per_second)
    processor.process_frames(frames_folder, output_frames_folder)
    processor.video_olustur(video_path, output_frames_folder, output_video_path)
    processor.clean_up(frames_folder, output_frames_folder)


elif islem == 3:
    video_path = "Video Yolu Girilecek" 
    frames_folder = "frames"
    analyzer = DuyguAnalizi(frames_folder, video_path)
    analyzer.frame_frame()
    analyzer.duygu_analizi()
    analyzer.temizle()


else:
    print("Geçersiz İşlem!")