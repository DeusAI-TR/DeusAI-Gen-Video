import os  # Dosya ve klasör işlemleri için
import cv2  # Görüntü işleme için gerekli kütüphane
import torch  # PyTorch, derin öğrenme modellerini çalıştırmak için
import numpy as np  # Sayısal işlemler için
from PIL import Image  # Görüntü işlemede yardımcı kütüphane
import matplotlib.pyplot as plt  # Görüntüleri görselleştirmek için
import shutil  # Dosya ve klasörleri taşımak/silmek için

# AnimeGANv2 işlemlerini yapmak için sınıf
class AnimeGANv2Processor:
    def __init__(self, model_name="face_paint_512_v2", size=512):
        # AnimeGANv2 modelini yükle
        print("AnimeGANv2 modeli yükleniyor...")
        self.model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained=model_name).eval()
        print("AnimeGANv2 modeli başarıyla yüklendi!")

        # Face2paint yardımcı fonksiyonunu yükle
        print("Face2paint yardımcı fonksiyonu yükleniyor...")
        self.face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", size=size)
        print("Face2paint yardımcı fonksiyonu başarıyla yüklendi!")

    # Tek bir kareyi AnimeGANv2 ile işleyen metot
    def process_frame(self, frame):
        # OpenCV ile alınan kareyi PIL formatına dönüştür
        input_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # AnimeGANv2 kullanarak işlenmiş kareyi al
        output_image = self.face2paint(self.model, input_image)
        # Sonucu OpenCV formatına dönüştür ve geri döndür
        return cv2.cvtColor(np.array(output_image), cv2.COLOR_RGB2BGR)

    # Videoyu karelere ayıran metot
    def frame_frame(self, video_path, frames_folder, frames_per_second):
        print("Video karelere ayrılıyor...")
        os.makedirs(frames_folder, exist_ok=True)  # Kareler için klasör oluştur

        video = cv2.VideoCapture(video_path)  # Video dosyasını aç
        if not video.isOpened():
            raise ValueError("Video oynatılamadı!")  # Eğer video açılamazsa hata ver

        fps = video.get(cv2.CAP_PROP_FPS)  # Videonun FPS değerini al
        frame_index = 0  # Kare indeksini başlat

        while True:
            ret, frame = video.read()  # Videodan bir kare oku
            if not ret:  # Eğer video biterse döngüyü sonlandır
                break

            current_frame_time = video.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Şu anki kare zamanını saniye cinsine çevir

            # Belirli aralıklarla kareleri kaydet
            if int(current_frame_time * frames_per_second) > frame_index:
                frame_filename = os.path.join(frames_folder, f"frame_{frame_index:04d}.png")
                cv2.imwrite(frame_filename, frame)  # Kareyi dosyaya yaz
                frame_index += 1

    # Kareleri AnimeGANv2 ile işleyen metot
    def process_frames(self, frames_folder, output_frames_folder):
        print("Kareler işleniyor...")
        os.makedirs(output_frames_folder, exist_ok=True)  # İşlenmiş kareler için klasör oluştur

        # Kare dosyalarını sırayla listele
        frame_files = [f for f in sorted(os.listdir(frames_folder)) if f.endswith(".png")]

        for frame_file in frame_files:
            input_path = os.path.join(frames_folder, frame_file)
            output_path = os.path.join(output_frames_folder, frame_file)

            frame = cv2.imread(input_path)  # Kareyi oku
            processed_frame = self.process_frame(frame)  # AnimeGANv2 ile işle
            cv2.imwrite(output_path, processed_frame)  # İşlenmiş kareyi kaydet

    # İşlenmiş karelerden video oluşturan metot
    def video_olustur(self, video_path, output_frames_folder, output_video_path):
        print("İşlenmiş karelerden video oluşturuluyor...")
        video = cv2.VideoCapture(video_path)  # Orijinal video dosyasını aç
        if not video.isOpened():
            raise ValueError("Orijinal video oynatılamadı!")  # Eğer video açılamazsa hata ver

        fps = video.get(cv2.CAP_PROP_FPS)  # Orijinal videonun FPS değerini al
        frame_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # Kare boyutlarını al

        # İşlenmiş karelerin dosyalarını sırayla listele
        frame_filenames = sorted([
            os.path.join(output_frames_folder, fname) for fname in os.listdir(output_frames_folder)
            if fname.endswith(".png")
        ])

        if not frame_filenames:  # Eğer işlenmiş kare bulunmazsa hata ver
            raise ValueError("Belirtilen klasörde kareler bulunamadı!")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4 formatında bir video oluşturmak için codec
        out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)  # Video yazıcıyı başlat

        for frame_file in frame_filenames:
            frame = cv2.imread(frame_file)  # Kareyi oku
            if frame is None:  # Eğer kare okunamazsa hata ver
                raise ValueError(f"İstenilen dosyadan kareler okunamadı: {frame_file}")
            if (frame.shape[1], frame.shape[0]) != frame_size:  # Eğer kare boyutu uyumsuzsa yeniden boyutlandır
                frame = cv2.resize(frame, frame_size)

            out.write(frame)  # Kareyi videoya ekle

        out.release()  # Videoyu kaydet ve kapat
        print(f"Video başarı ile kaydedildi: {output_video_path}")

    # Geçici dosyaları temizleyen metot
    def clean_up(self, *folders):
        print("İşlem dosyaları siliniyor...")
        for folder in folders:  # Geçici klasörleri temizle
            try:
                if os.path.exists(folder):
                    shutil.rmtree(folder)
                    print(f"Dosya '{folder}' silindi.")
            except Exception as e:
                print(f"Silinme sırasında bir hata oluştu: {e}")