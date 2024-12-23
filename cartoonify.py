import os  # Dosya ve klasör işlemleri için
import cv2  # Görüntü işleme için gerekli kütüphane
import shutil  # Dosya ve klasörleri taşımak/silmek için
from PIL import Image  # Görüntü işlemede yardımcı kütüphane
import torch  # PyTorch, derin öğrenme modellerini çalıştırmak için
from tqdm import tqdm  # İşlem ilerleme çubuğu göstermek için
from network.Transformer import Transformer  # Cartoonify modeli
from test_from_code import transform  # Görüntü dönüştürme işlemini yapan fonksiyon

# Video üzerinde "Cartoonify" işlemleri yapmak için sınıf
class Cartoonify:
    def __init__(self, video_path, output_video_path, style, pretrained_models_dir="pretrained_models", frames_per_second=24):
        # Sınıfın temel özelliklerini ayarla
        self.video_path = video_path  # İşlenecek video dosyası
        self.output_video_path = output_video_path  # Çıktı olarak üretilecek video yolu
        self.style = style  # Kullanılacak stil
        self.pretrained_models_dir = pretrained_models_dir  # Önceden eğitilmiş modellerin bulunduğu klasör
        self.frames_per_second = frames_per_second  # İşlenecek karelerin sayısı (FPS)
        self.frames_folder = os.path.join(os.getcwd(), "frames")  # Karelerin geçici olarak kaydedileceği klasör
        self.output_frames_folder = os.path.join(os.getcwd(), "output_frames")  # İşlenmiş karelerin kaydedileceği klasör
        self.models = {}  # Modelleri saklamak için bir sözlük

    # Videoyu karelere ayıran metot
    def frame_frame(self):
        print("Video karelere ayrılıyor...")
        os.makedirs(self.frames_folder, exist_ok=True)  # Karelerin kaydedileceği klasörü oluştur

        video = cv2.VideoCapture(self.video_path)  # Videoyu aç
        if not video.isOpened():  # Eğer video açılamazsa hata ver
            raise ValueError("Video oynatılamadı!")

        fps = video.get(cv2.CAP_PROP_FPS)  # Videonun FPS değerini al
        frame_index = 0  # Kare indeksini sıfırla

        while True:
            ret, frame = video.read()  # Videodan bir kare oku
            if not ret:  # Eğer video biterse döngüyü sonlandır
                break

            current_frame_time = video.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Şu anki kare zamanını saniye cinsine çevir

            # Belirli aralıklarla kareleri kaydet
            if int(current_frame_time * self.frames_per_second) > frame_index:
                frame_filename = os.path.join(self.frames_folder, f"frame_{frame_index:04d}.png")
                cv2.imwrite(frame_filename, frame)  # Kareyi dosyaya yaz
                frame_index += 1

    # Gerekli stil modelini yükleyen metot
    def model_yukle(self):
        print("Modeller yükleniyor...")
        for style in tqdm([self.style], desc="Loading models"):  # Belirtilen stil için modeli yükle
            model_path = os.path.join(self.pretrained_models_dir, f"{style}_net_G_float.pth")
            model = Transformer()  # Transformer modeli oluştur
            model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)  # Modeli yükle
            model.eval()  # Modeli değerlendirme moduna al
            self.models[style] = model  # Modeli sözlüğe ekle

    # Kareleri işleyen metot
    def kare_isleme(self):
        print("Kareler işleniyor...")
        os.makedirs(self.output_frames_folder, exist_ok=True)  # İşlenmiş kareler için klasör oluştur

        frame_files = sorted([
            os.path.join(self.frames_folder, fname) for fname in os.listdir(self.frames_folder)
            if fname.endswith(".png")  # Sadece PNG dosyalarını işle
        ])

        for frame_file in tqdm(frame_files, desc="Stylizing frames"):
            output_image = transform(self.models, self.style, frame_file)  # Görüntüyü stilize et
            output_image_path = os.path.join(self.output_frames_folder, os.path.basename(frame_file))
            output_image.save(output_image_path, format="PNG")  # Stilize edilmiş kareyi kaydet

    # İşlenmiş karelerden video oluşturan metot
    def video_olustur(self):
        print("İşlenmiş karelerden video oluşturuluyor...")
        video = cv2.VideoCapture(self.video_path)  # Orijinal videoyu aç
        if not video.isOpened():  # Eğer video açılamazsa hata ver
            raise ValueError("Orijinal video oynatılamadı!")

        fps = video.get(cv2.CAP_PROP_FPS)  # Orijinal videonun FPS değerini al
        frame_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # Kare boyutlarını al

        frame_filenames = sorted([
            os.path.join(self.output_frames_folder, fname) for fname in os.listdir(self.output_frames_folder)
            if fname.endswith(".png")  # Sadece PNG dosyalarını oku
        ])

        if not frame_filenames:  # Eğer işlenmiş kare bulunmazsa hata ver
            raise ValueError("Belirtilen klasörde kareler bulunamadı!")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4 formatında bir video oluşturmak için codec
        out = cv2.VideoWriter(self.output_video_path, fourcc, fps, frame_size)  # Video yazıcıyı başlat

        for frame_file in frame_filenames:
            frame = cv2.imread(frame_file)  # Kareyi oku
            if frame is None:  # Eğer kare okunamazsa hata ver
                raise ValueError(f"İstenilen dosyadan kareler okunamadı: {frame_file}")
            if (frame.shape[1], frame.shape[0]) != frame_size:  # Eğer kare boyutu uyumsuzsa yeniden boyutlandır
                frame = cv2.resize(frame, frame_size)

            out.write(frame)  # Kareyi videoya ekle

        out.release()  # Videoyu kaydet ve kapat
        print(f"Video başarı ile kaydedildi: {self.output_video_path}")

    # Geçici dosyaları temizleyen metot
    def clean_up(self):
        print("İşlem dosyaları siliniyor...")
        for folder in [self.frames_folder, self.output_frames_folder]:  # Geçici klasörleri temizle
            try:
                if os.path.exists(folder):
                    shutil.rmtree(folder)
                    print(f"Dosya '{folder}' silindi.")
            except Exception as e:
                print(f"Silinme sırasında bir hata oluştu: {e}")

    # Tüm işlemleri sırasıyla gerçekleştiren ana metot
    def run(self):
        self.model_yukle()  # Modeli yükle
        self.frame_frame()  # Karelere ayır
        self.kare_isleme()  # Kareleri işle
        self.video_olustur()  # Videoyu oluştur
        self.clean_up()  # Geçici dosyaları temizle