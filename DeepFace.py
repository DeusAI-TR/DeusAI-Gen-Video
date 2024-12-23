import cv2  # Görüntü işleme için gerekli kütüphane
import os  # Dosya ve klasör işlemleri için
import json  # JSON dosya işlemleri için
import shutil  # Dosya ve klasör taşımak/silmek için
from deepface import DeepFace  # Derin öğrenme tabanlı yüz ve duygu analizi için
from deep_translator import GoogleTranslator  # Duygu adlarını çevirmek için

# Duygu analizi işlemleri için bir sınıf tanımlıyoruz
class DuyguAnalizi:
    def __init__(self, frames_folder, video_path, output_file="DuyguAnaliziSonuclari.json"):
        # Sınıfın temel özelliklerini tanımlıyoruz
        self.video_path = video_path  # Analiz yapılacak video yolu
        self.frames_folder = frames_folder  # Karelerin kaydedileceği klasör
        self.output_file = output_file  # Analiz sonuçlarının kaydedileceği dosya
        self.emotion_counts = {}  # Duygu frekanslarını saklamak için bir sözlük

    # Videodan kareleri çıkarıp belirtilen klasöre kaydeden metot
    def frame_frame(self):
        # Klasör oluşturuluyor (varsa hata almayız)
        os.makedirs(self.frames_folder, exist_ok=True)

        # Videoyu aç
        video = cv2.VideoCapture(self.video_path)
        fps = video.get(cv2.CAP_PROP_FPS)  # Videonun FPS değeri
        frame_index = 0  # Başlangıçta kare indeksi sıfır
        frames_per_second = 24  # Her saniyeden alınacak kare sayısı

        while True:
            # Videodan bir kare oku
            ret, frame = video.read()
            if not ret:  # Video bittiğinde döngüden çık
                break

            # Şu anki kare zamanını saniye olarak hesapla
            current_frame_time = video.get(cv2.CAP_PROP_POS_MSEC) / 1000
            
            # Belirli aralıklarla kare kaydet
            if int(current_frame_time * frames_per_second) > frame_index:
                frame_filename = os.path.join(self.frames_folder, f"frame_{frame_index:04d}.jpg")
                cv2.imwrite(frame_filename, frame)  # Kareyi dosyaya kaydet
                frame_index += 1

        video.release()  # Video dosyasını serbest bırak
        print(f"Tüm kareler '{self.frames_folder}' klasörüne kaydedildi.")

    # Kareler üzerinde duygu analizi yapan metot
    def duygu_analizi(self):
        # Kare dosyalarını sırayla al
        frame_files = sorted([f for f in os.listdir(self.frames_folder) if f.endswith(('.jpg', '.png'))])

        for frame_file in frame_files:
            frame_path = os.path.join(self.frames_folder, frame_file)  # Her bir kare yolu

            try:
                # DeepFace ile duygu analizi yap
                result = DeepFace.analyze(img_path=frame_path, actions=['emotion'])
                dominant_emotion = result[0]["dominant_emotion"]  # En baskın duygu

                # Duygu sayacını güncelle
                if dominant_emotion in self.emotion_counts:
                    self.emotion_counts[dominant_emotion] += 1
                else:
                    self.emotion_counts[dominant_emotion] = 1

                print(f"{frame_file} - Dominant Emotion: {dominant_emotion}")
            except Exception as e:
                print(f"{frame_file} için analiz hatası: {e}")

        # Analiz sonuçlarını işleyip kaydet
        self.baskin_duygu()
        self.kaydet()
        print("Duygu analizi tamamlandı ve sonuçlar JSON dosyasına kaydedildi.")

    # En baskın duyguyu bulan ve tercüme eden metot
    def baskin_duygu(self):
        dominant_emotion = max(self.emotion_counts, key=self.emotion_counts.get)  # En sık görülen duygu
        
        # Sadece en baskın duyguyu sakla
        self.emotion_counts = {dominant_emotion: self.emotion_counts[dominant_emotion]}

        # İngilizce duyguyu Türkçeye çevir
        translated_text = GoogleTranslator(source='en', target='tr').translate(dominant_emotion)
        
        print(f"Kişinin Duygu Durumu: {translated_text}")

    # Sonuçları bir JSON dosyasına kaydeden metot
    def kaydet(self):
        with open(self.output_file, "w") as json_file:
            json.dump(self.emotion_counts, json_file, indent=4)  # Sonuçları JSON formatında yaz

    # Karelerin olduğu klasörü temizleyen metot
    def temizle(self):
        try:
            if os.path.exists(self.frames_folder):  # Klasör varsa sil
                shutil.rmtree(self.frames_folder)
                print(f"Klasör '{self.frames_folder}' silindi.")
        except Exception as e:
            print(f"Temizleme sırasında hata oluştu: {e}")