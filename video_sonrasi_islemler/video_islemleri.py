import cv2

class VideoIslemleri:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Video dosyası açılamadı!")

    def video_oynat(self, resize_dim=(800, 600)):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            new_frame = cv2.resize(frame, resize_dim)
            cv2.imshow('Video', new_frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def hiz_ayarlama(self, speed_factor=1.0, resize_dim=(800, 700)):
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        delay = int(1000 / fps / speed_factor)

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, resize_dim)
            cv2.imshow("Video", frame)

            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def siyah_beyaz(self, resize_dim=(800, 700)):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.resize(gray_frame, resize_dim)
            cv2.imshow("Siyah-Beyaz Video", gray_frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def gri_tonuna_kayan_video(self, resize_dim=(800, 700)):
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            alpha = (current_frame / frame_count) ** 2
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
            blended_frame = cv2.addWeighted(frame, 1 - alpha, gray_frame, alpha, 0)
            blended_frame = cv2.resize(blended_frame, resize_dim)

            cv2.imshow("Transition Video", blended_frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            current_frame += 1

        self.cap.release()
        cv2.destroyAllWindows()

    def renkliye_kayan_video(self, resize_dim=(800, 700)):
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            alpha = current_frame / frame_count
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
            blended_frame = cv2.addWeighted(frame, alpha, gray_frame, 1 - alpha, 0)
            blended_frame = cv2.resize(blended_frame, resize_dim)

            cv2.imshow("Transition to Color Video", blended_frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            current_frame += 1

        self.cap.release()
        cv2.destroyAllWindows()

    def bulaniklastirma(self, resize_dim=(800, 700), kernel_size=(25, 25)):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            blurred_frame = cv2.GaussianBlur(frame, kernel_size, 3)
            blurred_frame = cv2.resize(blurred_frame, resize_dim)
            cv2.imshow('Bulanık Video', blurred_frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def kenar_algilama(self, resize_dim=(800, 700), threshold1=50, threshold2=150):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.resize(gray_frame, resize_dim)
            edges = cv2.Canny(gray_frame, threshold1, threshold2)

            cv2.imshow('Kenar Algılama', edges)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def video_kesme(self, start_time, end_time, output_path, resize_dim=(800, 700)):
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, resize_dim)

        while self.cap.isOpened():
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            if current_frame > end_frame:
                break

            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, resize_dim)
            out.write(frame)

            cv2.imshow("Kesilmiş Video", frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        self.cap.release()
        out.release()
        cv2.destroyAllWindows()

# Kullanım Örneği
# video_processor = VideoIslemleri("video_yolu.mp4")
# video_processor.play_video()
# video_processor.change_speed(speed_factor=2.0)
# video_processor.apply_grayscale()
# video_processor.cut_video(5, 10, "output.mp4")