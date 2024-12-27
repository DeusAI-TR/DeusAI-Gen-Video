import cv2

def process_and_save_grayscale_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_path, fourcc, fps, (720, 480), isColor=False)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.resize(gray_frame, (720, 480))

        out.write(gray_frame)

    cap.release()
    out.release()
    print(f"Processed video saved to {output_path}")

process_and_save_grayscale_video("C:/Users/alpnn/Desktop/GPU/video.mp4", "siyah_beyaz_video.mp4")