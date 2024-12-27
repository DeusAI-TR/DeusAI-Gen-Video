import cv2

def blend_and_save_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_path, fourcc, fps, (720, 480))

    current_frame = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        alpha = (current_frame / frame_count) ** 2

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

        blended_frame = cv2.addWeighted(frame, 1 - alpha, gray_frame, alpha, 0)

        blended_frame = cv2.resize(blended_frame, (720, 480))

        out.write(blended_frame)

        current_frame += 1

    cap.release()
    out.release()
    print(f"Blended video saved to {output_path}")

blend_and_save_video("C:/Users/alpnn/Desktop/GPU/video.mp4", "gri_tonu_video.mp4")