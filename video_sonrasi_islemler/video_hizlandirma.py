import cv2

def speed_up_video(input_path, output_path, speed_factor=2.0):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    new_fps = int(fps * speed_factor)

    out = cv2.VideoWriter(output_path, fourcc, new_fps, (720, 480))

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (720, 480))

        out.write(frame)

    cap.release()
    out.release()
    print(f"Video saved to {output_path}")

speed_up_video("C:/Users/alpnn/Desktop/GPU/video.mp4", "hizlandirilmis_video.mp4", speed_factor=2.0)