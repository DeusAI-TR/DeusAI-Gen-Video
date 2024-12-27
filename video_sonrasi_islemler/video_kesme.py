import cv2

def cut_and_save_video(input_path, output_path, start_time, end_time):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    out = cv2.VideoWriter(output_path, fourcc, fps, (720, 480))

    while cap.isOpened():
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if current_frame > end_frame:
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (720, 480))

        out.write(frame)

    cap.release()
    out.release()
    print(f"Video segment saved to {output_path}")

cut_and_save_video("C:/Users/alpnn/Desktop/GPU/video.mp4", "kesilmis_video.mp4", start_time=0, end_time=2)