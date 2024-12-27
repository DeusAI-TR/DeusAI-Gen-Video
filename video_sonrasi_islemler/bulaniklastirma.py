import cv2

def blur_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break

        blurred_frame = cv2.GaussianBlur(frame, (25, 25), 3)

        out.write(blurred_frame)

    cap.release()
    out.release()
    print(f"Blurred video saved to {output_path}")

blur_video("C:/Users/alpnn/Desktop/GPU/video.mp4", "bulaniklastirilmis_video.mp4")