import cv2

def reverse_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        print("Error: The video could not be read or is empty.")
        return

    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

    for frame in reversed(frames):
        out.write(frame)

    out.release()
    print("Video successfully reversed.")

reverse_video("C:/Users/alpnn/Desktop/GPU/video.mp4", "dondurulmus_video.mp4")