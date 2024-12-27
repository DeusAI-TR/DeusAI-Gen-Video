import cv2

def edge_detection(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_path, fourcc, fps, (720, 480), isColor=False)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.resize(gray_frame, (720, 480))

        edges = cv2.Canny(gray_frame, 50, 150)

        out.write(edges)

    cap.release()
    out.release()
    print(f"Processed video saved to {output_path}")

edge_detection("C:/Users/alpnn/Desktop/GPU/video.mp4", "kenarlari_algilanmis_video.mp4")