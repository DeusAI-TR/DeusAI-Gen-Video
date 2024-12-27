import cv2

def pan_and_zoom(input_path, output_path, pan_factor):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        zoom_width = int(width / pan_factor)
        zoom_height = int(height / pan_factor)
        x1 = (width - zoom_width) // 2
        x2 = x1 + zoom_width
        y1 = (height - zoom_height) // 2
        y2 = y1 + zoom_height

        if zoom_width > 0 and zoom_height > 0 and x2 <= width and y2 <= height:
            zoomed_frame = cv2.resize(frame[y1:y2, x1:x2], (width, height))
            out.write(zoomed_frame)
        else:
            print("Warning: Invalid crop dimensions. Skipping frame.")

    cap.release()
    out.release()
    print("Pan and zoom applied successfully.")

pan_and_zoom("C:/Users/alpnn/Desktop/GPU/video.mp4", "kaydirilmis_yaklastirilmis_video.mp4", pan_factor = 1.5)