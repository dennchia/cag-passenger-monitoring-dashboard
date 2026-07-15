import cv2

print("Scanning for available cameras...")
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"✓ Camera found at index {i}: {width}x{height}")
        cap.release()
    else:
        cap.release()

print("\nIf no cameras found above, try your RTSP URL instead:")
print("python cctv_detect_humans_feet.py --source rtsp://your_camera_ip/stream")
