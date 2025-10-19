from ultralytics import YOLO
import os
import cv2

# Load the trained YOLO model
model = YOLO("runs/detect/face-eye-detector/weights/best.pt")

# Folder where all results will be saved
output_folder = "results"
os.makedirs(output_folder, exist_ok=True)

# Ask the user what they want to process
choice = input("What do you want to process? (image/folder/video): ").strip().lower()

if choice == "image":
    img_path = input("Enter the path to the image: ").strip()
    if not os.path.isfile(img_path):
        print("Image not found!")
    else:
        print(f"Processing image: {img_path}")
        results = model.predict(source=img_path, save=True, conf=0.25, project=output_folder, name="single_image")
        for r in results:
            print(f"Detections for {img_path}:")
            for j, box in enumerate(r.boxes):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                print(f"  Detection {j+1}: Class = {r.names[cls]}, Confidence = {conf*100:.1f}%")
        print("Result saved in 'results/single_image'.")

elif choice == "folder":
    img_folder = input("Enter the path to the folder: ").strip()
    if not os.path.isdir(img_folder):
        print("Folder not found!")
    else:
        img_paths = [os.path.join(img_folder, f) for f in os.listdir(img_folder)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not img_paths:
            print("No images found in the folder!")
        else:
            print(f"Processing {len(img_paths)} images...")
            for i, img_path in enumerate(img_paths):
                print(f"\nImage {i+1}: {img_path}")
                results = model.predict(source=img_path, save=True, conf=0.25, project=output_folder, name="folder_images")
                for r in results:
                    for j, box in enumerate(r.boxes):
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        print(f"  Detection {j+1}: {r.names[cls]} ({conf*100:.1f}%)")
            print("All results saved in 'results/folder_images'.")

elif choice == "video":
    video_path = input("Enter the path to the video file (or press Enter to use the webcam): ").strip()

    # Open video file or webcam
    cap = cv2.VideoCapture(0 if video_path == "" else video_path)

    if not cap.isOpened():
        print("Failed to open video!")
    else:
        # Define output video parameters
        output_video_path = os.path.join(output_folder, "detected_video.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        print("Video processing started. Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run detection on the current frame
            results = model.predict(source=frame, conf=0.25, verbose=False)

            # Draw results on the frame
            annotated_frame = results[0].plot()

            # Show and save the annotated frame
            cv2.imshow("Detections", annotated_frame)
            out.write(annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Video saved at: {output_video_path}")

else:
    print("Invalid choice. Please enter 'image', 'folder', or 'video'.")
