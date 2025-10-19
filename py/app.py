import os
import sys
import cv2
import ultralytics
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO

# === CONFIG CHECK ===
cfg_path = os.path.join(os.path.dirname(ultralytics.__file__), "cfg", "default.yaml")
if not os.path.exists(cfg_path):
    print(f"Warning: YOLO config not found at {cfg_path}")

# === PATHS ===
BASE_PATH = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_PATH, "best.pt")

OUTPUT_FOLDER = os.path.join(BASE_PATH, "results")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === LOAD YOLO MODEL ===
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    messagebox.showerror("Model Error", f"Failed to load YOLO model:\n{e}")
    sys.exit(1)

# ======== FUNCTIONS ========

def show_main_menu():
    """Return to main menu"""
    for widget in root.winfo_children():
        widget.destroy()
    build_main_menu()


def show_image_window(img_path):
    """Display processed image"""
    for widget in root.winfo_children():
        widget.destroy()

    tk.Label(root, text="Processing result", font=("Arial", 14), bg="#f3f3f3").pack(pady=10)

    img = Image.open(img_path)
    img = img.resize((500, 400))
    tk_img = ImageTk.PhotoImage(img)

    img_label = tk.Label(root, image=tk_img)
    img_label.image = tk_img
    img_label.pack(pady=10)

    tk.Button(root, text="⬅ Back to Menu", font=("Arial", 12), command=show_main_menu).pack(pady=10)


def process_image():
    img_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Images", "*.jpg;*.jpeg;*.png")]
    )
    if not img_path:
        return

    results = model.predict(source=img_path, save=True, conf=0.25,
                            project=OUTPUT_FOLDER, name="single_image")

    result_folder = os.path.join(OUTPUT_FOLDER, "single_image")
    saved_files = [f for f in os.listdir(result_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not saved_files:
        messagebox.showerror("Error", "Failed to save the result!")
        return

    result_img_path = os.path.join(result_folder, saved_files[-1])
    show_image_window(result_img_path)


def process_folder():
    folder = filedialog.askdirectory(title="Select a folder with images")
    if not folder:
        return

    images = [os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        messagebox.showwarning("Warning", "No images found in the folder!")
        return

    for img_path in images:
        model.predict(source=img_path, save=True, conf=0.25,
                    project=OUTPUT_FOLDER, name="folder_images")

    messagebox.showinfo("Done", f"Results saved to '{OUTPUT_FOLDER}/folder_images'")


def process_video():
    video_path = filedialog.askopenfilename(
        title="Select a video",
        filetypes=[("Video files", "*.mp4;*.avi;*.mov")]
    )
    if not video_path:
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Failed to open the video!")
        return

    # === Video display window ===
    for widget in root.winfo_children():
        widget.destroy()
    root.title("Video Processing")

    lbl_video = tk.Label(root)
    lbl_video.pack()

    btn_back = tk.Button(root, text="⬅ Back to Menu", font=("Arial", 12),
                        command=lambda: stop_video(cap))
    btn_back.pack(pady=10)

    output_video_path = os.path.join(OUTPUT_FOLDER, "detected_video.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    def update_frame():
        ret, frame = cap.read()
        if not ret:
            cap.release()
            out.release()
            messagebox.showinfo("Done", f"Video saved to: {output_video_path}")
            show_main_menu()
            return

        results = model.predict(source=frame, conf=0.25, verbose=False)
        annotated = results[0].plot()
        out.write(annotated)

        img_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil = img_pil.resize((640, 360))
        tk_img = ImageTk.PhotoImage(img_pil)
        lbl_video.imgtk = tk_img
        lbl_video.configure(image=tk_img)

        lbl_video.after(10, update_frame)

    def stop_video(capture):
        capture.release()
        out.release()
        show_main_menu()

    update_frame()


# ======== MAIN MENU ========

def build_main_menu():
    root.title("Face & Eye Detection")
    root.geometry("600x500")
    root.configure(bg="#f3f3f3")

    tk.Label(root, text="Choose processing mode", font=("Arial", 16), bg="#f3f3f3").pack(pady=30)
    tk.Button(root, text="📸 Image", font=("Arial", 13), width=20, command=process_image).pack(pady=10)
    tk.Button(root, text="📂 Folder with images", font=("Arial", 13), width=20, command=process_folder).pack(pady=10)
    tk.Button(root, text="🎥 Video", font=("Arial", 13), width=20, command=process_video).pack(pady=10)


# ======== RUN APP ========

root = tk.Tk()
build_main_menu()
root.mainloop()
