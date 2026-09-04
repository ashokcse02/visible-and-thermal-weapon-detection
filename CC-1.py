from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
import cv2
import torch

MODEL_PATH = "best.pt"   # your trained model
DATA_YAML = "data.yaml"  # dataset yaml (only needed for training)


class WeaponSystem:

    def __init__(self):
        print("🔄 Initializing System...")
        self.model = YOLO(MODEL_PATH) if self.model_exists() else None
        print("✅ Ready.\n")

    def model_exists(self):
        try:
            YOLO(MODEL_PATH)
            return True
        except:
            return False

    # ================= TRAIN =================
    def train_model(self):
        print("🚀 Starting Training...")

        device = 0 if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        model = YOLO("yolov8n.pt")  # lighter for CPU
        model.train(
            data=DATA_YAML,
            epochs=50,
            imgsz=416,
            batch=4,
            device=device
        )

        print("✅ Training Complete.")

    # ================= TEST =================
    def test_model(self):
        if self.model is None:
            print("❌ Model not found. Train first.")
            return

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        file_path = filedialog.askopenfilename(
            title="Select Image to Test",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )

        root.destroy()

        if not file_path:
            return

        # results = self.model.predict(
        #      source=file_path,
        #      conf=0.3,
        #      device="cpu",
        #      imgsz=640,
        #      save=False
        # )
        results = self.model.predict(
             source=file_path,
             conf=0.25,  # lower confidence
             iou=0.5,
             imgsz=640
         )

        annotated = results[0].plot()

        cv2.imshow("Detection Result", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ================= MENU =================
    def menu(self):
        print("----- WEAPON DETECTION SYSTEM -----")
        print("1: Train")
        print("2: Test")
        choice = input(">> ")

        if choice == "1":
            self.train_model()
        elif choice == "2":
            self.test_model()
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    app = WeaponSystem()
    app.menu()