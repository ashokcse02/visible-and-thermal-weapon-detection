from ultralytics import YOLO  # Fixed import (was typo in your code)
import tkinter as tk
from tkinter import filedialog
import cv2
import torch
import os
import matplotlib.pyplot as plt
from pathlib import Path

MODEL_PATH = "best.pt"
DATA_YAML = "data.yaml"

class WeaponSystem:

    def __init__(self):
        print("🔄 Initializing System...")
        self.model = None
        if self.model_exists():
            self.model = YOLO(MODEL_PATH)
            print("✅ Model loaded successfully.")
        else:
            print("⚠️ No trained model found. Use option 1 to train.")
        print()

    def model_exists(self):
        try:
            model = YOLO(MODEL_PATH)
            # Try a dummy prediction to verify model works
            import numpy as np
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            model.predict(dummy, verbose=False)
            return True
        except Exception as e:
            print(f"Model validation error: {e}")
            return False

    # ================= TRAINING WITH REGULARIZATION =================
    def train_model(self):
        print("🚀 Starting Training with Regularization...")
        
        # Check if GPU available
        device = 0 if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Load pretrained model (not starting from scratch)
        model = YOLO("yolov8n.pt")
        
        # Enhanced training parameters to reduce overfitting
        model.train(
            data=DATA_YAML,
            epochs=50,              # More epochs but with regularization
            imgsz=640,                # Original size, not reduced
            batch=8 if device == 0 else 4,  # Smaller batches for CPU
            device=device,
            
            # Data augmentation (crucial for generalization)
            hsv_h=0.015,              # Hue augmentation
            hsv_s=0.7,                 # Saturation augmentation
            hsv_v=0.4,                 # Value augmentation
            degrees=10.0,              # Small rotation (thermal images can be rotated)
            translate=0.1,              # Translation
            scale=0.5,                  # Scale augmentation
            shear=2.0,                  # Shear augmentation
            perspective=0.0005,         # Perspective
            flipud=0.1,                 # Vertical flip (thermal images upside down)
            fliplr=0.5,                  # Horizontal flip
            
            # Mosaic and mixup for better generalization
            mosaic=1.0,                  # Mosaic augmentation
            mixup=0.2,                    # Mixup augmentation
            copy_paste=0.1,                # Copy-paste augmentation
            
            # Regularization
            dropout=0.1,                    # Dropout
            weight_decay=0.0005,             # L2 regularization
            
            # Learning rate schedule
            lr0=0.01,                         # Initial learning rate
            lrf=0.01,                          # Final learning rate factor
            momentum=0.937,                      # Momentum
            warmup_epochs=3,                       # Warmup epochs
            warmup_momentum=0.8,                      # Warmup momentum
            warmup_bias_lr=0.1,                         # Warmup bias learning rate
            
            # Other settings
            workers=4 if device == 0 else 2,             # Data loading workers
            project='thermal_weapon_training',             # Project name
            name='experiment1',                               # Experiment name
            exist_ok=True,                                       # Overwrite existing
            pretrained=True,                                      # Use pretrained
            optimizer='SGD',                                        # SGD works well for thermal
            verbose=True,                                             # Show training progress
            seed=42,                                                    # Reproducibility
            deterministic=True,                                           # Deterministic
            single_cls=False,                                              # Multi-class if needed
            rect=False,                                                      # Rectangular training
            cos_lr=True,                                                     # Cosine learning rate
            patience=20,                                                       # Early stopping patience
            save=True,                                                          # Save checkpoints
            save_period=10,                                                       # Save every 10 epochs
            cache=False,                                                            # Don't cache images
        )
        
        print("✅ Training Complete.")
        
        # Test on validation set
        self.validate_model()

    # ================= VALIDATION =================
    def validate_model(self):
        """Validate model on validation set"""
        if self.model is None:
            print("❌ No model to validate.")
            return
        
        print("\n📊 Running Validation...")
        metrics = self.model.val(data=DATA_YAML, device='cpu' if not torch.cuda.is_available() else 0)
        
        print(f"\n📈 Validation Results:")
        print(f"   mAP50: {metrics.box.map50:.4f}")
        print(f"   mAP50-95: {metrics.box.map:.4f}")
        print(f"   Precision: {metrics.box.mp:.4f}")
        print(f"   Recall: {metrics.box.mr:.4f}")
        
        return metrics

    # ================= TEST WITH ENSEMBLE =================
    def test_model(self):
        if self.model is None:
            print("❌ Model not found. Train first.")
            return

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        file_path = filedialog.askopenfilename(
            title="Select Image to Test",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )

        root.destroy()

        if not file_path:
            return

        # Test with different confidence thresholds
        conf_thresholds = [0.1, 0.15, 0.2, 0.25]
        all_results = []
        
        print(f"\n🔍 Testing image: {Path(file_path).name}")
        print(f"Image size: {cv2.imread(file_path).shape[:2]}")
        
        for conf in conf_thresholds:
            results = self.model.predict(
                source=file_path,
                conf=conf,
                iou=0.45,
                imgsz=640,
                device='cpu' if not torch.cuda.is_available() else 0,
                augment=True,  # Test-time augmentation
                half=False,
                retina_masks=False,
                max_det=10,
            )
            all_results.append(results[0])
            
            # Print detections for each confidence
            boxes = results[0].boxes
            if len(boxes) > 0:
                print(f"\n📌 Detections at conf={conf}:")
                for i, box in enumerate(boxes):
                    cls = int(box.cls[0])
                    conf_score = float(box.conf[0])
                    name = results[0].names[cls]
                    print(f"   {i+1}: {name} ({conf_score:.3f})")
        
        # Use the best result (highest confidence with detections)
        best_result = None
        best_conf = 0
        for i, result in enumerate(all_results):
            if len(result.boxes) > 0:
                max_box_conf = max([float(b.conf[0]) for b in result.boxes])
                if max_box_conf > best_conf:
                    best_conf = max_box_conf
                    best_result = result
        
        if best_result is None:
            best_result = all_results[0]  # Use first if no detections
        
        # Plot and save results
        annotated = best_result.plot()
        
        # Add confidence threshold info to image
        cv2.putText(annotated, f"Best conf threshold: {best_conf:.2f}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show image
        cv2.imshow("Detection Result (Press any key to close)", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save result
        output_dir = "test_results"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"result_{Path(file_path).stem}.jpg")
        cv2.imwrite(output_path, annotated)
        print(f"\n💾 Result saved to: {output_path}")

    # ================= BATCH TEST ON MULTIPLE IMAGES =================
    def batch_test(self):
        """Test on multiple images from a folder"""
        if self.model is None:
            print("❌ Model not found. Train first.")
            return
        
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        
        folder_path = filedialog.askdirectory(title="Select Folder with Test Images")
        root.destroy()
        
        if not folder_path:
            return
        
        # Get all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(folder_path).glob(f"*{ext}"))
            image_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))
        
        if not image_files:
            print("❌ No images found in selected folder.")
            return
        
        print(f"\n📁 Found {len(image_files)} images to test.")
        
        # Create output directory
        output_dir = os.path.join(folder_path, "detection_results")
        os.makedirs(output_dir, exist_ok=True)
        
        total_detections = 0
        for img_path in image_files:
            print(f"\n🔍 Processing: {img_path.name}")
            
            results = self.model.predict(
                source=str(img_path),
                conf=0.15,  # Lower confidence for batch testing
                iou=0.45,
                imgsz=640,
                device='cpu' if not torch.cuda.is_available() else 0,
                augment=True,
                max_det=5,
            )
            
            # Save result
            annotated = results[0].plot()
            output_path = os.path.join(output_dir, f"det_{img_path.name}")
            cv2.imwrite(output_path, annotated)
            
            # Count detections
            det_count = len(results[0].boxes)
            total_detections += det_count
            print(f"   Detections: {det_count}")
        
        print(f"\n✅ Batch testing complete!")
        print(f"   Total detections: {total_detections}")
        print(f"   Results saved to: {output_dir}")

    # ================= EXPORT MODEL =================
    def export_model(self):
        """Export model to different formats"""
        if self.model is None:
            print("❌ Model not found.")
            return
        
        print("\n📦 Exporting model...")
        
        # Export to ONNX
        onnx_path = self.model.export(format="onnx", imgsz=640)
        print(f"   ONNX model saved to: {onnx_path}")
        
        # Export to TorchScript
        torchscript_path = self.model.export(format="torchscript", imgsz=640)
        print(f"   TorchScript model saved to: {torchscript_path}")
        
        print("✅ Export complete!")

    # ================= MENU =================
    def menu(self):
        while True:
            print("\n" + "="*50)
            print("   WEAPON DETECTION SYSTEM - Thermal Images")
            print("="*50)
            print("1: Train New Model (with regularization)")
            print("2: Validate Model")
            print("3: Test Single Image")
            print("4: Batch Test (multiple images)")
            print("5: Export Model (ONNX/TorchScript)")
            print("6: Exit")
            print("-"*50)
            
            choice = input(">> ").strip()

            if choice == "1":
                self.train_model()
            elif choice == "2":
                if self.model:
                    self.validate_model()
                else:
                    print("❌ No model to validate. Train first.")
            elif choice == "3":
                if self.model:
                    self.test_model()
                else:
                    print("❌ No model to test. Train first.")
            elif choice == "4":
                if self.model:
                    self.batch_test()
                else:
                    print("❌ No model to test. Train first.")
            elif choice == "5":
                if self.model:
                    self.export_model()
                else:
                    print("❌ No model to export. Train first.")
            elif choice == "6":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    app = WeaponSystem()
    app.menu()