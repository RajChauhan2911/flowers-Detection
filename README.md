# 🌸 Flower Detection using SVM and Computer Vision

## 📌 Overview

This project is a Machine Learning and Computer Vision based flower detection system developed using Python, OpenCV, and Scikit-learn.

The system detects flowers in images using:
- HOG (Histogram of Oriented Gradients)
- HSV Color Histograms
- Sliding Window Detection
- Support Vector Machine (SVM)
- Non-Maximum Suppression (NMS)

The project can:
- Train a flower detection model
- Detect flowers in custom images
- Draw bounding boxes around detected flowers
- Save detection results automatically

The model was trained on a custom annotated flower dataset containing approximately 2500–3000 flower images across 5 flower categories.

---

# 🧠 Features

✅ Flower object detection  
✅ Multi-class flower detection  
✅ 5 flower categories  
✅ Large annotated custom dataset  
✅ Bounding box generation  
✅ HOG feature extraction  
✅ HSV color histogram features  
✅ Sliding window object detection  
✅ Multi-scale image detection  
✅ Non-Maximum Suppression (NMS)  
✅ Data augmentation  
✅ SVM-based classification  
✅ GUI image picker using Tkinter  
✅ Detection result visualization  

---

# 🛠️ Technologies Used

- Python
- OpenCV
- NumPy
- Scikit-learn
- Scikit-image
- Tkinter
- Joblib
- PIL

---

# 📂 Project Structure

```text
flowers-detection/
│
├── flowers_annotated/        # Dataset folder
│   ├── rose/
│   ├── sunflower/
│   ├── daisy/
│   ├── tulip/
│   └── dandelion/
│
├── flower_detection.py       # Main training and detection script
├── pick_and_detect.py        # GUI image picker
├── flower_svm_model.pkl      # Trained SVM model
├── detection_result.jpg      # Output detection image
├── README.md
└── requirements.txt
```

---

# 📊 Dataset

The dataset contains annotated flower images organized into class folders.

## Flower Classes

- 🌹 Rose
- 🌻 Sunflower
- 🌼 Daisy
- 🌷 Tulip
- 🌿 Dandelion

Each class contains approximately:

```text
500–600 images
```

Total dataset size is approximately:

```text
2500–3000 images
```

Dataset structure:

```text
flowers_annotated/
│
├── rose/
├── sunflower/
├── daisy/
├── tulip/
└── dandelion/
```

---

# ⚙️ Working Principle

## 1️⃣ Feature Extraction

The system extracts:
- HOG features for flower shape detection
- HSV color histograms for flower color information

---

## 2️⃣ Data Augmentation

Training images are augmented using:
- Horizontal flipping
- Rotation
- Brightness adjustment

to improve model robustness and generalization.

---

## 3️⃣ Model Training

The project trains:
- Linear SVM classifier
- KNN image classifier

for flower recognition and object detection.

---

## 4️⃣ Sliding Window Detection

The image is scanned using:
- Multi-scale sliding windows

to locate flowers at different positions and sizes.

---

## 5️⃣ Non-Maximum Suppression (NMS)

Overlapping detections are removed to improve detection accuracy and reduce duplicate bounding boxes.

---

# 🎯 Detection Pipeline

```text
Input Image
     ↓
Feature Extraction
     ↓
Sliding Window Search
     ↓
SVM Classification
     ↓
Confidence Filtering
     ↓
Non-Maximum Suppression
     ↓
Final Flower Detection
```

---

# 🖼️ Output Example

The system generates:
- Original image
- Detected flower image
- Red bounding boxes
- Confidence scores

Output file:

```text
detection_result.jpg
```

---

# ▶️ How to Run

## Clone Repository

```bash
git clone https://github.com/your-username/flowers-detection.git
```

---

# 📦 Install Dependencies

```bash
pip install opencv-python numpy scikit-learn scikit-image tqdm pillow joblib
```

---

# 🚀 Train the Model

```bash
python flower_detection.py --dataset flowers_annotated --train
```

---

# 🔍 Detect Flowers in Image

```bash
python flower_detection.py --image flower.jpg
```

---

# 🖱️ GUI Image Picker

```bash
python pick_and_detect.py
```

This opens a file picker window to select an image for flower detection.

---

# 📈 Model Capabilities

- Multi-class flower recognition
- Bounding box localization
- Confidence-based filtering
- Detection visualization
- Automatic output saving
- Multi-scale detection support

---

# 📚 Concepts Used

- Computer Vision
- Object Detection
- Feature Engineering
- HOG Features
- HSV Color Histograms
- Sliding Window Detection
- Support Vector Machine (SVM)
- Non-Maximum Suppression (NMS)
- Machine Learning Pipelines


# 📈 Learning Outcomes

This project helped in understanding:

- Traditional object detection techniques
- Image feature extraction
- Machine Learning pipelines
- Sliding window detection
- Bounding box generation
- OpenCV image processing
- SVM classifiers


---

# 👨‍💻 Author

Raj Chauhan

---

# ⭐ GitHub

If you found this project useful, consider giving it a ⭐ on GitHub.
