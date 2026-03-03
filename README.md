# 🚗 Vehicle & People Surveillance CV System

A multi-stage computer vision pipeline that detects cars and people,
then zooms in to identify license plates and faces.

## 🔍 Demo
![Sample Output](assets/sample_output_car.jpg)

## 🏗️ Architecture
Input Frame → [Primary Detector] → Car ROI → [Plate Detector] → Plate Crop
                                → Person ROI → [Face Detector] → Face Crop

## 📊 Model Performance
| Model | mAP50 | mAP50-95 |
|---|---|---|
| Primary (Car+Person) | X.XXX | X.XXX |
| License Plate | X.XXX | X.XXX |
| Face Detector | 0.812 | 0.513 |

## 🗂️ Datasets
- Car Object Detection (Kaggle)
- People Detection (Kaggle)
- Indonesian Vehicle License Plate (Kaggle)
- Face Detection Dataset (Kaggle)

## 🚀 Run It Yourself
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](YOUR_KAGGLE_NOTEBOOK_URL)

## 🛠️ Tech Stack
- YOLOv8 (Ultralytics)
- OpenCV
- PyTorch
- Python 3.12
