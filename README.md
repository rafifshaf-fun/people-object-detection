# 🚗 Vehicle & People Surveillance CV System

A multi-stage computer vision pipeline that detects cars and people,
then zooms in to identify license plates and faces.

## 🔍 Demo
![Sample Output](samples/Sample-output-car.png) 

![Sample Face](samples/Sample-output-face.png)

## 🏗️ Architecture
Input Frame → [Primary Detector] → Car ROI → [Plate Detector] → Plate Crop
                                → Person ROI → [Face Detector] → Face Crop

## 📊 Model Performance (2 epochs)
| Model | mAP50 | mAP50-95 |
|---|---|---|
| Primary (Car+Person) | 0.807 | 0.479 |
| License Plate | 0.83 | 0.539 |
| Face Detector | 0.864 | 0.57 |

## 🗂️ Datasets
- Car Object Detection (Kaggle)
- People Detection (Kaggle)
- Indonesian Vehicle License Plate (Kaggle)
- Face Detection Dataset (Kaggle)

## 🚀 Run It Yourself
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/romeoshaffun/people-and-car-object-detection)

## 🛠️ Tech Stack
- YOLOv8 (Ultralytics)
- OpenCV
- PyTorch
- Python 3.12
