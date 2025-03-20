# AutoSeatAdjust-EmbeddedAI

**An AI-based driver seat positioning system that uses camera-based eye detection and XGBoost regression to automatically adjust seat distance and height on embedded devices.**

## Table of Contents
1. [Overview](#overview)  
2. [Project Structure](#project-structure)  
3. [Dataset](#dataset)  
4. [Installation & Requirements](#installation--requirements)  
5. [Training](#training)  
6. [Inference](#inference)  
7. [Eye Detection (Optional)](#eye-detection-optional)  
8. [Future Improvements](#future-improvements)  
9. [License](#license)

---

## Overview
Many vehicles today allow manual seat adjustments for comfort and safety. This repository demonstrates an **embedded AI approach** that automatically positions the driver’s seat (distance and height) based on the driver’s **eye coordinates** in a real-time camera feed.  

### Key Features
- **Eye coordinate input** from live camera or a stored dataset.  
- **XGBoost Regression** models that learn to map `(eye_x, eye_y)` → `(seat_x, seat_y)`.  
- **Modular design** for training, inference, and (optional) eye-detection integration.  
- **Embeddable** in systems such as Raspberry Pi, NVIDIA Jetson, or similar edge devices.

---

## Project Structure

```
AutoSeatAdjust-EmbeddedAI/
├── data/
│   └── seat_adjustment_data.csv    # CSV dataset of eye coords and seat coords
├── models/
│   ├── seat_adjustment_xgboost_x.json  # Trained model for seat_x
│   └── seat_adjustment_xgboost_y.json  # Trained model for seat_y
├── src/
│   ├── train_seat_adjuster.py      # Script to train XGBoost regressors
│   ├── infer_seat_adjuster.py      # Inference script for seat adjustment
│   └── eye_detection.py            # (Optional) Eye detection using OpenCV
├── README.md                       # This README
└── requirements.txt                # Python dependencies
```

- **data/**: Contains your dataset (CSV file, images, or other data).  
- **models/**: Stores trained models for seat adjustment.  
- **src/**: Core scripts for training, inference, and optional eye detection.  

---

## Dataset
You need a CSV file that maps **eye coordinates** to **seat coordinates**, for example:

| eye_x | eye_y | seat_x | seat_y |
|-------|-------|--------|--------|
| 150.2 | 75.0  | 30.0   | 15.2   |
| 140.0 | 70.4  | 28.0   | 17.0   |
| ...   | ...   | ...    | ...    |

- **eye_x, eye_y**: 2D coordinates of the driver’s eye position in the camera frame.  
- **seat_x, seat_y**: Desired seat adjustment values (distance & height).

Store the CSV in `data/seat_adjustment_data.csv` (or update paths accordingly).

---

## Installation & Requirements

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Embedded-Vision-AI/AutoSeatAdjust-EmbeddedAI.git
   cd AutoSeatAdjust-EmbeddedAI
   ```

2. **Install required Python packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **GPU Support (Optional)**:  
   - If you want to train on a GPU, ensure you have the required [CUDA](https://developer.nvidia.com/cuda-zone) drivers and libraries installed.  
   - You might need to install `xgboost` with GPU support specifically.

---

## Training

**Script**: `src/train_seat_adjuster.py`

1. **Prepare your CSV** with columns: `eye_x`, `eye_y`, `seat_x`, `seat_y`.  
2. **Update `csv_path`** in the script if necessary (defaults to `data/seat_adjustment_data.csv`).  
3. **Run the training script**:
   ```bash
   cd src
   python train_seat_adjuster.py
   ```
4. **Outputs**:
   - Two model files (by default):  
     - `models/seat_adjustment_xgboost_x.json`  
     - `models/seat_adjustment_xgboost_y.json`

The script trains two separate XGBoost regressors to predict `seat_x` and `seat_y` from `(eye_x, eye_y)`.

---

## Inference

**Script**: `src/infer_seat_adjuster.py`

1. **Verify model paths** in the script:
   ```python
   model_x_path = "models/seat_adjustment_xgboost_x.json"
   model_y_path = "models/seat_adjustment_xgboost_y.json"
   ```
2. **(Optional) Provide real eye coordinates**.  
   - In the example, `get_current_eye_coordinates()` returns dummy `(eye_x, eye_y)` values.  
   - Integrate with an actual eye detection function or any other method to get real-time driver eye position.

3. **Run the inference script**:
   ```bash
   python infer_seat_adjuster.py
   ```
4. **Predicted seat coordinates** are printed to console. In a real scenario, these would be sent to your seat control module.

---

## Eye Detection (Optional)

If you need to **detect eyes** in real time:

1. **Install OpenCV** (already in `requirements.txt` if you are using it).
2. **Download Haar Cascade for Eyes** (for instance, `haarcascade_eye.xml` from OpenCV’s GitHub).
3. **Use the `eye_detection.py` script**:
   ```bash
   python eye_detection.py
   ```
   This script will open your webcam, detect eyes, and display bounding boxes. You can adjust the code to compute the eye center `(eye_x, eye_y)` and feed it to the inference script.

---

## Future Improvements

- **Multi-output XGBoost**: Currently, we train two separate regressors. Explore multi-output methods or other frameworks supporting direct 2D regression.  
- **Improved Eye Tracking**: Instead of Haar cascades, use a more robust model (Dlib, Mediapipe, or a custom CNN).  
- **Calibration**: Provide user calibration options for each driver’s preference.  
- **Edge Deployment**: Optimize model size and performance using quantization or pruning for real-time embedded inference.

---

## License
This project is offered under the [MIT License](https://opensource.org/licenses/MIT). Feel free to modify and adapt for your use case.  

---

## References

1. **XGBoost** – [Official GitHub](https://github.com/dmlc/xgboost), [Documentation](https://xgboost.readthedocs.io/)  
2. **scikit-image** – [Docs](https://scikit-image.org/docs/stable/)  
3. **NVIDIA Jetson** – [Developer Site](https://developer.nvidia.com/embedded-computing)
