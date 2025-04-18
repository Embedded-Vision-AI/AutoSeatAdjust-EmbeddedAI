# AutoSeatAdjust-EmbeddedAI

**An AI-powered adaptive seat positioning system** that predicts ideal longitudinal seat distance and seat angle from the driver’s height. Designed for embedded use on systems like Raspberry Pi or NVIDIA Jetson, this project trains and runs lightweight regression models with user feedback support.

---

## 📌 Features
- **One input, two predictions**: Maps driver **Height (cm)** → **True Longitudinal Distance** and **Seat Angle**
- **XGBoost-based regressors** with support for real-time inference
- **User feedback loop**: Adjustments can be saved and used to retrain the model instantly
- **Excel dataset support** with safe appends via openpyxl
- **Cross-platform** (Mac, Windows, Ubuntu)

---

## 🧠 How It Works
1. Train two separate **XGBoost** regressors using historical seat fitment data
2. Save the trained models in a single `.pth` file using `joblib`
3. During inference, prompt the user for **height**
4. Predict **seat distance** and **seat angle**
5. If the prediction is wrong, collect corrected values, save them back to the Excel file, and retrain the models

---

## 💾 Dataset Format

Your Excel file (`Extrapolated_Participant_Data.xlsx`) should contain the following columns:

| Height (cm) | True Long. Distance (cm) | True Seat Angle (from vertical) |
|-------------|--------------------------|----------------------------------|
| 172         | 85.94                    | 14.75                           |
| ...         | ...                      | ...                             |

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/YourUsername/AutoSeatAdjust-EmbeddedAI.git
cd AutoSeatAdjust-EmbeddedAI
```

### 2. Set Up a Virtual Environment

#### macOS / Ubuntu
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r dependencies.txt
```

#### Windows (PowerShell)
```powershell
py -m venv venv
venv\Scripts\Activate.ps1
pip install -r dependencies.txt
```

---

## 🧪 Training the Model

```bash
python DTreeClassifier.py
```

- Loads the Excel dataset
- Trains two regressors
- Saves them to `models/pred_model.pth`

---

## 🤖 Running Inference with Feedback

```bash
python infer_height.py
```

- Input: Driver height
- Output: Predicted seat position and angle
- If the prediction is off, enter your corrected values and the system will:
  - Append them to the dataset
  - Retrain the model with the new datapoint

---

## 🔄 Real-Time Updates

Any new data collected during use is:
- Stored safely in the same Excel file
- Used for immediate retraining
- Persisted via `joblib` in a `.pth` file

---

## 🚀 Future Work
- Add GUI for in-car use
- Replace height input with real-time eye-tracking using OpenCV
- Deploy on Jetson Orin or Raspberry Pi with model quantization
- Integrate calibration profiles per driver

---

## 📜 License
MIT License – use, modify, or deploy freely.

---

## 🙌 Acknowledgements
- [XGBoost](https://xgboost.readthedocs.io/)
- [scikit-learn](https://scikit-learn.org/)
- [openpyxl](https://openpyxl.readthedocs.io/)

---

Need help? Open an issue or start a discussion! 💬
