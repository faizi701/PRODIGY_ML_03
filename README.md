# 🤖 Hand Gesture Recognition with CNN (LeapGestRecog Dataset)

This project trains a Convolutional Neural Network (CNN) from scratch to classify 10 different hand gestures using the LeapGestRecog dataset. The model achieves over **99.87% accuracy** and is also converted to **TensorFlow Lite** for lightweight deployment.



## 📌 Dataset
- **Source**: [LeapGestRecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)
- **Classes**:  
  `['c', 'down', 'fist', 'fist_moved', 'index', 'l', 'ok', 'palm', 'palm_moved', 'thumb']`
- **Image Size**: 128x128 (grayscale)

---

🧠 Model Details
- **Architecture**: CNN (custom built using Keras)
- **Accuracy**: 99.87% on test set
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Conversion**: TensorFlow Lite (TFLite) model available



📁 Project Structure

├── models/
│ ├── leap_gesture_model.h5 # Trained model
│ ├── leap_gesture_model.tflite # TFLite model for deployment
│ └── label_map.npy # Saved label mapping
├── src/
│ └── main.py # Full training + testing code
├── README.md
├── requirements.txt




## 🚀 How to Run
ss
```bash
git clone https://github.com/your-username/gesture-recognition.git
cd gesture-recognition
pip install -r requirements.txt
python src/main.py
```

TensorFlow Lite Model
The model is converted to TFLite for mobile/edge use.
Input:  (None, 128, 128, 1)
Output: (None, 10)
