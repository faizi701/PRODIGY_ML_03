import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# File paths
MODEL_PATH = r"C:\Users\DELL\OneDrive\Desktop\Hand Gestures\leap_gesture_model.tflite"
LABEL_MAP_PATH = r"C:\Users\DELL\OneDrive\Desktop\Hand Gestures\label_map.npy"

# Load the TFLite model and label map
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

try:
    label_map = np.load(LABEL_MAP_PATH, allow_pickle=True).item()
    st.success("Label map loaded successfully")
except Exception as e:
    st.error(f"Error loading label map: {e}")
    st.stop()

# Streamlit app
st.title("Hand Gesture Recognition")
st.write("Use your webcam for real-time gesture recognition or upload an image to predict a hand gesture.")

# Webcam-based real-time inference
class GestureTransformer(VideoTransformerBase):
    def __init__(self):
        self.predicted_gesture = "No prediction"
        self.confidence = 0.0
        self.frame_count = 0

    def transform(self, frame):
        try:
            # Process every 5th frame to improve performance
            self.frame_count += 1
            if self.frame_count % 5 != 0:
                return frame.to_ndarray(format="bgr24")

            img = frame.to_ndarray(format="bgr24")
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Enhance contrast if needed
            img_gray = cv2.equalizeHist(img_gray)
            img_resized = cv2.resize(img_gray, (128, 128)) / 255.0
            img_input = img_resized.reshape(1, 128, 128, 1).astype(np.float32)

            # Run inference
            interpreter.set_tensor(input_details[0]['index'], img_input)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])
            predicted_label = np.argmax(prediction)
            confidence = prediction[0][predicted_label]

            # Update prediction with confidence threshold
            if confidence > 0.7:
                self.predicted_gesture = label_map[predicted_label]
                self.confidence = confidence
            else:
                self.predicted_gesture = "Uncertain"
                self.confidence = confidence

            # Display prediction on the frame
            cv2.putText(img, f"{self.predicted_gesture} ({self.confidence:.4f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display probabilities in sidebar
            with st.sidebar:
                st.write("Prediction Probabilities:")
                for i, prob in enumerate(prediction[0]):
                    st.write(f"{label_map[i]}: {prob:.4f}")

            return img
        except Exception as e:
            st.error(f"Error processing frame: {e}")
            return frame.to_ndarray(format="bgr24")

st.header("Real-Time Webcam Testing")
try:
    webrtc_streamer(
        key="gesture_recognition",
        video_transformer_factory=GestureTransformer,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
    st.write("Webcam feed should display above. If not, check browser permissions or webcam connection.")
except Exception as e:
    st.error(f"Webcam initialization error: {e}")
    st.write("Troubleshooting steps: Ensure webcam is working (tested with OpenCV), grant browser permissions, and try a different browser (e.g., Chrome, Firefox).")

# Image upload option
st.header("Upload Image for Testing")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    try:
        # Read and preprocess the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        img = cv2.equalizeHist(img)  # Enhance contrast
        img = cv2.resize(img, (128, 128)) / 255.0
        img = img.reshape(1, 128, 128, 1).astype(np.float32)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        predicted_label = np.argmax(prediction)
        predicted_gesture = label_map[predicted_label]
        confidence = prediction[0][predicted_label]

        # Display results
        st.image(img.reshape(128, 128), caption='Uploaded Image', use_column_width=True)
        if confidence > 0.7:
            st.write(f"Predicted Gesture: **{predicted_gesture}**")
            st.write(f"Prediction Confidence: {confidence:.4f}")
        else:
            st.write("Prediction: **Uncertain**")
            st.write(f"Confidence: {confidence:.4f}")

        # Display probabilities
        st.write("Prediction Probabilities:")
        for i, prob in enumerate(prediction[0]):
            st.write(f"{label_map[i]}: {prob:.4f}")
    except Exception as e:
        st.error(f"Error processing uploaded image: {e}")
