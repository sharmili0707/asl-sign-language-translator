import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pyttsx3
import time

# Load model and labels
model = load_model('sign_model.h5')
label_classes = np.load('label_classes.npy', allow_pickle=True)

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Text-to-Speech setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # Use female voice if available

# Webcam setup
cap = cv2.VideoCapture(0)
print("📷 Webcam started... Press 'Q' to quit")

# Track prediction history
current_text = ""
last_letter = ""
last_time = time.time()
COOLDOWN = 1.2  # Delay to avoid repeated predictions

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])

            if len(data) == 63:
                prediction = model.predict(np.array([data]), verbose=0)
                class_id = np.argmax(prediction)
                confidence = np.max(prediction)
                predicted_label = str(label_classes[class_id])

                if predicted_label != last_letter and (time.time() - last_time) > COOLDOWN:
                    current_text += predicted_label
                    last_letter = predicted_label
                    last_time = time.time()
                    print(f"📣 Letter Added: {predicted_label} → {current_text}")

                cv2.putText(frame, f'{predicted_label} ({confidence:.2f})',
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        cv2.putText(frame, "No hand detected", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)

    # Show full predicted word
    cv2.putText(frame, f'📝 {current_text}', (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 0), 3)

    cv2.imshow('🖐️ Sign-to-Text Translator - Press Q to Quit', frame)

    key = cv2.waitKey(1) & 0xFF

    # Add space
    if key == ord(' '):
        current_text += ' '
        print("🟦 Space added")

    # Backspace (DEL key or 'b')
    elif key == 8 or key == ord('b'):
        current_text = current_text[:-1]
        print("🔙 Removed last letter")

    # Clear all text
    elif key == ord('c'):
        current_text = ""
        print("🧹 Text cleared")

    # Speak the text (Enter key)
    elif key == 13:  # Enter key
        if current_text.strip():
            try:
                engine.stop()  # stop if engine is already speaking
                print(f"🗣️ Speaking: {current_text}")
                engine.say(current_text)
                engine.runAndWait()
            except Exception as e:
                print("❌ TTS Error:", e)

    # Quit
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
