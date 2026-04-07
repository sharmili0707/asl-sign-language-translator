import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib

st.title("🤟 ASL Sign Language Translator")

model = joblib.load("asl_model.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

FRAME_WINDOW = st.image([])
run = st.checkbox("Start Camera")

cap = cv2.VideoCapture(0)

def extract_landmarks(hand_landmarks):
    data = []
    for lm in hand_landmarks.landmark:
        data.extend([lm.x, lm.y, lm.z])
    return np.array(data).reshape(1, -1)

if run:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            landmarks = extract_landmarks(handLms)
            prediction = model.predict(landmarks)

            st.success(f"Predicted Sign: {prediction[0]}")

    FRAME_WINDOW.image(rgb)

cap.release()
