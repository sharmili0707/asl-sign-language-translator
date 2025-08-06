import os
import cv2
import mediapipe as mp
import csv

print("🔍 Starting landmark extraction...")

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Point to the dataset folder — change if needed
base_path = 'asl_alphabet_train'

# Debug: Check what's inside the folder
print("🗂️ Contents of base_path:", os.listdir(base_path))

output_file = 'asl_landmarks.csv'

with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['label'] + [f'{i}_{a}' for i in range(21) for a in ['x', 'y', 'z']])

    total_saved = 0

    for label in os.listdir(base_path):
        label_path = os.path.join(base_path, label)

        if not os.path.isdir(label_path):
            print(f"⛔ Skipping {label_path} (not a folder)")
            continue

        print(f"🔤 Processing letter: {label}")
        image_count = 0
        not_detected = 0

        try:
            images = os.listdir(label_path)
            print(f"📸 Found {len(images)} images in '{label}'")
        except Exception as e:
            print(f"❌ Error reading {label_path}: {e}")
            continue

        for img_name in images:
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f"⚠️ Could not load image: {img_path}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img_rgb)

            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                row = [label] + [coord for lm in hand.landmark for coord in (lm.x, lm.y, lm.z)]
                writer.writerow(row)
                image_count += 1
                total_saved += 1
            else:
                not_detected += 1

            if image_count >= 50:  # limit per letter to avoid long time
                break

        print(f"✅ Saved: {image_count} | ❌ Not Detected: {not_detected} for {label}")

print(f"\n🎉 Extraction complete! Total samples saved: {total_saved}")


