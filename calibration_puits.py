import cv2
import numpy as np
import json
from picamera2 import Picamera2
import time
import RPi.GPIO as GPIO

# === Calibration manuelle (résultats précédents) ===
K = np.array([[5282.32084, 0.0, 849.301187],
              [0.0, 5279.52624, 295.988566],
              [0.0, 0.0, 1.0]])
dist = np.array([[0.729973676, 8.28433969, -0.0403485395, -0.00650735379, -295.103845]])

# === GPIO : LED sur GPIO 18 ===
GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)
GPIO.output(18, GPIO.HIGH)
print("[INFO] LED allumée (GPIO 18)")

# === Initialisation caméra ===
picam2 = Picamera2()
sensor_width, sensor_height = picam2.sensor_resolution
coeff_zoom = 1.0

crop_width = int(sensor_width / coeff_zoom)
crop_height = int(sensor_height / coeff_zoom)
crop_x = (sensor_width - crop_width) // 2
crop_y = (sensor_height - crop_height) // 2

config= picam2.create_video_configuration(
    controls={"ScalerCrop": (crop_x, crop_y, crop_width, crop_height)},
    main={"size": (1280, 720), "format":"RGB888"}
)
picam2.configure(config)
picam2.start()
time.sleep(2)

# === Capture et correction de distorsion ===
print("[INFO] Capture en cours...")
image_distorted = picam2.capture_array()
h, w = image_distorted.shape[:2]
new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, new_K, (w, h), cv2.CV_32FC1)
image = cv2.remap(image_distorted, mapx, mapy, cv2.INTER_LINEAR)
cv2.imwrite("image_redressee.png", image)

# === Prétraitement : CLAHE + flou ===
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)
blurred = cv2.medianBlur(enhanced, 5)

# === HoughCircles ===
circles = cv2.HoughCircles(
    blurred,
    method=cv2.HOUGH_GRADIENT,
    dp=1.0,
    minDist=120,
    param1=80,
    param2=20,
    minRadius=60,
    maxRadius=90
)

centres_puits = []

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i, (x, y, r) in enumerate(circles[0, :]):
        cv2.circle(image, (x, y), r, (0, 255, 0), 2)
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
        centres_puits.append((int(x), int(y), int(r)))
        print(f"[DEBUG] Puits #{i+1} → (x={x}, y={y}), rayon={r}")
else:
    print("[INFO] Aucun puits détecté.")

# === Sauvegarde ===
with open("centres_puits.json", "w") as f:
    json.dump(centres_puits, f, indent=2)
cv2.imwrite("puits_detectes.png", image)

# === Nettoyage ===
picam2.stop()
GPIO.output(18, GPIO.LOW)
GPIO.cleanup()
print("[INFO] LED éteinte (GPIO 18) et GPIO nettoyé")
print(f"[INFO] {len(centres_puits)} puits détectés.")
