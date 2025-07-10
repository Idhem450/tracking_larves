from picamera2 import Picamera2
import cv2
import time
import os

# === PARAMÈTRES ===
save_dir = "images_calibration"
os.makedirs(save_dir, exist_ok=True)

# === INITIALISATION DE LA CAMÉRA AVEC ZOOM ===
picam2 = Picamera2()
sensor_width, sensor_height = picam2.sensor_resolution
coeff_zoom = 1.2

crop_width = int(sensor_width / coeff_zoom)
crop_height = int(sensor_height / coeff_zoom)
crop_x = (sensor_width - crop_width) // 2
crop_y = (sensor_height - crop_height) // 2

# Configuration avec zoom via ScalerCrop
config = picam2.create_video_configuration(
    controls={"ScalerCrop": (crop_x, crop_y, crop_width, crop_height)},
    main={"size": (1920, 1080), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()
time.sleep(2)  # Laisse le temps au capteur de s'ajuster

print("Appuie sur 'espace' pour capturer une image, 'q' pour quitter.")

# === BOUCLE DE CAPTURE ===
i = 0
while True:
    frame = picam2.capture_array()
    display = frame.copy()
    cv2.putText(display, f"Image #{i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Capture Damier", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        filename = os.path.join(save_dir, f"calib_{i:02d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"[✔] Image enregistrée : {filename}")
        i += 1
    elif key == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()
