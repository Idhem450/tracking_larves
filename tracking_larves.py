import time
import json
import csv
import cv2
import numpy as np
import pigpio
from picamera2 import Picamera2

# === Paramètres de calibration caméra ===
K = np.array([[5282.32084, 0.0, 849.301187],
              [0.0, 5279.52624, 295.988566],
              [0.0, 0.0, 1.0]])
dist = np.array([[0.729973676, 8.28433969, -0.0403485395,
                  -0.00650735379, -295.103845]])

# === Classe pour suivre une larve ===
class Larve:
    def __init__(self, numero):
        self.numero = numero
        self.positions = []      # [(t, x, y, immobile)]
        self.derniere_pos = None

    def update(self, pos, t, seuil_distance=5):
        if self.derniere_pos is None:
            immobile = 0
        else:
            dx = pos[0] - self.derniere_pos[0]
            dy = pos[1] - self.derniere_pos[1]
            dist = np.sqrt(dx**2 + dy**2)
            immobile = int(dist < seuil_distance)
        self.positions.append((round(t, 2), pos[0], pos[1], immobile))
        self.derniere_pos = pos

# === LED PWM pigpio ===
def allumer_led_progressivement(pi, gpio=18, paliers=4, delay=0.5):
    pi.set_PWM_frequency(gpio, 10_000)
    for i in range(1, paliers + 1):
        duty = int((255 / paliers) * i)
        pi.set_PWM_dutycycle(gpio, duty)
        time.sleep(delay)

def eteindre_led_progressivement(pi, gpio=18, paliers=4, delay=0.5):
    for i in reversed(range(1, paliers + 1)):
        duty = int((255 / paliers) * i)
        pi.set_PWM_dutycycle(gpio, duty)
        time.sleep(delay)
    pi.set_PWM_dutycycle(gpio, 0)
    pi.stop()

# === Charger les puits [x, y, rayon] ===
def charger_puits(fichier="centres_puits.json"):
    with open(fichier, "r") as f:
        data = json.load(f)
    return [(int(x), int(y), int(rayon)) for x, y, rayon in data]

# === Initialisation caméra avec distorsion corrigée ===
def initialiser_camera():
    picam2 = Picamera2()
    sensor_width, sensor_height = picam2.sensor_resolution
    coeff_zoom = 1.0
    crop_width = int(sensor_width / coeff_zoom)
    crop_height = int(sensor_height / coeff_zoom)
    crop_x = (sensor_width - crop_width) // 2
    crop_y = (sensor_height - crop_height) // 2

    config = picam2.create_video_configuration(
        controls={
            "ScalerCrop": (crop_x, crop_y, crop_width, crop_height),
            "FrameRate": 5,
            "AwbEnable": False,
            "AeEnable": False,
            "AnalogueGain": 2.0,
            "ExposureTime": 480
        },
        main={"size": (1280, 720), "format": "RGB888"}
    )

    picam2.configure(config)
    picam2.start()
    time.sleep(2)

    test_img = picam2.capture_array()
    h, w = test_img.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, new_K, (w, h), cv2.CV_32FC1)

    return picam2, mapx, mapy

# === Initialisation MOG2 par puits ===
def initialiser_mog2_par_puits(nb_puits):
    return [cv2.createBackgroundSubtractorMOG2(history=125,
                                              varThreshold=60,
                                              detectShadows=False)
            for _ in range(nb_puits)]

# === Détection de larve avec MOG2 et CLAHE ===
def detecter_larve_mog2(frame, cx, cy, rayon, mog2):
    x1, x2 = cx - rayon, cx + rayon
    y1, y2 = cy - rayon, cy + rayon
    roi = frame[y1:y2, x1:x2].copy()
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Amélioration du contraste local
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Masque circulaire
    mask = np.zeros_like(gray)
    cv2.circle(mask, (rayon, rayon), rayon, 255, -1)
    gray = cv2.bitwise_and(gray, gray, mask=mask)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    fgmask = mog2.apply(gray)
    thresh = cv2.threshold(fgmask, 30, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    min_area = float('inf')
    best_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 4 < area < 400 and area < min_area:
            min_area = area
            best_cnt = cnt

    if best_cnt is not None:
        M = cv2.moments(best_cnt)
        if M["m00"] != 0:
            cx_rel = int(M["m10"] / M["m00"])
            cy_rel = int(M["m01"] / M["m00"])
            abs_pos = (cx_rel + x1, cy_rel + y1)
            cv2.circle(frame, abs_pos, 4, (0, 255, 0), -1)  # vert
            return frame, abs_pos

    return frame, None

# === Programme principal ===
def main():
    puits = charger_puits()
    larves = [Larve(i + 1) for i in range(len(puits))]

    pi = pigpio.pi()
    allumer_led_progressivement(pi)

    picam2, mapx, mapy = initialiser_camera()
    mog2_par_puits = initialiser_mog2_par_puits(len(puits))

    print("[INFO] Tracking en cours. Appuyez sur 'q' pour quitter.")
    t0 = time.time()

    while True:
        distorted = picam2.capture_array()
        frame = cv2.remap(distorted, mapx, mapy, cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        t = time.time() - t0

        for i, (cx, cy, r) in enumerate(puits):
            frame, pos = detecter_larve_mog2(frame, cx, cy, r,
                                             mog2_par_puits[i])
            if pos:
                larves[i].update(pos, t)
            else:
                # Si la larve n'est pas détectée, afficher sa dernière position
                if larves[i].derniere_pos is not None:
                    cv2.circle(frame, larves[i].derniere_pos, 4,
                               (0, 0, 255), -1)  # rouge

        cv2.imshow("Tracking multi-larves (MOG2 + CLAHE)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Nettoyage
    picam2.stop()
    cv2.destroyAllWindows()
    eteindre_led_progressivement(pi)

    # === Export CSV ===
    for larve in larves:
        nom_fichier = f"trajectoire_larve_{larve.numero}.csv"
        with open(nom_fichier, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "x", "y", "immobile"])
            for ligne in larve.positions:
                writer.writerow(ligne)
        print(f"[EXPORT] Données larve {larve.numero} sauvegardées dans "
              f"{nom_fichier}")

if __name__ == "__main__":
    main()
