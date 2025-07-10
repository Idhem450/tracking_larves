import cv2
import numpy as np
import glob
import os

# === PARAMÈTRES UTILISATEUR ===
pattern_size = (9, 6)          # (cols, rows) 
square_size = 25.0             
images_folder = "images_calibration"
output_file = "calibration_data.npz"

# === PRÉPARATION DES POINTS 3D (monde réel) ===
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size  # applique la taille réelle

objpoints = []  # Points 3D
imgpoints = []  # Points 2D

# === CHARGEMENT DES IMAGES ===
images = glob.glob(os.path.join(images_folder, "*.jpg")) + \
         glob.glob(os.path.join(images_folder, "*.png"))

if not images:
    raise ValueError(f"Aucune image trouvée dans le dossier '{images_folder}'.")

ok = 0
fail = 0

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        print(f" Damier détecté dans : {os.path.basename(fname)}")
        objpoints.append(objp)
        # amélioration précision des coins
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Affichage (optionnel, désactivé si SSH)
        # cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
        # cv2.imshow('Calibration - Coins détectés', img)
        # cv2.waitKey(200)
        ok += 1
    else:
        print(f" ÉCHEC détection dans : {os.path.basename(fname)}")
        fail += 1

# cv2.destroyAllWindows()  

print("\n--- Bilan détection ---")
print(f"  {ok} images valides")
print(f"  {fail} images non valides")

# === CALIBRATION ===
if ok < 3:
    print("\n Trop peu d’images valides pour une calibration fiable .")
else:
    print("\n Calibration en cours...")
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("\n Calibration terminée.")
    print(" Matrice intrinsèque (K):\n", K)
    print(" Coefficients de distorsion:\n", dist.ravel())

    np.savez(output_file, K=K, dist=dist)
    print(f"\n Paramètres sauvegardés dans : {output_file}")
