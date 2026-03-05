#!/usr/bin/env python3
import os
import glob
import cv2
import numpy as np

# ========= USER SETTINGS =========
IMAGE_GLOB = "../data/calib_images_640x480/*.png"
CHESSBOARD_SIZE = (9, 6)   # internal corners (cols, rows)
SQUARE_SIZE_MM = 20.0      # change to your actual printed square size (mm)
OUTPUT_NPZ = "../data/camera_intrinsics_640x480_20cmfocus.npz"
OUTPUT_YAML = "../data/camera_intrinsics_640x480_20cmfocus.yaml"
# =================================

def save_yaml(path, camera_matrix, dist_coeffs, image_size, rms, mean_error, used_images):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", int(image_size[0]))
    fs.write("image_height", int(image_size[1]))
    fs.write("camera_matrix", camera_matrix)
    fs.write("dist_coeffs", dist_coeffs)
    fs.write("rms_reprojection_error", float(rms))
    fs.write("mean_reprojection_error", float(mean_error))
    fs.write("num_images_used", int(used_images))
    fs.release()

def main():
    image_paths = sorted(glob.glob(IMAGE_GLOB))
    if not image_paths:
        raise FileNotFoundError(f"No images found matching: {IMAGE_GLOB}")

    print(f"Found {len(image_paths)} images")

    # Object points for one chessboard view
    # (0,0,0), (1,0,0), ..., scaled by actual square size
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    objpoints = []  # 3D points in world coordinates
    imgpoints = []  # 2D points in image plane

    img_size = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    used = 0
    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            print(f"Skipping unreadable image: {p}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])  # (w, h)

        found, corners = cv2.findChessboardCorners(
            gray, CHESSBOARD_SIZE,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if not found:
            # Optional fallback to findChessboardCornersSB (more robust in some cases)
            try:
                found_sb, corners_sb = cv2.findChessboardCornersSB(gray, CHESSBOARD_SIZE)
                if found_sb:
                    found = True
                    corners = corners_sb
            except Exception:
                pass

        if found:
            # cornerSubPix expects the classic detector corners shape; safe to try
            try:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            except Exception:
                pass

            objpoints.append(objp.copy())
            imgpoints.append(corners)
            used += 1
            print(f"[OK] {p}")
        else:
            print(f"[NO] {p}")

    if used < 10:
        raise RuntimeError(f"Only {used} valid images found. Capture more diverse views (recommend 15-30).")

    print(f"\nUsing {used} valid images for calibration")
    print(f"Image size: {img_size[0]} x {img_size[1]}")

    # Run calibration
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )

    # Compute mean reprojection error
    total_error = 0.0
    total_points = 0
    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        err = cv2.norm(imgpoints[i], projected, cv2.NORM_L2)
        n = len(projected)
        total_error += (err * err)
        total_points += n

    mean_error = np.sqrt(total_error / total_points) if total_points > 0 else float("nan")

    print("\n=== Calibration Results ===")
    print(f"RMS reprojection error (OpenCV): {rms:.6f}")
    print(f"Mean reprojection error (px):    {mean_error:.6f}")
    print("Camera matrix K:\n", camera_matrix)
    print("Distortion coeffs:\n", dist_coeffs.ravel())

    # Save NPZ
    np.savez(
        OUTPUT_NPZ,
        image_width=img_size[0],
        image_height=img_size[1],
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rms=rms,
        mean_reprojection_error=mean_error,
        chessboard_size=np.array(CHESSBOARD_SIZE),
        square_size_mm=SQUARE_SIZE_MM,
    )
    print(f"\nSaved NPZ: {OUTPUT_NPZ}")

    # Save YAML (OpenCV-readable)
    save_yaml(OUTPUT_YAML, camera_matrix, dist_coeffs, img_size, rms, mean_error, used)
    print(f"Saved YAML: {OUTPUT_YAML}")


if __name__ == "__main__":
    main()