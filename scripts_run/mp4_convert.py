import os
import cv2

# ==== CONFIG ====
VIDEO_PATH = "/home/user1/rosbags/C0002.MP4"
OUTPUT_DIR = "/home/user1/WildGS-SLAM/datasets/video/C0002"

STRIDE = 1  # keep every Nth frame, increase to subsample a high-fps video

# ==== SETUP ====
os.makedirs(f"{OUTPUT_DIR}/rgb", exist_ok=True)
rgb_txt = open(f"{OUTPUT_DIR}/rgb.txt", "w")
rgb_txt.write("# timestamp filename\n")

# ==== VIDEO READER ====
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video FPS: {fps}, total frames: {total_frames}")

# ==== MAIN LOOP ====
frame_idx = 0
saved_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % STRIDE == 0:
        ts = frame_idx / fps

        rgb_name = f"frame_{saved_idx:06d}.png"
        write_ok = cv2.imwrite(f"{OUTPUT_DIR}/rgb/{rgb_name}", frame)
        if not write_ok:
            print(f"Failed to write {rgb_name}")

        rgb_txt.write(f"{ts:.6f} rgb/{rgb_name}\n")
        saved_idx += 1

        if saved_idx % 50 == 0:
            print(f"Saved frames: {saved_idx} (video frame {frame_idx} / {total_frames})")

    frame_idx += 1

# ==== CLEANUP ====
cap.release()
rgb_txt.close()

print(f"Done. Saved {saved_idx} frames to {OUTPUT_DIR}/rgb")
