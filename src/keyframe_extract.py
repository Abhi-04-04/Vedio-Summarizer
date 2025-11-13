import os
import cv2
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

# Input and output paths
video_path = "data/example.mp4"
keyframe_dir = "data/keyframes"
os.makedirs(keyframe_dir, exist_ok=True)

# Initialize video and scene manager
video_manager = VideoManager([video_path])
scene_manager = SceneManager()
scene_manager.add_detector(ContentDetector(threshold=10.0))

# Detect scenes
video_manager.start()
scene_manager.detect_scenes(frame_source=video_manager)
scene_list = scene_manager.get_scene_list()

print(f"✅ Detected {len(scene_list)} scenes")

# # Save a keyframe from each scene
# cap = cv2.VideoCapture(video_path)
# for i, (start_time, end_time) in enumerate(scene_list):
#     # Take the middle frame of the scene
#     frame_time = (start_time.get_seconds() + end_time.get_seconds()) / 2
#     cap.set(cv2.CAP_PROP_POS_MSEC, frame_time * 1000)
#     success, frame = cap.read()
#     if success:
#         frame_path = os.path.join(keyframe_dir, f"keyframe_{i+1}.jpg")
#         cv2.imwrite(frame_path, frame)
#         print(f"📸 Saved keyframe_{i+1}.jpg at {frame_time:.2f}s")
if len(scene_list) == 0:
    print("⚠️ No scene changes detected — using fallback frame sampling...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
    sample_rate = 10  # capture every 10 seconds

    frame_idx = 1
    for t in range(0, duration, sample_rate):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        success, frame = cap.read()
        if success:
            frame_path = os.path.join(keyframe_dir, f"keyframe_{frame_idx}.jpg")
            cv2.imwrite(frame_path, frame)
            print(f"📸 Saved fallback keyframe_{frame_idx}.jpg at {t}s")
            frame_idx += 1

    cap.release()
else:
    # Save a keyframe from each detected scene
    cap = cv2.VideoCapture(video_path)
    for i, (start_time, end_time) in enumerate(scene_list):
        frame_time = (start_time.get_seconds() + end_time.get_seconds()) / 2
        cap.set(cv2.CAP_PROP_POS_MSEC, frame_time * 1000)
        success, frame = cap.read()
        if success:
            frame_path = os.path.join(keyframe_dir, f"keyframe_{i+1}.jpg")
            cv2.imwrite(frame_path, frame)
            print(f"📸 Saved keyframe_{i+1}.jpg at {frame_time:.2f}s")

    cap.release()

video_manager.release()
   

cap.release()
video_manager.release()
