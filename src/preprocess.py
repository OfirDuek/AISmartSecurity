import cv2
import torch
import numpy as np
from torchvision import transforms
from typing import List


FRAMES_PER_CLIP = 16
CLIP_DURATION   = 3        # seconds
IMG_SIZE        = 224

# Normalization values used during training
MEAN = [0.45, 0.45, 0.45]
STD  = [0.225, 0.225, 0.225]

frame_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])


def extract_clips(video_path: str) -> List[torch.Tensor]:
    """
    Splits the video into 3-second clips and extracts 16 evenly spaced frames
    from each clip.

    Returns a list of tensors, each of shape (C, T, H, W) = (3, 16, 224, 224).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps == 0 or total_frames == 0:
        raise ValueError("Video has no frames or invalid FPS.")

    frames_per_clip = int(fps * CLIP_DURATION)
    clips = []
    clip_start = 0

    while clip_start < total_frames:
        clip_end = min(clip_start + frames_per_clip, total_frames)
        indices  = np.linspace(clip_start, clip_end - 1, FRAMES_PER_CLIP, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                frame_tensor = frames[-1] if frames else torch.zeros(3, IMG_SIZE, IMG_SIZE)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = frame_transform(frame)
            frames.append(frame_tensor)

        clip_tensor = torch.stack(frames)           # [T, C, H, W]
        clip_tensor = clip_tensor.permute(1, 0, 2, 3)  # [C, T, H, W]
        clips.append(clip_tensor)

        clip_start += frames_per_clip

    cap.release()
    print(f"[+] Extracted {len(clips)} clips of {CLIP_DURATION}s from video.")
    return clips
