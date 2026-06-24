import argparse
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from utils import download_video, is_youtube_url
from preprocess import extract_clips
from model import load_model, predict


def run(source: str):
    # Step 1: Get the video file
    try:
        if is_youtube_url(source):
            print("[*] Detected YouTube URL. Downloading...")
            video_path = download_video(source, output_dir="videos")
        else:
            if not os.path.exists(source):
                print(f"[!] Error: File not found: {source}")
                sys.exit(1)
            video_path = source
            print(f"[*] Using local file: {video_path}")

        # Step 2: Split video into 3-second clips and extract 16 frames each
        print("[*] Splitting video into 3-second clips...")
        clips = extract_clips(video_path)
    except Exception as exc:
        print(f"[!] Error while preparing video input: {exc}")
        sys.exit(1)

    # Step 3: Load model
    print("[*] Loading model...")
    model = load_model()

    # Step 4: Run inference on each clip - violent if ANY clip is violent
    print("[*] Running inference on all clips...")
    is_violent = False
    for i, clip in enumerate(clips):
        result = predict(model, clip)
        print(f"    Clip {i+1}/{len(clips)}: {'VIOLENT' if result else 'OK'}")
        if result:
            is_violent = True

    # Step 5: Print final result
    print("\n" + "=" * 30)
    if is_violent:
        print("Violence detected.")
    else:
        print("No violence detected.")
    print("=" * 30 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Smart Security - Violence Detection")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a local video file or a YouTube URL",
    )
    args = parser.parse_args()
    run(args.input)
