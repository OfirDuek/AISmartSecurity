import os
import yt_dlp


def download_video(url: str, output_dir: str = "videos") -> str:
    """
    Downloads a YouTube video and returns the local file path.
    """
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        "format": "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,  # הורד רק סרטון בודד, לא פלייליסט
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info["id"]
        ext = info.get("ext", "mp4")
        output_path = os.path.join(output_dir, f"{video_id}.{ext}")

    print(f"[+] Video downloaded: {output_path}")
    return output_path


def is_youtube_url(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")
