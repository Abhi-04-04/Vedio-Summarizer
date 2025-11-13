import os
import sys
import subprocess

def extract_audio_ffmpeg(video_path, audio_path):
    """
    Extract audio from video using FFmpeg.
    Falls back to MoviePy if FFmpeg is not installed.
    """
    try:
        # Check if FFmpeg is installed
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # Build FFmpeg command
        ffmpeg_path = r"C:\Users\abhir\Downloads\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"
        cmd = [
            ffmpeg_path,
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            audio_path,
            "-y"
        ]

        subprocess.run(cmd, check=True)
        print(f"✅ Audio extracted successfully: {audio_path}")

    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ FFmpeg not found or failed. Falling back to MoviePy...")
        try:
            from moviepy.editor import VideoFileClip
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(audio_path, codec='pcm_s16le')
            print(f"✅ Audio extracted successfully using MoviePy: {audio_path}")
        except Exception as e:
            print(f"❌ MoviePy extraction failed: {e}")


if __name__ == "__main__":
    # Usage example: python audio_extract.py "input.mp4" "output.wav"

    if len(sys.argv) < 3:
        print("Usage: python audio_extract.py <input_video> <output_audio>")
        sys.exit(1)

    video_path = sys.argv[1]
    audio_path = sys.argv[2]

    if not os.path.exists(video_path):
        print(f"❌ Input file not found: {video_path}")
        sys.exit(1)

    extract_audio_ffmpeg(video_path, audio_path)
