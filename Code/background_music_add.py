from moviepy.editor import VideoFileClip, AudioFileClip
from pydub import AudioSegment
import math

def extract_audio(video_path, output_audio_path, audio_codec='mp3'):
    videoclip = VideoFileClip(video_path)

    audioclip = videoclip.audio

    audioclip.write_audiofile(output_audio_path, codec=audio_codec)

def combine_audio(background_music_path, original_audio_path, output_audio_path, background_music_strength=1.0):
    audio_clip1 = AudioSegment.from_file(background_music_path, format="mp3")

    audio_clip2 = AudioSegment.from_file(original_audio_path, format="mp3")

    max_duration = max(len(audio_clip1), len(audio_clip2))
    audio_clip1 = audio_clip1[:max_duration]
    audio_clip2 = audio_clip2[:max_duration]

    scaled_audio_clip1 = audio_clip1 - 10 * math.log10(1/background_music_strength)

    overlapped_audio = scaled_audio_clip1.overlay(audio_clip2)

    overlapped_audio.export(output_audio_path, format="mp3")

# def add_combined_audio_to_video(video_path, combined_audio_path, output_video_path, video_codec='libx264', audio_codec='aac'):
#     # Load the video clip
#     video_clip = VideoFileClip(video_path)

#     # Load the combined audio
#     combined_audio = AudioFileClip(combined_audio_path)

#     # Set the duration of the combined audio to match the video duration
#     combined_audio = combined_audio.set_duration(video_clip.duration)

#     # Set the video clip's audio to the combined audio
#     video_clip = video_clip.set_audio(combined_audio)

#     # Write the new video file with the combined audio
#     video_clip.write_videofile(output_video_path, codec=video_codec, audio_codec=audio_codec)
    
def add_combined_audio_to_video(video_path, combined_audio_path, output_video_path, video_codec='libx264', audio_codec='aac'):
    video_clip = VideoFileClip(video_path)

    combined_audio = AudioFileClip(combined_audio_path)

    video_clip = video_clip.set_audio(combined_audio)

    video_clip.write_videofile(output_video_path, codec=video_codec, audio_codec=audio_codec)
