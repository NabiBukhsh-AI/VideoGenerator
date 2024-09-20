from pydub import AudioSegment
import subprocess
import math
import cv2
import os

offset = 50

def get_audio_duration(audio_file):
    return len(AudioSegment.from_file(audio_file))

def write_text(segment, highlighted_word, frame, video_writer, font_color, highlighted_color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 4
    words = segment.split(" ")
    text_start_x = (frame.shape[1] - cv2.getTextSize(segment, font, font_scale, thickness)[0][0]) // 2
    text_y = (frame.shape[0] // 2) + cv2.getTextSize(segment, font, font_scale, thickness)[0][1]

    org = (text_start_x, text_y)
    for word in words:
        if word == highlighted_word:
            frame = cv2.putText(frame, word, org, font, font_scale, highlighted_color, thickness, cv2.LINE_AA)
        else:
            frame = cv2.putText(frame, word, org, font, font_scale, font_color, thickness, cv2.LINE_AA)
        word_size = cv2.getTextSize(word, font, font_scale, thickness)[0]
        org = (org[0] + word_size[0] + 10, org[1])
    video_writer.write(frame)

def add_narration_to_video(narrations, input_video, output_dir, output_file):
    cap = cv2.VideoCapture(input_video)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    temp_video = os.path.join(output_dir, "with_transcript.avi")
    out = cv2.VideoWriter(temp_video, fourcc, 30, (int(cap.get(3)), int(cap.get(4))))
    full_narration = AudioSegment.empty()

    for i, narration in enumerate(narrations):
        audio = os.path.join(output_dir, "narrations", f"narration_{i+1}.mp3")
        duration = get_audio_duration(audio)
        full_narration += AudioSegment.from_file(audio)

        words = narration.split(" ")
        line_length = 6 
        lines = [' '.join(words[j:j + line_length]) for j in range(0, len(words), line_length)]

        total_words = sum([len(line.split()) for line in lines])
        current_word_index = 0

        for line in lines:
            words_in_line = line.split()
            segment_duration = duration * len(words_in_line) / total_words
            segment_frames = math.floor(segment_duration / 1000 * 30)

            for frame_index in range(segment_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                word_progress = frame_index / segment_frames
                word_index_in_line = min(int(word_progress * len(words_in_line)), len(words_in_line) - 1)
                highlighted_word = words_in_line[word_index_in_line]
                font_color = (255, 255, 255)
                write_text(line, highlighted_word, frame, out, font_color, (0, 255, 0))

            current_word_index += len(words_in_line)

        remaining_frames = math.floor(duration / 1000 * 30) - sum([math.floor(duration * len(line.split(" ")) / len(words) / 1000 * 30) for line in lines])
        for _ in range(remaining_frames):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    temp_narration = os.path.join(output_dir, "narration.mp3")
    full_narration.export(temp_narration, format="mp3")

    ffmpeg_command = [
        'ffmpeg',
        '-y',
        '-i', temp_video,
        '-i', temp_narration,
        '-map', '0:v',  
        '-map', '1:a',   
        '-c:v', 'copy', 
        '-c:a', 'aac',   
        '-strict', 'experimental',
        os.path.join(output_dir, output_file)
    ]

    subprocess.run(ffmpeg_command, capture_output=True)

    os.remove(temp_video)
    os.remove(temp_narration)
