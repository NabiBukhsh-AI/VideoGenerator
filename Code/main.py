import streamlit as st
from openai import OpenAI
import time
import json
import os
import fitz
import narration
import images
import video
from background_music_add import extract_audio, combine_audio, add_combined_audio_to_video

api_key = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

artist = ''

def extract_text_from_pdf(pdf_path):
    try:
        pdf_document = fitz.open(pdf_path)
        extracted_text = ""

        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text = page.get_text()
            text = text.replace('\n', ' ')
            extracted_text += text

        pdf_document.close()
        return extracted_text

    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

def main():
    st.title("Manuel Short Video Generator")

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    uploaded_music = st.file_uploader("Upload background music", type=["mp3"])

    artist = st.text_area("Enter voice artist:", height=30)

    if uploaded_file:
        source_material = extract_text_from_pdf(uploaded_file)

        if source_material is not None:
            st.subheader("Source Material:")
            st.write(source_material)

            generate_button = st.button("Generate")

            if generate_button:
                if uploaded_music:
                    strength = st.slider("Set Background Music Strength", 0.0, 1.0, 0.5, 0.1)
                    generate_narration_with_music(source_material, uploaded_music, artist, strength)
                else:
                    generate_narration(source_material, artist)

def generate_narration(source_material, artist):
    short_id = str(int(time.time()))
    output_file = "short.avi"
    basedir = os.path.join("shorts", short_id)

    if not os.path.exists(basedir):
        os.makedirs(basedir)

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": """You are a YouTube short narration generator. You generate 30 seconds to 1 minute of narration. The shorts you create have a background that fades from image to image as the narration is going on.

    You will need to generate descriptions of images for each of the sentences in the short. They will be passed to an AI image generator. DO NOT IN ANY CIRCUMSTANCES use names of celebrities or people in the image descriptions. It is illegal to generate images of celebrities. Only describe persons without their names. Do not reference any real person or group in the image descriptions. Don't mention the female figure or other sexual content in the images because they are not allowed.

    You are however allowed to use any content, including real names in the narration. Only image descriptions are restricted.

    Note that the narration will be fed into a text-to-speech engine, so don't use special characters.

    Respond with a pair of an image description in square brackets and a narration below it. Both of them should be on their own lines, as follows:

    ###

    [Description of a background image]

    Narrator: "One sentence of narration"

    [Description of a background image]

    Narrator: "One sentence of narration"

    [Description of a background image]

    Narrator: "One sentence of narration"

    ###

    The short should be 6 sentences maximum.

    You should add a description of a fitting backround image in between all of the narrations. It will later be used to generate an image with AI.
    """
            },
            {
                "role": "user",
                "content": f"Create a YouTube short narration based on the following source material:\n\n{source_material}"
            }
        ]
    )

    response_text = response.choices[0].message.content
    response_text.replace("’", "'").replace("`", "'").replace("…", "...").replace("“", '"').replace("”", '"')

    with open(os.path.join(basedir, "response.txt"), "w") as f:
        f.write(response_text)

    data, narrations = narration.parse(response_text)
    with open(os.path.join(basedir, "data.json"), "w") as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"Generating narration...")
    narration.create(data, os.path.join(basedir, "narrations"), f'{artist}')

    print("Generating images...")
    images.create_from_data(data, os.path.join(basedir, "images"))

    print("Generating video...")
    video.create(narrations, basedir, output_file)

    st.success(f"DONE! Here's your video: {os.path.join(basedir, output_file)}")

def generate_narration_with_music(source_material, uploaded_music, artist, strength):
    short_id = str(int(time.time()))
    output_file = "short.avi"
    basedir = os.path.join("shorts", short_id)

    if not os.path.exists(basedir):
        os.makedirs(basedir)

    music_path = os.path.join(basedir, "background_music.mp3")
    uploaded_music.seek(0)
    with open(music_path, "wb") as music_file:
        music_file.write(uploaded_music.read())

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": """You are a YouTube short narration generator. You generate 30 seconds to 1 minute of narration. The shorts you create have a background that fades from image to image as the narration is going on.

    You will need to generate descriptions of images for each of the sentences in the short. They will be passed to an AI image generator. DO NOT IN ANY CIRCUMSTANCES use names of celebrities or people in the image descriptions. It is illegal to generate images of celebrities. Only describe persons without their names. Do not reference any real person or group in the image descriptions. Don't mention the female figure or other sexual content in the images because they are not allowed.

    You are however allowed to use any content, including real names in the narration. Only image descriptions are restricted.

    Note that the narration will be fed into a text-to-speech engine, so don't use special characters.

    Respond with a pair of an image description in square brackets and a narration below it. Both of them should be on their own lines, as follows:

    ###

    [Description of a background image]

    Narrator: "One sentence of narration"

    [Description of a background image]

    Narrator: "One sentence of narration"

    [Description of a background image]

    Narrator: "One sentence of narration"

    ###

    The short should be 6 sentences maximum.

    You should add a description of a fitting backround image in between all of the narrations. It will later be used to generate an image with AI.
    """
            },
            {
                "role": "user",
                "content": f"Create a YouTube short narration based on the following source material:\n\n{source_material}"
            }
        ]
    )

    response_text = response.choices[0].message.content
    response_text.replace("’", "'").replace("`", "'").replace("…", "...").replace("“", '"').replace("”", '"')

    with open(os.path.join(basedir, "response.txt"), "w") as f:
        f.write(response_text)

    data, narrations = narration.parse(response_text)
    with open(os.path.join(basedir, "data.json"), "w") as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"Generating narration...")
    narration.create(data, os.path.join(basedir, "narrations"), f'{artist}')

    print("Generating images...")
    images.create_from_data(data, os.path.join(basedir, "images"))

    print("Generating video...")
    video.create(narrations, basedir, output_file)

    st.success(f"DONE! Here's your video: {os.path.join(basedir, output_file)}")

    original_video_path = os.path.join(basedir, "short.avi")

    original_audio_path = os.path.join(basedir, "new_audio.mp3")
    extract_audio(original_video_path, original_audio_path)

    combined_audio_path = os.path.join(basedir, "combined_audio.mp3")
    combine_audio(music_path, original_audio_path, combined_audio_path, background_music_strength=strength)

    output_video_path = os.path.join(basedir, 'output_video_with_combined_audio.avi')
    add_combined_audio_to_video(original_video_path, combined_audio_path, output_video_path)

    st.success(f"DONE! Here's your video: {output_video_path}")

if __name__ == "__main__":
    main()