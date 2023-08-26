import os
import gradio
import openai
import config

openai.api_key = config.OPENAI_API_KEY

sys_usr_messages=[
    {"role": "system", "content": "You are a helpful assistant"},
]

def transcribe(audio):
    global sys_usr_messages

    audio_file_with_extension = audio + ".wav"
    os.rename(audio,audio_file_with_extension)

    audio_file = open(audio_file_with_extension, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)

    sys_usr_messages.append({"role": "user", "content": transcript["text"]})

    chat_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = sys_usr_messages
    )

    dialogue = chat_response["choices"][0]["message"]["content"]

    return dialogue

ui = gradio.Interface(fn=transcribe, inputs=gradio.Audio(source="microphone", type="filepath"), outputs="text").launch()

ui.launch()