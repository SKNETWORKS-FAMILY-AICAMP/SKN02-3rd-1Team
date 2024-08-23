from openai import OpenAI
import pyaudio
import wave
import streamlit as st
from utils.rag_utils import *

def record_audio(output_filename, record_seconds=5, sample_rate=44100, chunk_size=1024, channels=2):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)

    st.write("목소리를 듣고 있어요.")

    frames = []
    for _ in range(0, int(sample_rate / chunk_size * record_seconds)):
        data = stream.read(chunk_size)
        frames.append(data)

    st.write("다 들었어요. 잠시만 기다려주세요.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    client = OpenAI()

    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

    audio_file = open("output.wav", "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )

    prompt = transcription.text

    return prompt

def audio_btn(retriever, rag_prompt_custom, llm):
    if st.button("REC", type="primary"):
        prompt = record_audio("output.wav", record_seconds=5)
        input_audio = True
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        res = rag_chain(prompt, retriever, rag_prompt_custom, llm)
        st.session_state.chat_history.append({"role": "ai", "content": res})
    else:
        input_audio = False

    return input_audio

