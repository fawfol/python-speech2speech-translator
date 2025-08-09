import gradio as gr
import assemblyai as aai
from translate import Translator
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
import uuid
from pathlib import Path
import os

def audio_transcription(audio_file):
    aai.settings.api_key = "b52fa5fa...." 
    
    if not aai.settings.api_key:
        raise gr.Error("AssemblyAI API key not set. Please set it in the code or as an environment variable.")

    transcriber = aai.Transcriber()
    transcription = transcriber.transcribe(audio_file)
    
    return transcription

def text_translation(text):
    translator_es = Translator(from_lang="en", to_lang="es")
    es_text = translator_es.translate(text)
    translator_ja = Translator(from_lang="en", to_lang="ja")
    ja_text = translator_ja.translate(text)
    
    return es_text, ja_text
    
def text_to_speech(text):
    elevenlabs_api_key = "sk_5....."
    if not elevenlabs_api_key:
        raise gr.Error("ElevenLabs API key not set. Please set it in the code or as an environment variable.")

    client = ElevenLabs(api_key=elevenlabs_api_key)

    response = client.text_to_speech.convert(
        voice="pNInz6obpgDQFcFnaJgB",
        optimize_streaming_latency="0",
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_multilingual_v2",
        voice_settings=VoiceSettings(
            stability=0.5,
            similarity_boost=0.8,
            style=0.5,
            use_speaker_boost=True,
        ),
    )

    save_file_path = f"{uuid.uuid4()}.mp3"

    with open(save_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

    print(f"{save_file_path}: A new audio file was saved successfully!")

    return save_file_path

def voice_to_voice(audio_file):
    transcription_response = audio_transcription(audio_file)
    
    if transcription_response.status == aai.TranscriptStatus.error:
        raise gr.Error(transcription_response.error)
    else:
        text = transcription_response.text

    es_translation, ja_translation = text_translation(text)

    es_audio_path = text_to_speech(es_translation)
    ja_audio_path = text_to_speech(ja_translation)

    es_path = Path(es_audio_path) 
    ja_path = Path(ja_audio_path)
    
    return es_path, ja_path

audio_input = gr.Audio(
    sources=("microphone"),
    type="filepath"
)

demo = gr.Interface(
    fn=voice_to_voice,
    inputs=audio_input,
    outputs=[gr.Audio(label="Spanish"), gr.Audio(label="Japanese")]
)

if __name__ == "__main__":
    demo.launch()
