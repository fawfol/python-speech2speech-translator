import gradio as gr
import assemblyai as aai
from translate import Translator
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
import uuid
from pathlib import Path
import os # Import os for environment variables (good practice)

# It's recommended to set API keys using environment variables
# For example, in your terminal before running:
# export ASSEMBLYAI_API_KEY="YOUR_ASSEMBLYAI_API_KEY"
# export ELEVENLABS_API_KEY="sk_YOUR_ELEVENLABS_API_KEY"

def audio_transcription(audio_file):
    # Ensure your AssemblyAI API key is set, preferably via environment variables
    # aai.settings.api_key = os.environ.get("ASSEMBLYAI_API_KEY") # Recommended
    aai.settings.api_key = "b52fa5fa" 
    
    if not aai.settings.api_key:
        raise gr.Error("AssemblyAI API key not set. Please set it in the code or as an environment variable.")

    transcriber = aai.Transcriber()
    transcription = transcriber.transcribe(audio_file)
    
    return transcription

def text_translation(text): # 'text' is now a parameter
    translator_es = Translator(from_lang="en", to_lang="es")
    es_text = translator_es.translate(text) # Consistent variable name

    translator_tr = Translator(from_lang="en", to_lang="tr")
    tr_text = translator_tr.translate(text) # Consistent variable name

    translator_ja = Translator(from_lang="en", to_lang="ja")
    ja_text = translator_ja.translate(text) # Consistent variable name
    
    return es_text, tr_text, ja_text
    
def text_to_speech(text):
    # Ensure your ElevenLabs API key is set, preferably via environment variables
    # elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY") # Recommended
    elevenlabs_api_key = "sk_5"
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
            use_speaker_boost=True, # Corrected boolean value
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
    # Transcribe audio
    transcription_response = audio_transcription(audio_file)
    
    if transcription_response.status == aai.TranscriptStatus.error:
        raise gr.Error(transcription_response.error)
    else:
        text = transcription_response.text

    # Translate text
    es_translation, tr_translation, ja_translation = text_translation(text) # Pass 'text'

    # Convert translated text to speech
    es_audio_path = text_to_speech(es_translation)
    tr_audio_path = text_to_speech(tr_translation)
    ja_audio_path = text_to_speech(ja_translation)

    es_path = Path(es_audio_path) # Corrected variable name
    tr_path = Path(tr_audio_path) # Corrected variable name
    ja_path = Path(ja_audio_path) # Corrected variable name
    
    return es_path, tr_path, ja_path

audio_input = gr.Audio(
    sources=("microphone"),
    type="filepath"
)

demo = gr.Interface(
    fn=voice_to_voice,
    inputs=audio_input,
    outputs=[gr.Audio(label="Spanish"), gr.Audio(label="Turkish"), gr.Audio(label="Japanese")]
)

if __name__ == "__main__":
    demo.launch()
