import os
import torch
import whisper
from pydub import AudioSegment
from langdetect import detect
from TTS.api import TTS
from transformers import MarianMTModel, MarianTokenizer

# --- SETUP ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("small", device=device)  # Use "medium" or "large" for better accuracy
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

INPUT_AUDIO = "input.wav"         # Your source audio file
SPEAKER_WAV = "speaker.wav"       # Reference voice for cloning
OUTPUT_AUDIO = "final_hindi.wav"  # Final output filename

# --- 1. TRANSCRIBE TO ENGLISH ---
print("🔍 Transcribing English using Whisper...")
result = model.transcribe(INPUT_AUDIO, fp16=torch.cuda.is_available())
english_text = result['text'].strip()
print(f"📝 English Transcription:\n{english_text}")

# --- 2. TRANSLATE TO HINDI ---
def translate_to_hindi(text):
    if detect(text) == "hi":
        return text
    print("🌐 Translating to Hindi...")
    model_name = "Helsinki-NLP/opus-mt-en-hi"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    translated = model.generate(**inputs)
    hindi_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return hindi_text

hindi_text = translate_to_hindi(english_text)
print(f"🇮🇳 Hindi Translation:\n{hindi_text}")

# --- 3. SYNTHESIZE HINDI AUDIO ---
print("🎤 Synthesizing Hindi audio with Coqui XTTS...")
tts.tts_to_file(
    text=hindi_text,
    speaker_wav=SPEAKER_WAV,
    language="hi",
    file_path=OUTPUT_AUDIO
)

print(f"\n✅ Final Hindi audio saved as: {OUTPUT_AUDIO}")
