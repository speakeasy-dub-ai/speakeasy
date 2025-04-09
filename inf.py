import os
import torch
import whisper
from pydub import AudioSegment
from langdetect import detect
from TTS.api import TTS
from transformers import MarianMTModel, MarianTokenizer

# --- SETUP ---
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("small", device=device)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

INPUT_AUDIO = "input.wav"
SPEAKER_WAV = "speaker.wav"
CHUNK_DIR = "chunks"
OUTPUT_DIR = "outputs"
FINAL_OUTPUT = "final_output.wav"

os.makedirs(CHUNK_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. SPLIT AUDIO INTO ~15s CHUNKS ---
def split_audio(input_file, chunk_length=15000):  # 15 seconds
    audio = AudioSegment.from_file(input_file)
    chunks = []
    for i in range(0, len(audio), chunk_length):
        chunk = audio[i:i+chunk_length]
        chunk_path = os.path.join(CHUNK_DIR, f"chunk_{i//chunk_length}.wav")
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks

# --- 2. TRANSCRIBE USING WHISPER ---
def transcribe_whisper(chunk_path):
    result = whisper_model.transcribe(chunk_path, fp16=torch.cuda.is_available())
    return result['text'].strip()

# --- 3. TRANSLATE TO HINDI ---
def translate_to_hindi(text):
    if detect(text) == 'hi':
        return text
    model_name = 'Helsinki-NLP/opus-mt-en-hi'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# --- 4. SYNTHESIZE AUDIO ---
def synthesize(text, index):
    out_path = os.path.join(OUTPUT_DIR, f"output_{index}.wav")
    tts.tts_to_file(text=text, speaker_wav=SPEAKER_WAV, language="hi", file_path=out_path)
    return out_path

# --- 5. COMBINE OUTPUT ---
def combine_chunks(chunk_files, output_path):
    combined = AudioSegment.empty()
    for file in chunk_files:
        combined += AudioSegment.from_file(file)
    combined.export(output_path, format="wav")

# --- MAIN ---
chunk_paths = split_audio(INPUT_AUDIO)
output_paths = []

for i, chunk_path in enumerate(chunk_paths):
    print(f"\n🔹 Processing chunk {i+1}/{len(chunk_paths)}: {chunk_path}")
    en_text = transcribe_whisper(chunk_path)
    if not en_text:
        print(f"⚠️ Skipping chunk {i} due to transcription failure.")
        continue
    print(f"📝 English Transcription: {en_text}")
    hi_text = translate_to_hindi(en_text)
    print(f"🇮🇳 Hindi Translation: {hi_text}")
    out_audio = synthesize(hi_text, i)
    output_paths.append(out_audio)

combine_chunks(output_paths, FINAL_OUTPUT)
print(f"\n✅ Final dubbed Hindi audio saved at: {FINAL_OUTPUT}")
