import os
import torch
import whisper
from pydub import AudioSegment
from langdetect import detect
from TTS.api import TTS
from transformers import MarianMTModel, MarianTokenizer
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
from pathlib import Path
import uuid

# --- SETUP ---
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("small", device=device)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

UPLOAD_DIR = "uploads"
CHUNK_DIR = "chunks"
OUTPUT_DIR = "outputs"
os.makedirs(CHUNK_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="SpeakEasy API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Processing Functions ---
def split_audio(input_file, chunk_length=15000):
    audio = AudioSegment.from_file(input_file)
    chunks = []
    for i in range(0, len(audio), chunk_length):
        chunk = audio[i:i+chunk_length]
        chunk_path = os.path.join(CHUNK_DIR, f"chunk_{i//chunk_length}.wav")
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks

def transcribe_whisper(chunk_path):
    result = whisper_model.transcribe(chunk_path, fp16=torch.cuda.is_available())
    return result['text'].strip()

def translate_to_hindi(text):
    if detect(text) == 'hi':
        return text
    model_name = 'Helsinki-NLP/opus-mt-en-hi'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def synthesize(text, index, speaker_path):
    out_path = os.path.join(OUTPUT_DIR, f"output_{index}.wav")
    tts.tts_to_file(text=text, speaker_wav=speaker_path, language="hi", file_path=out_path)
    return out_path

def combine_chunks(chunk_files, output_path):
    combined = AudioSegment.empty()
    for file in chunk_files:
        combined += AudioSegment.from_file(file)
    combined.export(output_path, format="wav")

async def process_audio(input_path, speaker_path, output_path):
    chunk_paths = split_audio(input_path)
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
        out_audio = synthesize(hi_text, i, speaker_path)
        output_paths.append(out_audio)

    combine_chunks(output_paths, output_path)
    print(f"\n✅ Final dubbed Hindi audio saved at: {output_path}")
    
    # Clean up chunks
    for path in chunk_paths:
        if os.path.exists(path):
            os.remove(path)

@app.post("/speakeasy")
async def speakeasy_endpoint(
    background_tasks: BackgroundTasks,
    input_audio: UploadFile = File(...),
    speaker_audio: UploadFile = File(...)
):
    # Generate unique ID for this request
    request_id = str(uuid.uuid4())
    
    # Save uploaded files
    input_path = os.path.join(UPLOAD_DIR, f"{request_id}_input.wav")
    speaker_path = os.path.join(UPLOAD_DIR, f"{request_id}_speaker.wav")
    output_path = os.path.join(OUTPUT_DIR, f"{request_id}_output.wav")
    
    # Save input audio
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(input_audio.file, buffer)
    
    # Save speaker audio
    with open(speaker_path, "wb") as buffer:
        shutil.copyfileobj(speaker_audio.file, buffer)
    
    # Process audio in background
    background_tasks.add_task(
        process_audio, 
        input_path=input_path,
        speaker_path=speaker_path,
        output_path=output_path
    )
    
    return {
        "message": "Audio processing started",
        "request_id": request_id,
        "result_url": f"/speakeasy/result/{request_id}"
    }

@app.get("/speakeasy/result/{request_id}")
async def get_result(request_id: str):
    output_path = os.path.join(OUTPUT_DIR, f"{request_id}_output.wav")
    
    if not os.path.exists(output_path):
        return {"status": "processing"}
    
    return FileResponse(
        path=output_path,
        media_type="audio/wav",
        filename="dubbed_audio.wav"
    )

@app.get("/")
async def root():
    return {"message": "SpeakEasy API is running. Use /speakeasy endpoint to dub audio."}
