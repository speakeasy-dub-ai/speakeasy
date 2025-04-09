import os
import torch
import whisper
from pydub import AudioSegment
from langdetect import detect
from TTS.api import TTS
from transformers import MarianMTModel, MarianTokenizer
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# --- SETUP ---
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("small", device=device)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

CHUNK_DIR = "chunks"
OUTPUT_DIR = "outputs"
FINAL_OUTPUT = "final_output.wav"
os.makedirs(CHUNK_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# GUI LOGIC
class App:
    def __init__(self, master):
        self.master = master
        master.title("🎙️ SpeakEasy")
        master.geometry("500x400")
        master.configure(bg="#f5f5f5")

        self.input_path = ""
        self.speaker_path = ""

        # UI
        tk.Label(master, text="Dub Anytime, Anywhere", bg="#f5f5f5", font=("Helvetica", 16, "bold")).pack(pady=10)

        tk.Button(master, text="🎵 Select Input Audio", command=self.select_input).pack(pady=5)
        tk.Button(master, text="🧑‍🎤 Select Speaker Audio", command=self.select_speaker).pack(pady=5)
        tk.Button(master, text="🚀 Start Dubbing", command=self.run_pipeline).pack(pady=10)

        self.progress = ttk.Progressbar(master, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(pady=20)

        self.status = tk.Label(master, text="", bg="#f5f5f5", font=("Helvetica", 10))
        self.status.pack()

    def select_input(self):
        self.input_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
        self.status.config(text=f"Selected input: {os.path.basename(self.input_path)}")

    def select_speaker(self):
        self.speaker_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
        self.status.config(text=f"Selected speaker: {os.path.basename(self.speaker_path)}")

    def run_pipeline(self):
        if not self.input_path or not self.speaker_path:
            messagebox.showerror("Missing Files", "Please select both input and speaker files.")
            return

        self.status.config(text="🔄 Splitting audio...")
        self.master.update()

        chunk_paths = split_audio(self.input_path)
        total = len(chunk_paths)
        self.progress["maximum"] = total
        output_paths = []

        for i, chunk_path in enumerate(chunk_paths):
            self.status.config(text=f"🔹 Chunk {i+1}/{total}")
            self.master.update()

            en_text = transcribe_whisper(chunk_path)
            if not en_text:
                continue

            hi_text = translate_to_hindi(en_text)
            out_audio = synthesize(hi_text, i, self.speaker_path)
            output_paths.append(out_audio)

            self.progress["value"] = i + 1
            self.master.update()

        self.status.config(text="🔗 Merging audio...")
        self.master.update()

        combine_chunks(output_paths, FINAL_OUTPUT)
        self.status.config(text=f"✅ Done! Output saved at: {FINAL_OUTPUT}")
        messagebox.showinfo("Success", f"Hindi dubbed audio saved at:\n{FINAL_OUTPUT}")

root = tk.Tk()
app = App(root)
root.mainloop()
