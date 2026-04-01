import os
import json
import asyncio
import httpx
import re
import aiomysql
import wave
import io
import base64

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai

# ======================================================
# CONFIGURATION
# ======================================================

app = FastAPI()

# Izinkan CORS agar bisa diakses dari domain frontend/browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
AI_MODEL = os.getenv("AI_MODEL", "gemini-2.0-flash") # Menggunakan model terbaru/stabil

STT_URL = "https://stt.skendern8n.com/stt"
TTS_URL = "https://tts.skendern8n.com/tts"

# Database Configuration
DB_SETTINGS = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "",
    "db": "ihub_crm"
}

genai.configure(api_key=GEMINI_API_KEY)

# Reuse client untuk efisiensi koneksi HTTP
http_client = httpx.AsyncClient(timeout=30)

# ======================================================
# CORE FUNCTIONS
# ======================================================

async def speech_to_text(audio_bytes: bytes):
    """Mengirim file audio ke layanan STT external."""
    files = {"file": ("audio.webm", audio_bytes, "audio/webm")}
    try:
        res = await http_client.post(STT_URL, files=files)
        res.raise_for_status()
        return res.json().get("text", "")
    except Exception as e:
        print(f"❌ STT Error: {e}")
        return ""

async def text_to_speech(text: str):
    """Mengubah teks menjadi audio WAV via TTS external."""
    try:
        res = await http_client.post(
            TTS_URL,
            json={
                "text": text,
                "voice": "en_us-lessac-medium"
            }
        )
        res.raise_for_status()
        return res.content
    except Exception as e:
        print(f"❌ TTS Error: {e}")
        return b""

# ======================================================
# ENDPOINT: CHAT (FOR HTTP DEEPCALL.JS)
# ======================================================

@app.post("/chat")
async def chat(file: UploadFile = File(...)):
    """
    Endpoint utama untuk deepcall.js
    Alur: Terima WebM -> STT -> Gemini -> TTS -> Hex String Response
    """
    try:
        # 1. Baca bytes dari file yang diupload browser
        audio_in_bytes = await file.read()
        
        # 2. Convert Suara ke Teks (STT)
        user_text = await speech_to_text(audio_in_bytes)
        
        if not user_text or not user_text.strip():
            return JSONResponse({
                "user_text": "",
                "ai_text": "Maaf, saya tidak bisa mendengar suara Anda dengan jelas.",
                "audio_base64": "" # Tetap kirim string kosong agar JS tidak error
            })

        # 3. Kirim ke Gemini AI (Mode non-streaming untuk HTTP)
        model = genai.GenerativeModel(AI_MODEL)
        
        # Gunakan asyncio.to_thread agar tidak memblokir event loop
        response = await asyncio.to_thread(model.generate_content, user_text)
        ai_text = response.text

        # 4. Convert Respon AI ke Suara (TTS)
        audio_out_bytes = await text_to_speech(ai_text)

        # 5. Konversi Audio Bytes ke HEX String
        # Sesuai dengan deepcall.js: playAudio(data.audio_base64) 
        # yang menggunakan logic: hex.match(/.{1,2}/g)
        audio_hex = audio_out_bytes.hex()

        return {
            "user_text": user_text,
            "ai_text": ai_text,
            "audio_base64": audio_hex
        }

    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# ======================================================
# UTILITIES & HEALTH CHECK
# ======================================================

@app.get("/")
async def home():
    return {
        "status": "Voice AI Engine Active (HTTP Mode)",
        "model": AI_MODEL,
        "endpoint": "/chat"
    }

# Jika dijalankan langsung
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
