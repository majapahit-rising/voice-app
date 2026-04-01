import os
import asyncio
import httpx
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ======================================================
# CONFIGURATION
# ======================================================

app = FastAPI()

# Middleware CORS untuk koneksi dari browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys & Model
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
AI_MODEL = os.getenv("AI_MODEL", "gemini-2.5-flash")

# Endpoint Services
STT_URL = "https://stt.skendern8n.com/stt"
TTS_URL = "https://tts.skendern8n.com/tts"

# Inisialisasi Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Client HTTP dengan timeout panjang untuk proses berantai
http_client = httpx.AsyncClient(timeout=45.0)

# ======================================================
# CORE FUNCTIONS
# ======================================================

async def speech_to_text(audio_bytes: bytes):
    """Mengirim file audio WebM ke service STT."""
    files = {"file": ("audio.webm", audio_bytes, "audio/webm")}
    try:
        res = await http_client.post(STT_URL, files=files)
        if res.status_code == 200:
            return res.json().get("text", "")
        print(f"❌ STT Failed: {res.status_code}")
        return ""
    except Exception as e:
        print(f"❌ STT Exception: {e}")
        return ""

async def text_to_speech(text: str):
    """Mengubah teks menjadi audio menggunakan service TTS."""
    try:
        # Menggunakan en_US (S besar) sesuai hasil tes Postman Anda yang working
        payload = {
            "text": text,
            "voice": "en_US-lessac-medium" 
        }
        res = await http_client.post(TTS_URL, json=payload)
        
        if res.status_code == 200:
            return res.content
        else:
            print(f"❌ TTS Server Error: {res.status_code} - {res.text}")
            return b""
    except Exception as e:
        print(f"❌ TTS Connection Error: {e}")
        return b""

# ======================================================
# ENDPOINTS
# ======================================================

@app.get("/welcome")
async def welcome():
    """
    Endpoint yang dipanggil browser saat pertama kali 'Call' dimulai.
    Memberikan suara sambutan tanpa menunggu user bicara.
    """
    welcome_text = "Hello, this is trial for TTS."
    print(f"📢 Sending Welcome: {welcome_text}")
    
    audio_bytes = await text_to_speech(welcome_text)
    audio_hex = audio_bytes.hex() if audio_bytes else ""
    
    return {
        "ai_text": welcome_text,
        "audio_base64": audio_hex
    }

@app.post("/chat")
async def chat(file: UploadFile = File(...)):
    """
    Endpoint utama untuk percakapan.
    Menerima Audio -> STT -> Gemini -> TTS -> Hex Audio Response.
    """
    try:
        # 1. Baca audio dari request
        audio_in_bytes = await file.read()
        
        # 2. Proses STT (Voice to Text)
        user_text = await speech_to_text(audio_in_bytes)
        print(f"👤 User: {user_text}")

        if not user_text or not user_text.strip():
            return {
                "user_text": "",
                "ai_text": "I couldn't hear you clearly.",
                "audio_base64": ""
            }

        # 3. Kirim teks ke Gemini AI
        model = genai.GenerativeModel(AI_MODEL)
        # to_thread agar pemrosesan AI tidak mengunci server
        response = await asyncio.to_thread(model.generate_content, user_text)
        ai_text = response.text
        print(f"🤖 AI: {ai_text}")

        # 4. Proses TTS (Text to Voice)
        audio_out_bytes = await text_to_speech(ai_text)
        
        # 5. Konversi ke HEX String (Sesuai kebutuhan deepcall.js Anda)
        audio_hex = audio_out_bytes.hex() if audio_out_bytes else ""

        return {
            "user_text": user_text,
            "ai_text": ai_text,
            "audio_base64": audio_hex
        }

    except Exception as e:
        print(f"🔥 CRITICAL ERROR: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/")
async def home():
    """Health Check."""
    return {
        "status": "Online",
        "mode": "HTTP_VOICE_CALL",
        "endpoints": ["/welcome", "/chat"]
    }

# ======================================================
# RUNNER
# ======================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
