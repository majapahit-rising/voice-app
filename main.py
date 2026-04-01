import os
import asyncio
import httpx
import google.generativeai as genai
import mysql.connector # Library untuk MySQL
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ======================================================
# CONFIGURATION & DATABASE
# ======================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Konfigurasi Database dari Cloudflare Tunnel
# Gunakan alamat tunnel tanpa 'https://' untuk host database
DB_CONFIG = {
    "host": "number-studied-shaved-teach.trycloudflare.com",
    "user": "ihubs_user",
    "password": "1234qwer",
    "database": "ihub_crm",
    "port": 3306,
    "auth_plugin": "caching_sha2_password" # Penting untuk MySQL 8.4
}

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
AI_MODEL = os.getenv("AI_MODEL", "gemini-2.0-flash") # Update ke versi stabil terbaru

STT_URL = "https://stt.skendern8n.com/stt"
TTS_URL = "https://tts.skendern8n.com/tts"

genai.configure(api_key=GEMINI_API_KEY)
http_client = httpx.AsyncClient(timeout=45.0)

# ======================================================
# DATABASE FUNCTIONS
# ======================================================

async def get_welcome_message_from_db():
    """Mengambil pesan welcome yang aktif dari database lokal."""
    try:
        # Menjalankan blocking database call di thread terpisah agar FastAPI tetap kencang
        def fetch():
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor(dictionary=True)
            
            # Mengambil 1 pesan yang is_active = 1
            query = "SELECT message_text FROM aiorder_welcome_messages WHERE is_active = 1 LIMIT 1"
            cursor.execute(query)
            result = cursor.fetchone()
            
            cursor.close()
            conn.close()
            return result["message_text"] if result else "Hello, how can I help you today?"

        return await asyncio.to_thread(fetch)
    except Exception as e:
        print(f"❌ DB Error: {e}")
        return "Hello, this is a fallback welcome message."

# ==========================
# CORE FUNCTIONS (STT/TTS)
# ==========================

async def speech_to_text(audio_bytes: bytes):
    files = {"file": ("audio.webm", audio_bytes, "audio/webm")}
    try:
        res = await http_client.post(STT_URL, files=files)
        return res.json().get("text", "") if res.status_code == 200 else ""
    except: return ""

async def text_to_speech(text: str):
    try:
        payload = {"text": text, "voice": "en_US-lessac-medium"}
        res = await http_client.post(TTS_URL, json=payload)
        return res.content if res.status_code == 200 else b""
    except: return b""

# ======================================================
# ENDPOINTS
# ======================================================

@app.get("/welcome")
async def welcome():
    # GANTI: Sekarang mengambil teks dari database
    welcome_text = await get_welcome_message_from_db()
    
    print(f"📢 Sending Welcome from DB: {welcome_text}")
    
    audio_bytes = await text_to_speech(welcome_text)
    audio_hex = audio_bytes.hex() if audio_bytes else ""
    
    return {
        "ai_text": welcome_text,
        "audio_base64": audio_hex
    }

@app.post("/chat")
async def chat(file: UploadFile = File(...)):
    try:
        audio_in_bytes = await file.read()
        user_text = await speech_to_text(audio_in_bytes)

        if not user_text or not user_text.strip():
            return {"user_text": "", "ai_text": "I couldn't hear you.", "audio_base64": ""}

        model = genai.GenerativeModel(AI_MODEL)
        response = await asyncio.to_thread(model.generate_content, user_text)
        ai_text = response.text

        audio_out_bytes = await text_to_speech(ai_text)
        audio_hex = audio_out_bytes.hex() if audio_out_bytes else ""

        return {
            "user_text": user_text,
            "ai_text": ai_text,
            "audio_base64": audio_hex
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/")
async def home():
    return {"status": "Online", "database": "Connected via Tunnel"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
