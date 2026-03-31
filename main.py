import os
import httpx
from fastapi import FastAPI
import google.generativeai as genai

app = FastAPI()

# Konfigurasi Gemini dari Environment Variable Render
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
AI_MODEL = os.getenv("AI_MODEL", "gemini-2.5-flash") 
genai.configure(api_key=GEMINI_API_KEY)

# URL Layanan Lokal Anda via Cloudflare
PIPER_URL = "https://tts.skendern8n.com/api/tts"

async def process_voice_logic(text: str):
    """Fungsi inti: Teks -> Gemini -> Piper"""
    try:
        # 1. Tanya ke Gemini
        model = genai.GenerativeModel(AI_MODEL)
        response = model.generate_content(text)
        answer_text = response.text

        # 2. Perintahkan Piper di rumah untuk bicara
        async with httpx.AsyncClient() as client:
            # Menggunakan voice 'af_sky' sesuai settingan Anda
            await client.get(PIPER_URL, params={"text": answer_text, "voice": "af_sky"})
        
        return answer_text
    except Exception as e:
        return f"Error: {str(e)}"

@app.post("/chat")
async def chat_via_text(user_text: str):
    """Endpoint untuk tes manual via Postman (Input Teks)"""
    answer = await process_voice_logic(user_text)
    return {
        "gemini_response": answer,
        "tts_status": "Sent to Local Piper",
        "target": "tts.skendern8n.com"
    }

@app.post("/stt-bridge")
async def bridge_from_whisper(text: str):
    """Endpoint khusus untuk menerima kiriman otomatis dari Whisper lokal"""
    print(f"Menerima suara dari rumah: {text}")
    answer = await process_voice_logic(text)
    return {"status": "processed", "reply": answer}

@app.get("/")
def home():
    return {
        "status": "Online",
        "model": AI_MODEL,
        "bridge": "Render to IhubLLM Active"
    }
