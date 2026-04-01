import os
import json
import asyncio
import httpx
import re
import aiomysql
import wave
import io

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import google.generativeai as genai

# ======================================================
# CONFIG
# ======================================================

app = FastAPI()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
AI_MODEL = os.getenv("AI_MODEL", "gemini-2.5-flash")

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

# Reuse client untuk performa maksimal
http_client = httpx.AsyncClient(timeout=20)

# ======================================================
# DATABASE LOGIC
# ======================================================

async def get_welcome_message_from_db():
    """Mengambil pesan selamat datang yang aktif dari MySQL."""
    try:
        conn = await aiomysql.connect(
            host=DB_SETTINGS["host"],
            port=DB_SETTINGS["port"],
            user=DB_SETTINGS["user"],
            password=DB_SETTINGS["password"],
            db=DB_SETTINGS["db"]
        )
        async with conn.cursor(aiomysql.DictCursor) as cur:
            query = "SELECT message_text FROM aiorder_welcome_message WHERE is_active = 1 LIMIT 1"
            await cur.execute(query)
            result = await cur.fetchone()
            return result['message_text'] if result else "Halo, ada yang bisa saya bantu?"
    except Exception as e:
        print(f"❌ DATABASE ERROR: {e}")
        return "Halo, selamat datang di layanan kami."
    finally:
        if 'conn' in locals():
            conn.close()

# ======================================================
# CORE ENGINE: STT, TTS, GEMINI
# ======================================================

async def speech_to_text(audio_bytes: bytes):
    files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
    try:
        res = await http_client.post(STT_URL, files=files)
        return res.json().get("text", "")
    except Exception as e:
        print(f"STT Error: {e}")
        return ""

async def text_to_speech(text: str):
    try:
        res = await http_client.post(
            TTS_URL,
            json={
                "text": text,
                "voice": "en_us-lessac-medium"
            }
        )
        return res.content
    except Exception as e:
        print(f"TTS Error: {e}")
        return b""

async def stream_gemini_sentences(text: str):
    model = genai.GenerativeModel(AI_MODEL)
    response = model.generate_content(text, stream=True)
    buffer = ""

    for chunk in response:
        if not chunk.text:
            continue
        buffer += chunk.text
        parts = re.split(r'([.!?])', buffer)

        for i in range(0, len(parts) - 1, 2):
            sentence = parts[i] + parts[i + 1]
            yield sentence.strip()

        if len(parts) % 2 == 1:
            buffer = parts[-1]
        else:
            buffer = ""

# ======================================================
# CONVERT WAV TO PCM
# ======================================================

def wav_to_pcm16(wav_bytes: bytes) -> bytes:
    try:
        with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()

            print(f"🔊 WAV INFO → channels={channels}, width={sample_width}, rate={framerate}")

            frames = wf.readframes(wf.getnframes())

            # ✅ convert ke mono kalau stereo
            if channels == 2:
                import struct
                mono = bytearray()

                for i in range(0, len(frames), 4):  # 2 channel × 2 byte
                    left = struct.unpack('<h', frames[i:i+2])[0]
                    right = struct.unpack('<h', frames[i+2:i+4])[0]
                    avg = int((left + right) / 2)
                    mono.extend(struct.pack('<h', avg))

                frames = bytes(mono)

            return frames

    except Exception as e:
        print(f"❌ WAV DECODE ERROR: {e}")
        return b""

# ======================================================
# AUDIO STREAMER HELPER
# ======================================================

async def process_and_send_audio(ws: WebSocket, text: str):
    if not text.strip():
        return

    # Kirim teks ke UI
    await ws.send_json({
        "type": "ai-text",
        "text": text
    })

    # Ambil audio dari TTS
    audio_bytes = await text_to_speech(text)

    if not audio_bytes:
        return

    # ✅ FIX UTAMA: decode WAV → PCM
    if audio_bytes[:4] == b'RIFF':
        audio_bytes = wav_to_pcm16(audio_bytes)

    if not audio_bytes:
        return

    # ✅ streaming chunk aman
    chunk_size = 1024

    for i in range(0, len(audio_bytes), chunk_size):
        chunk = audio_bytes[i:i + chunk_size]

        # wajib genap (PCM16)
        if len(chunk) % 2 != 0:
            chunk = chunk[:-1]

        if chunk:
            await ws.send_bytes(chunk)
            await asyncio.sleep(0.005)
# ======================================================
# WEBSOCKET REALTIME
# ======================================================

@app.websocket("/ws/deepcall")
async def deepcall(ws: WebSocket):
    await ws.accept()
    print("🚀 Connection established")

    audio_buffer = bytearray()
    is_ai_speaking = False

    # --- ACTION: WELCOME MESSAGE ---
    # Dipicu langsung saat koneksi WebSocket berhasil terbuka
    welcome_text = await get_welcome_message_from_db()
    print(f"📢 Welcome Message: {welcome_text}")
    
    is_ai_speaking = True
    await process_and_send_audio(ws, welcome_text)
    is_ai_speaking = False

    try:
        while True:
            try:
                data = await ws.receive()
            except RuntimeError:
                print("Client disconnected (runtime)")
                break

            # 1. HANDLE AUDIO BYTES (MIC)
            if "bytes" in data:
                audio_buffer.extend(data["bytes"])

                # Jika user memotong pembicaraan AI
                if is_ai_speaking:
                    await ws.send_json({"type": "interruption"})
                    is_ai_speaking = False

            # 2. HANDLE CONTROL MESSAGE (JSON)
            elif "text" in data:
                msg = json.loads(data["text"])

                if msg.get("type") == "end-utterance":
                    if len(audio_buffer) == 0:
                        continue

                    # Transcribe suara user
                    user_text = await speech_to_text(bytes(audio_buffer))
                    audio_buffer.clear() # Bersihkan buffer setelah diproses

                    await ws.send_json({
                        "type": "user-text",
                        "text": user_text
                    })

                    # Generate respon dari Gemini secara streaming
                    if user_text.strip():
                        async for sentence in stream_gemini_sentences(user_text):
                            is_ai_speaking = True
                            await process_and_send_audio(ws, sentence)
                        
                        is_ai_speaking = False

    except WebSocketDisconnect:
        print("Client disconnected cleanly")
    except Exception as e:
        print(f"WS ERROR: {e}")
    finally:
        print("Connection closed")

# ======================================================
# SIMPLE CHAT (DEBUG)
# ======================================================

@app.post("/chat")
async def chat(user_text: str):
    model = genai.GenerativeModel(AI_MODEL)
    response = model.generate_content(user_text)

    return JSONResponse({
        "response": response.text
    })

# ======================================================
# HEALTH CHECK
# ======================================================

@app.get("/")
def home():
    return {
        "status": "Realtime Voice AI Ready",
        "model": AI_MODEL
    }
