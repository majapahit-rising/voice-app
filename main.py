import os
import json
import asyncio
import httpx
import re

from fastapi import FastAPI, WebSocket
from fastapi import WebSocketDisconnect
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

genai.configure(api_key=GEMINI_API_KEY)

# reuse client (WAJIB biar cepat)
http_client = httpx.AsyncClient(timeout=15)

# ======================================================
# GEMINI STREAM (ULTRA REALTIME)
# ======================================================

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
# STT
# ======================================================

async def speech_to_text(audio_bytes: bytes):
    files = {
        "file": ("audio.wav", audio_bytes, "audio/wav")
    }

    res = await http_client.post(STT_URL, files=files)
    return res.json().get("text", "")

# ======================================================
# TTS
# ======================================================

async def text_to_speech(text: str):
    res = await http_client.post(
        TTS_URL,
        json={
            "text": text,
            "voice": "en_us-lessac-medium"
        }
    )
    return res.content

# ======================================================
# WEBSOCKET REALTIME
# ======================================================

@app.websocket("/ws/deepcall")
async def deepcall(ws: WebSocket):
    await ws.accept()

    audio_buffer = bytearray()
    is_ai_speaking = False

    try:
        while True:
            try:
                data = await ws.receive()
            except RuntimeError:
                # client sudah disconnect
                print("Client disconnected (runtime)")
                break

            # =========================
            # AUDIO MASUK
            # =========================
            if "bytes" in data:
                audio_buffer.extend(data["bytes"])

                if is_ai_speaking:
                    await ws.send_json({"type": "interruption"})
                    is_ai_speaking = False

            # =========================
            # CONTROL
            # =========================
            elif "text" in data:
                msg = json.loads(data["text"])

                if msg.get("type") == "end-utterance":

                    user_text = await speech_to_text(bytes(audio_buffer))

                    await ws.send_json({
                        "type": "user-text",
                        "text": user_text
                    })

                    async for sentence in stream_gemini_sentences(user_text):

                        is_ai_speaking = True

                        await ws.send_json({
                            "type": "ai-text",
                            "text": sentence
                        })

                        audio_bytes = await text_to_speech(sentence)

                        for i in range(0, len(audio_bytes), 1024):
                            await ws.send_bytes(audio_bytes[i:i+1024])
                            await asyncio.sleep(0.005)

                    audio_buffer.clear()
                    is_ai_speaking = False

    except WebSocketDisconnect:
        print("Client disconnected cleanly")

    except Exception as e:
        print("WS ERROR:", e)

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
