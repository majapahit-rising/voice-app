from fastapi import FastAPI
import os
import google.generativeai as genai

app = FastAPI()

# Mengambil API Key dari Environment Variable di Render
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

@app.get("/")
def home():
    return {"status": "Server Render Aktif", "target_stt": "stt.skendern8n.com"}

@app.post("/chat")
async def chat(user_text: str):
    # Logika untuk mengirim pesan ke Gemini
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(user_text)
    return {"response": response.text}
