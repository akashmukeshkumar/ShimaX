import os
import uuid
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from groq import Groq
import edge_tts
import asyncio

# --- CONFIGURATION (SECURE) ---
# Humne key yahan se hata di hai. 
# Ab yeh server ke "Environment Variables" se key mangega.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI()

# Security: CORS Setup (Sab allow hai taake frontend connect ho sake)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Agar Key na mile to code phatne se bachane ke liye check
if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)
else:
    client = None
    print("⚠️ WARNING: API Key nahi mili! Render par 'Environment Variable' set karein.")

# --- SMART SEARCH (RAG Lite) ---
def find_relevant_context(query):
    """
    Student ke sawal se related chapter dhundta hai.
    """
    folder_path = "knowledge_base"
    best_file = None
    best_score = 0
    
    # Agar folder nahi mila (Server par upload nahi hua to)
    if not os.path.exists(folder_path):
        print(f"⚠️ Warning: '{folder_path}' folder nahi mila.")
        return "Mere paas abhi koi kitab (knowledge) nahi hai."

    query_words = set(query.lower().split())

    # Text files dhundo
    try:
        files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    except Exception as e:
        return ""
    
    # Filename Match Logic
    for filename in files:
        score = 0
        name_parts = filename.lower().replace('.txt', '').split('_')
        for part in name_parts:
            if part in query_words:
                score += 5 
        
        if score > best_score:
            best_score = score
            best_file = filename
    
    # Fallback: Agar match na ho to pehli file (Temporary)
    target_file = best_file if best_file else (files[0] if files else None)
    
    context = ""
    if target_file:
        try:
            with open(os.path.join(folder_path, target_file), "r", encoding="utf-8") as f:
                # 8000 characters limit for speed
                context = f.read()[:8000] 
                print(f"📖 Using Context from: {target_file}")
        except Exception as e:
            print(f"Error reading file: {e}")
    
    return context

@app.post("/talk")
async def talk_to_ai(file: UploadFile = File(...)):
    # Pehle check karo key hai ya nahi
    if not client:
        raise HTTPException(status_code=500, detail="Server Error: API Key missing.")

    session_id = str(uuid.uuid4())
    input_audio = f"temp_in_{session_id}.webm"
    output_audio = f"temp_out_{session_id}.mp3"

    try:
        # 1. User Audio Save
        with open(input_audio, "wb") as buffer:
            buffer.write(await file.read())

        # 2. Speech to Text (Sunna)
        with open(input_audio, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=(input_audio, f.read()),
                model="whisper-large-v3",
                language="ur" 
            )
        user_text = transcription.text
        print(f"🗣️ User: {user_text}")

        if not user_text or not user_text.strip():
            return {"error": "Kuch sunayi nahi diya"}

        # 3. Knowledge Fetch
        context_text = find_relevant_context(user_text)

        # 4. AI Brain (Sochna)
        system_prompt = f"""
        You are a helpful teacher named 'Sir AI'. 
        You teach CA students. Speak in a mix of Urdu and English (Roman Urdu).
        Keep answers short (2-3 sentences max) and conversational.
        
        Use this knowledge to answer:
        {context_text}
        """

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            model="llama3-8b-8192",
            temperature=0.6,
        )
        ai_reply = chat_completion.choices[0].message.content
        print(f"🤖 AI: {ai_reply}")

        # 5. Text to Speech (Bolna)
        communicate = edge_tts.Communicate(ai_reply, "ur-PK-AsadNeural")
        await communicate.save(output_audio)

        return FileResponse(output_audio, media_type="audio/mpeg", filename="reply.mp3")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Safai
        if os.path.exists(input_audio):
            os.remove(input_audio)
        # Output audio file Render khud clean kar deta hai baad mein, 
        # ya hum background task laga sakte hain, par abhi simple rakhein.

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)