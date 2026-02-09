import os
import uuid
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from groq import Groq
import edge_tts
import asyncio

# --- CONFIGURATION ---
# Aapki API Key yahan set kar di gayi hai
GROQ_API_KEY = "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"

app = FastAPI()

# Security: Sab allow kar rahe hain taake local testing mein masla na aye
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=GROQ_API_KEY)

# --- SMART SEARCH (RAG Lite) ---
def find_relevant_context(query):
    """
    Yeh function student ke sawal (query) se milta julta chapter
    dhund kar sirf uska text AI ko dega.
    """
    folder_path = "knowledge_base"
    best_file = None
    best_score = 0
    
    # Agar folder hi nahi hai to wapis jao
    if not os.path.exists(folder_path):
        return "Mere paas abhi koi kitab (knowledge) nahi hai."

    query_words = set(query.lower().split())

    # Har text file ko check karo
    files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    
    # Simple Search: Filename match
    for filename in files:
        score = 0
        name_parts = filename.lower().replace('.txt', '').split('_')
        for part in name_parts:
            if part in query_words:
                score += 5 
        
        if score > best_score:
            best_score = score
            best_file = filename
    
    # Agar koi specific file na mile, to pehli file utha lo (Temporary)
    target_file = best_file if best_file else (files[0] if files else None)
    
    context = ""
    if target_file:
        try:
            with open(os.path.join(folder_path, target_file), "r", encoding="utf-8") as f:
                # Sirf pehle 8,000 characters lo taake speed tez rahe
                context = f.read()[:8000] 
                print(f"📖 Using Context from: {target_file}")
        except Exception as e:
            print(f"Error reading file: {e}")
    
    return context

@app.post("/talk")
async def talk_to_ai(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    input_audio = f"temp_in_{session_id}.webm"
    output_audio = f"temp_out_{session_id}.mp3"

    try:
        # 1. User ki awaz save karo
        with open(input_audio, "wb") as buffer:
            buffer.write(await file.read())

        # 2. Speech to Text (Whisper) - Awaz ko likhai mein badlo
        with open(input_audio, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=(input_audio, f.read()),
                model="whisper-large-v3",
                language="ur" # Urdu focus
            )
        user_text = transcription.text
        print(f"🗣️ User said: {user_text}")

        if not user_text or not user_text.strip():
            return {"error": "Kuch sunayi nahi diya"}

        # 3. Relevant Knowledge Dhundo
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
        print(f"🤖 AI replied: {ai_reply}")

        # 5. Text to Speech (Bolna) - Urdu Voice
        # Voices: ur-PK-AsadNeural (Male) or ur-PK-UzmaNeural (Female)
        communicate = edge_tts.Communicate(ai_reply, "ur-PK-AsadNeural")
        await communicate.save(output_audio)

        return FileResponse(output_audio, media_type="audio/mpeg", filename="reply.mp3")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Safai (Cleanup) - Input file delete kar do
        if os.path.exists(input_audio):
            os.remove(input_audio)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)