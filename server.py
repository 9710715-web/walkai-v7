import os
import base64
from pathlib import Path
import httpx
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, Response, JSONResponse
from pydantic import BaseModel

BASE = Path(__file__).resolve().parent
app = FastAPI(title='WALKAI V7')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

class TTSRequest(BaseModel):
    text: str

@app.get('/')
async def index():
    return FileResponse(BASE / 'index.html')

@app.post('/api/tts')
async def tts(req: TTSRequest):
    if not OPENAI_API_KEY:
        return JSONResponse({'error':'OPENAI_API_KEY is not configured'}, status_code=503)
    payload={'model':'gpt-4o-mini-tts','voice':'marin','input':req.text[:4096], 'instructions':'Speak in natural Russian as a warm, intelligent local city guide. Calm, expressive, conversational, with short pauses. Never sound like a robotic announcement.','response_format':'mp3','speed':0.96}
    async with httpx.AsyncClient(timeout=60) as client:
        r=await client.post('https://api.openai.com/v1/audio/speech',headers={'Authorization':f'Bearer {OPENAI_API_KEY}'},json=payload)
    if r.status_code>=400:return Response(r.content,status_code=r.status_code,media_type=r.headers.get('content-type','application/json'))
    return Response(r.content,media_type='audio/mpeg',headers={'Cache-Control':'no-store'})

async def text_answer(prompt: str):
    async with httpx.AsyncClient(timeout=90) as client:
        r=await client.post('https://api.openai.com/v1/responses',headers={'Authorization':f'Bearer {OPENAI_API_KEY}','Content-Type':'application/json'},json={'model':'gpt-5.6-luna','input':prompt,'max_output_tokens':350})
    if r.status_code>=400:return Response(r.content,status_code=r.status_code,media_type=r.headers.get('content-type','application/json'))
    j=r.json();answer=j.get('output_text','')
    if not answer:
        for item in j.get('output',[]):
            for c in item.get('content',[]):
                if c.get('type')=='output_text':answer+=c.get('text','')
    return {'answer':answer.strip()}

@app.post('/api/ask-text')
async def ask_text(text: str=Form(...),place: str=Form(''),context: str=Form('')):
    if not OPENAI_API_KEY:return JSONResponse({'error':'OPENAI_API_KEY is not configured'},status_code=503)
    prompt=f"You are WALKAI, a knowledgeable local guide in Istanbul. Answer the visitor naturally and briefly in Russian, usually 2-5 sentences. Current place: {place}. Verified context: {context}. Visitor question: {text}. Clearly distinguish established facts from legends or uncertain stories. Do not invent details."
    out=await text_answer(prompt)
    if isinstance(out,Response):return out
    out['transcript']=text
    return out

@app.post('/api/ask')
async def ask(audio: UploadFile=File(...),place: str=Form(''),context: str=Form('')):
    if not OPENAI_API_KEY:return JSONResponse({'error':'OPENAI_API_KEY is not configured'},status_code=503)
    audio_bytes=await audio.read()
    files={'file':('question.webm',audio_bytes,'audio/webm')}
    data={'model':'gpt-4o-mini-transcribe','language':'ru','response_format':'json','prompt':f'Current place: {place}. Historical context: {context}'}
    async with httpx.AsyncClient(timeout=90) as client:
        tr=await client.post('https://api.openai.com/v1/audio/transcriptions',headers={'Authorization':f'Bearer {OPENAI_API_KEY}'},files=files,data=data)
    if tr.status_code>=400:return Response(tr.content,status_code=tr.status_code,media_type=tr.headers.get('content-type','application/json'))
    transcript=tr.json().get('text','')
    return await ask_text(transcript,place,context)

@app.post('/api/vision')
async def vision(image: UploadFile=File(...),place: str=Form(''),context: str=Form('')):
    if not OPENAI_API_KEY:return JSONResponse({'error':'OPENAI_API_KEY is not configured'},status_code=503)
    raw=await image.read()
    if len(raw)>12*1024*1024:return JSONResponse({'error':'Image is too large'},status_code=413)
    mime=image.content_type or 'image/jpeg'
    b64=base64.b64encode(raw).decode('ascii')
    prompt=f"You are WALKAI, an AI visual city guide in Istanbul. Examine the visitor photo carefully. Identify the most likely visible building, monument, street, artwork, sign, landscape or object. Return EXACTLY this simple format: first line starts with 'TITLE:' followed by a short Russian name; second line starts with 'TEXT:' followed by 3-5 concise Russian sentences explaining what it is, why it matters and what the visitor should notice. If the image is insufficient to identify it reliably, say so clearly and do not invent. Current route stop: {place}. Verified context: {context}."
    payload={'model':'gpt-5.6-luna','input':[{'role':'user','content':[{'type':'input_text','text':prompt},{'type':'input_image','image_url':f'data:{mime};base64,{b64}'}]}],'max_output_tokens':350}
    async with httpx.AsyncClient(timeout=90) as client:
        r=await client.post('https://api.openai.com/v1/responses',headers={'Authorization':f'Bearer {OPENAI_API_KEY}','Content-Type':'application/json'},json=payload)
    if r.status_code>=400:return Response(r.content,status_code=r.status_code,media_type=r.headers.get('content-type','application/json'))
    j=r.json();answer=j.get('output_text','')
    if not answer:
        for item in j.get('output',[]):
            for c in item.get('content',[]):
                if c.get('type')=='output_text':answer+=c.get('text','')
    answer=answer.strip()
    title='Объект на фото'
    description=answer
    lines=answer.splitlines()
    for idx,line in enumerate(lines):
        if line.strip().upper().startswith('TITLE:'):
            title=line.split(':',1)[1].strip() or title
        if line.strip().upper().startswith('TEXT:'):
            description=' '.join([x.strip() for x in lines[idx:]])
            description=description.split(':',1)[1].strip() if ':' in description else description
            break
    return {'title':title,'description':description.strip()}

@app.post('/api/vision-detail')
async def vision_detail(title: str=Form(...),description: str=Form(''),place: str=Form(''),context: str=Form('')):
    if not OPENAI_API_KEY:return JSONResponse({'error':'OPENAI_API_KEY is not configured'},status_code=503)
    prompt=f"You are WALKAI, a professional local guide in Istanbul. Give a more detailed but easy-to-listen Russian explanation about the object '{title}'. Existing identification/description: {description}. Current route stop: {place}. Verified context: {context}. Answer in 7-10 concise sentences, adding useful history, architecture, details to look at and one memorable fact. Distinguish legends from established facts and do not invent uncertain details. This text will be read aloud, so make it natural and conversational."
    out=await text_answer(prompt)
    if isinstance(out,Response):return out
    return {'title':title,'description':out.get('answer','').strip()}
