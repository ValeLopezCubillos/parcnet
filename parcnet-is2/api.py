from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse  # Importación añadida
from parcnet import PARCnet
import numpy as np
import librosa
import io
import soundfile as sf  # Importación añadida
import torch  # Importación añadida
from pathlib import Path
from fastapi.responses import StreamingResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar modelo
model = PARCnet(
    model_checkpoint=Path("pretrained_models/parcnet-is2_baseline_checkpoint.ckpt"),
    packet_dim=512,
    extra_pred_dim=256,
    ar_order=256,
    ar_diagonal_load=0.01,
    ar_context_dim=8,
    nn_context_dim=8,
    nn_fade_dim=64,
    device="cuda" if torch.cuda.is_available() else "cpu",
    lite=True
)

@app.post("/parcnet")
async def enhance_audio(file: UploadFile = File(...)):
    # Cargar audio
    audio, sr = librosa.load(io.BytesIO(await file.read()), sr=44100)
    
    # Simular pérdida de paquetes
    trace = np.random.randint(0, 2, size=len(audio)//512)
    
    # Mejorar audio
    enhanced = model(audio, trace)
    
    # Crear archivo en memoria
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, enhanced, sr, format='WAV')
    audio_buffer.seek(0)
    
    # Devolver como stream
    return StreamingResponse(audio_buffer, media_type="audio/wav", 
                           headers={"Content-Disposition": "attachment; filename=enhanced.wav"})

