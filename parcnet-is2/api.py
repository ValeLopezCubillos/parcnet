# api.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import numpy as np
import soundfile as sf
import torch
import uvicorn
import tempfile
from io import BytesIO
from pathlib import Path
from parcnet import PARCnet

app = FastAPI()

# Inicializar PARCnet
parcnet = PARCnet(
    model_checkpoint="pretrained_models/parcnet-is2_baseline_checkpoint.ckpt",
    packet_dim=512,
    extra_pred_dim=256,
    ar_order=256,
    ar_diagonal_load=0.01,
    ar_context_dim=8,
    nn_context_dim=8,
    nn_fade_dim=64,
    device="cuda" if torch.cuda.is_available() else "cpu",
    lite=True,
)

def detect_loss_trace(audio: np.ndarray, packet_size: int = 512, silence_threshold: float = 1e-4) -> np.ndarray:
    """Detecta paquetes perdidos por umbral de silencio."""
    num_packets = len(audio) // packet_size
    trace = np.ones(num_packets, dtype=int)
    for i in range(num_packets):
        packet = audio[i * packet_size: (i + 1) * packet_size]
        if np.all(np.abs(packet) < silence_threshold):
            trace[i] = 0
    return trace

@app.post("/parcnet")
async def enhance_audio(audio: UploadFile = File(...)):
    # Cargar audio
    raw = await audio.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
        tmp_audio.write(raw)
        tmp_audio_path = tmp_audio.name

    signal, sr = sf.read(tmp_audio_path)
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)  # convertir a mono
    signal = signal.astype(np.float32)

    # Detectar traza
    trace = detect_loss_trace(signal)

    # Reconstrucción
    enhanced = parcnet(signal, trace)

    # Guardar en buffer
    buffer = BytesIO()
    sf.write(buffer, enhanced, sr, format='WAV')
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="audio/wav", headers={"Content-Disposition": "inline; filename=enhanced.wav"})

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8001)

