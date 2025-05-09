from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import soundfile as sf
import torch
import tempfile
import uvicorn
import librosa
import traceback
from parcnet import PARCnet
from io import BytesIO
import subprocess
from scipy.signal import firwin, lfilter

app = FastAPI()

# Permitir acceso desde frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes restringir esto a tu dominio si lo deseas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

NOTE_STRINGS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def detect_loss_trace(audio: np.ndarray, packet_size: int = 512, silence_threshold: float = 1e-4) -> np.ndarray:
    """Detectar pérdida por bloques de silencio"""
    num_packets = len(audio) // packet_size
    trace = np.ones(num_packets, dtype=int)
    for i in range(num_packets):
        packet = audio[i * packet_size: (i + 1) * packet_size]
        if np.all(np.abs(packet) < silence_threshold):
            trace[i] = 0
    return trace

def get_note_from_freq(freq: float):
    """Convertir frecuencia en nota musical"""
    if freq <= 0.0:
        return None
    A4 = 440.0
    note_number = 69 + 12 * np.log2(freq / A4)
    midi = int(round(note_number))
    note_name = NOTE_STRINGS[midi % 12]
    octave = midi // 12 - 1
    return f"{note_name}{octave}"

@app.post("/detect_note2")
async def detect_note(audio: UploadFile = File(...)):
    try:
        raw = await audio.read()
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_webm:
            tmp_webm.write(raw)
            tmp_webm_path = tmp_webm.name

        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_wav_path = tmp_wav.name
        tmp_wav.close()


        ffmpeg_cmd = [
            "C:/ffmpeg/ffmpeg-7.1.1-essentials_build/bin/ffmpeg.exe", "-y",
            #"ffmpeg", "-y",
            "-i", tmp_webm_path,
            "-ar", "16000",  
            "-ac", "1",      
            tmp_wav_path
        ]
        subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        signal, sr = sf.read(tmp_wav_path)
        if signal.ndim > 1:
            signal = np.mean(signal, axis=1)
        signal = signal.astype(np.float32)

        trace = detect_loss_trace(signal)
        loss_ratio = np.mean(trace == 0)
        if loss_ratio > 0.01:
            print(f"Pérdida detectada: {loss_ratio*100:.2f}%")
            print("Ejecutando PARCnet para reconstrucción...")
            signal = parcnet(signal, trace)

        pitches = librosa.yin(signal, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
        freq = float(np.median(pitches))  
        note = get_note_from_freq(freq)

        return JSONResponse(content={"note": note, "frequency": freq})

    except Exception as e:
        print("❌ Error en /detect_note:", str(e))
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/detect_note")
async def detect_note(audio: UploadFile = File(...)):
    try:
        raw = await audio.read()
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_webm:
            tmp_webm.write(raw)
            tmp_webm_path = tmp_webm.name

        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_wav_path = tmp_wav.name
        tmp_wav.close()

        ffmpeg_cmd = [
            #"C:/ffmpeg/ffmpeg-7.1.1-essentials_build/bin/ffmpeg.exe", "-y",
            "ffmpeg", "-y",
            "-i", tmp_webm_path,
            "-ar", "16000",
            "-ac", "1",
            tmp_wav_path
        ]
        subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        signal, sr = sf.read(tmp_wav_path)
        if signal.ndim > 1:
            signal = np.mean(signal, axis=1)
        signal = signal.astype(np.float32)

        # 📌 Simular pérdida 
        packet_size = 512
        num_packets = len(signal) // packet_size
        loss_percent = 0.10  #10% de pérdida
        num_lost_packets = int(num_packets * loss_percent)
        loss_mask = np.ones(num_packets, dtype=int)
        lost_indices = np.random.choice(num_packets, num_lost_packets, replace=False)

        for i in lost_indices:
            loss_mask[i] = 0
            signal[i * packet_size:(i + 1) * packet_size] = 0.0

        trace = loss_mask
        loss_ratio = np.mean(trace == 0)
        print(f"🔧 Pérdida simulada: {loss_ratio*100:.2f}%, paquetes perdidos: {np.sum(trace == 0)}")
        pitches_before = librosa.yin(signal, fmin=librosa.note_to_hz('C2'),
                                     fmax=librosa.note_to_hz('C7'), sr=sr)
        freq_before = float(np.median(pitches_before))
        note_before = get_note_from_freq(freq_before)
        print(f"🕵️ Nota antes de PARCnet: {note_before} ({freq_before:.2f} Hz)")

        if loss_ratio > 0.01:
            print("⚙️ Ejecutando PARCnet para reconstrucción...")
            enhanced = parcnet(signal, trace)
            for i, lost in enumerate(trace):
                if lost == 0:
                    start = i * packet_size
                    end = start + packet_size
                    signal[start:end] = enhanced[start:end]
        pitches = librosa.yin(signal, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
        freq = float(np.median(pitches))
        note = get_note_from_freq(freq)

        print(f"🎵 Nota detectada despues de PARCnet: {note} ({freq:.2f} Hz)")

        return JSONResponse(content={"note": note, "frequency": freq})

    except Exception as e:
        print("❌ Error en /detect_note:", str(e))
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/parcnet")
async def enhance_audio(audio: UploadFile = File(...)):
    # Cargar audio
    raw = await audio.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
        tmp_audio.write(raw)
        tmp_audio_path = tmp_audio.name

    signal, sr = sf.read(tmp_audio_path)
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)  
    signal = signal.astype(np.float32)

    trace = detect_loss_trace(signal)

    enhanced = parcnet(signal, trace)

    buffer = BytesIO()
    sf.write(buffer, enhanced, sr, format='WAV')
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="audio/wav", headers={"Content-Disposition": "inline; filename=enhanced.wav"})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8081))
    uvicorn.run("api:app", host="0.0.0.0", port=port)


