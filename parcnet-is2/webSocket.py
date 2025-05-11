# webSocket.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import torch
import librosa

from parcnet import PARCnet  

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SAMPLE_RATE = 16000
PACKET_SIZE = 512        

parcnet = PARCnet(
    model_checkpoint="pretrained_models/parcnet-is2_baseline_checkpoint.ckpt",
    packet_dim=PACKET_SIZE,
    extra_pred_dim=256,
    ar_order=256,
    ar_diagonal_load=0.01,
    ar_context_dim=8,
    nn_context_dim=8,
    nn_fade_dim=64,
    device="cuda" if torch.cuda.is_available() else "cpu",
    lite=True,
)  

def detect_loss_trace(audio: np.ndarray, packet_size: int = PACKET_SIZE,
                      silence_threshold: float = 1e-4) -> np.ndarray:
    num_packets = len(audio) // packet_size
    trace = np.ones(num_packets, dtype=int)
    for i in range(num_packets):
        pkt = audio[i * packet_size : (i + 1) * packet_size]
        if np.all(np.abs(pkt) < silence_threshold):
            trace[i] = 0
    return trace  

NOTE_STRINGS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def get_note_from_freq(freq: float):
    if freq <= 0.0:
        return None
    A4 = 440.0
    note_number = 69 + 12 * np.log2(freq / A4)
    midi = int(round(note_number))
    note_name = NOTE_STRINGS[midi % 12]
    octave = midi // 12 - 1
    return f"{note_name}{octave}" 

def detect_note(audio: np.ndarray, sample_rate: int = SAMPLE_RATE):
    pitches = librosa.yin(
        audio,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sample_rate
    )
    freq = float(np.median(pitches))
    note = get_note_from_freq(freq)
    return note, freq

@app.websocket("/ws/parcnet")
async def websocket_parcnet(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            chunk = await ws.receive_bytes()
            audio = np.frombuffer(chunk, dtype=np.float32)
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            trace = detect_loss_trace(audio)
            enhanced = parcnet(audio, trace)
            enhanced = np.nan_to_num(enhanced, nan=0.0, posinf=0.0, neginf=0.0)
            note_name, frequency = detect_note(enhanced)
            await ws.send_bytes(enhanced.astype(np.float32).tobytes())
            await ws.send_json({
                "note": note_name,
                "frequency": frequency
            })

    except WebSocketDisconnect:
        print("Cliente desconectado")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("webSocket:app", host="0.0.0.0", port=8081)
