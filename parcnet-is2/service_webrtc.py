# service_webrtc.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import numpy as np
import librosa
import yaml
import asyncio
import torch
import os
import requests
from aiortc import (
    RTCPeerConnection,
    RTCConfiguration,
    RTCIceServer,
    RTCSessionDescription,
    MediaStreamTrack
)
from aiortc.contrib.media import MediaBlackhole

from parcnet import PARCnet

# Constantes
SR = 44100
LOSS_THRESHOLD = 0.10  # 10%

# Configura FastAPI y CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Carga configuración de PARCnet
with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)

parcnet = PARCnet(
    model_checkpoint=cfg["inference"]["model_checkpoint"],
    packet_dim=cfg["global"]["packet_dim"],
    extra_pred_dim=cfg["global"]["extra_pred_dim"],
    ar_order=cfg["AR"]["ar_order"],
    ar_diagonal_load=cfg["AR"]["diagonal_load"],
    ar_context_dim=cfg["AR"]["ar_context_dim"],
    nn_context_dim=cfg["neural_net"]["nn_context_dim"],
    nn_fade_dim=cfg["neural_net"]["fade_dim"],
    device="cuda" if torch.cuda.is_available() else "cpu",
    lite=cfg["neural_net"]["lite"],
)

def fetch_xirsys_ice():
    url = "https://global.xirsys.net/_turn/musicnet-demo"
    payload = {
        "ident": "ValeLopezCubillos",
        "secret": "08b07ede-306d-11f0-83bd-0242ac150002",
        "channel": "musicnet-demo"
    }
    res = requests.put(url, json=payload, timeout=5)
    res.raise_for_status()
    servers = res.json()["v"]["iceServers"]
    # Convierte a RTCIceServer
    return [
        RTCIceServer(
            urls=entry["urls"],
            username=entry.get("username"),
            credential=entry.get("credential")
        )
        for entry in servers
    ]

# 2) Antes de crear tu endpoint, obtén iceServers
ice_servers = fetch_xirsys_ice()
rtc_config  = RTCConfiguration(iceServers=ice_servers)

@app.post("/offer")
async def offer(request: Request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    # 1. Crea PeerConnection con ICE
    pc = RTCPeerConnection(rtc_config)
    dc = pc.createDataChannel("control")
    media_blackhole = MediaBlackhole()

    # 2. Estadísticas de pérdida
    stats = {"packetsLost": 0, "packetsReceived": 0, "loss_rate": 0.0}

    async def stats_loop():
        while True:
            report = await pc.getStats()
            for s in report.values():
                if s.type == "inbound-rtp" and s.kind == "audio":
                    stats["packetsLost"] = s.packetsLost
                    stats["packetsReceived"] = s.packetsReceived
                    if stats["packetsReceived"] > 0:
                        stats["loss_rate"] = stats["packetsLost"] / stats["packetsReceived"]
            await asyncio.sleep(1.0)

    asyncio.create_task(stats_loop())

    # 3. Procesamiento de pista de audio
    @pc.on("track")
    async def on_track(track: MediaStreamTrack):
        if track.kind != "audio":
            return

        buffer = []
        applied_parcnet = False

        try:
            while True:
                try:
                    frame = await track.recv()
                except Exception:
                    break

                # Normaliza y resamplea
                pcm = frame.to_ndarray()[0].astype(np.float32) / 32768.0
                if frame.sample_rate != SR:
                    pcm = librosa.resample(pcm, orig_sr=frame.sample_rate, target_sr=SR)

                buffer.append(pcm)

                # Aplica PARCnet si la pérdida supera el umbral
                if stats["loss_rate"] > LOSS_THRESHOLD:
                    full_signal = np.concatenate(buffer)
                    trace = np.zeros(len(full_signal) // cfg["global"]["packet_dim"], dtype=int)
                    enhanced = parcnet(full_signal, trace)
                    window = enhanced[-len(pcm):]
                    buffer = []
                    applied_parcnet = True
                else:
                    window = pcm
                    applied_parcnet = False

                # Detección de pitch
                f0 = librosa.yin(
                    y=window,
                    fmin=librosa.note_to_hz("C2"),
                    fmax=librosa.note_to_hz("C7"),
                    sr=SR,
                    frame_length=2048,
                    hop_length=512,
                )
                f0_clean = f0[~np.isnan(f0)]
                if len(f0_clean) == 0:
                    continue
                frequency = float(np.median(f0_clean))
                midi = librosa.hz_to_midi(frequency)
                midi_corrected = int(np.round(midi)) + 12
                note_name = librosa.midi_to_note(midi_corrected)

                # Log en consola solo si pasó por PARCnet
                if applied_parcnet:
                    print(
                        f"[LOSS_RATE: {stats['loss_rate']*100:.1f}%] "
                        f"[PARCnet applied: True] "
                        f"[Detected note: {note_name} @ {frequency:.1f} Hz]"
                    )

                # Envía datos al cliente
                dc.send(json.dumps({
                    "note": note_name,
                    "frequency": frequency,
                    "loss_rate": stats["loss_rate"]
                }))

        finally:
            track.stop()
            await media_blackhole.start()

    # 4. Intercambio SDP y ICE
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    # Espera a que termine ICE gathering
    while pc.iceGatheringState != "complete":
        await asyncio.sleep(0.1)

    return JSONResponse({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })
