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

# FastAPI + CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Carga modelo PARCnet
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
    ident  = "ValeLopezCubillos"
    secret = "08b07ede-306d-11f0-83bd-0242ac150002"
    url    = f"https://ValeLopezCubillos:08b07ede-306d-11f0-83bd-0242ac150002@global.xirsys.net/_turn/musicnet-demo"

    res = requests.put(
        url,
        headers={"Content-Type": "application/json"},
        json={"format": "urls"},
        timeout=5
    )
    res.raise_for_status()
    resp = res.json()
    container = resp.get("v") or resp.get("d")
    ice_data = container["iceServers"]

    # Asegura lista
    ice_entries = [ice_data] if isinstance(ice_data, dict) else ice_data

    # Devuelve instancias RTCIceServer
    return [
        RTCIceServer(
            urls=entry["urls"],
            username=entry.get("username"),
            credential=entry.get("credential")
        )
        for entry in ice_entries
    ]

# Construye configuración ICE
ice_servers = fetch_xirsys_ice()
rtc_config  = RTCConfiguration(iceServers=ice_servers)

@app.get("/ice")
def get_ice():
    # para debugging o cliente que pida ICE
    return {"iceServers": [
        {
            "urls": s.urls,
            "username": s.username,
            "credential": s.credential
        } for s in fetch_xirsys_ice()
    ]}

@app.post("/offer")
async def offer(request: Request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    # 1) crea PeerConnection con ICE dinámico
    pc = RTCPeerConnection(rtc_config)
    media_blackhole = MediaBlackhole()

    # 1.a) variable para triggers desde el cliente
    pending_trigger = False

    # 1.b) crea proactivamente un DataChannel en el servidor
    dc = pc.createDataChannel("control")
    print("📡 [SERVER] DataChannel creado proactivamente")

    @dc.on("open")
    def _():
        print("🟢 [SERVER] DataChannel READY")

    @dc.on("message")
    def _on_message(msg):
        print("📨 [SERVER] Mensaje recibido por DataChannel:", msg)
        # quizá el cliente envíe comandos también por aquí
        # pending_trigger = True  # si quisieras usarlo

    @dc.on("close")
    def _():
        print("⚠️ [SERVER] DataChannel cerrado por el cliente")

    # 2) logging ICE y conexión
    @pc.on("iceconnectionstatechange")
    def on_ice_state():
        print("🔄 [SERVER] ICE state:", pc.iceConnectionState)

    @pc.on("connectionstatechange")
    def on_conn_state():
        print("🛡️ [SERVER] Connection state:", pc.connectionState)

    # 3) captura el DataChannel que abre el cliente (por si importan los labels del cliente)
    @pc.on("datachannel")
    def on_datachannel(channel):
        nonlocal dc, pending_trigger
        dc = channel
        print("🟢 [SERVER] DataChannel abierto por el cliente")

        @dc.on("open")
        def _():
            print("🟢 [SERVER] DataChannel READY")

        @dc.on("message")
        def _on_trigger(msg):
            print("📨 [SERVER] Trigger recibido:", msg)
            pending_trigger = True

        @dc.on("close")
        def _():
            print("⚠️ [SERVER] DataChannel cerrado por el cliente")

    # 4) estadísticas de pérdida (igual que antes)…
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

    # 5) procesamiento de audio y detección de nota
    @pc.on("track")
    async def on_track(track: MediaStreamTrack):
        if track.kind != "audio":
            return

        buffer = []
        applied_parcnet = False

        try:
            while True:
                frame = await track.recv()
                pcm = frame.to_ndarray()[0].astype(np.float32) / 32768.0
                if frame.sample_rate != SR:
                    pcm = librosa.resample(pcm, orig_sr=frame.sample_rate, target_sr=SR)

                buffer.append(pcm)

                if pending_trigger:
                    pending_trigger = False
                    full_signal = np.concatenate(buffer)
                    trace = np.zeros(len(full_signal) // cfg["global"]["packet_dim"], dtype=int)
                    if stats["loss_rate"] > LOSS_THRESHOLD:
                        enhanced = parcnet(full_signal, trace)
                        window = enhanced[-len(pcm):]
                        applied_parcnet = True
                    else:
                        window = pcm
                        applied_parcnet = False

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
                        buffer.clear()
                        continue

                    frequency = float(np.median(f0_clean))
                    midi = librosa.hz_to_midi(frequency)
                    midi_corrected = int(np.round(midi)) + 12
                    note_name = librosa.midi_to_note(midi_corrected)

                    if applied_parcnet:
                        print(
                            f"[LOSS_RATE: {stats['loss_rate']*100:.1f}%] "
                            f"[PARCnet applied: True] "
                            f"[Detected note: {note_name} @ {frequency:.1f} Hz]"
                        )

                    if dc and dc.readyState == "open":
                        dc.send(json.dumps({
                            "note": note_name,
                            "frequency": frequency,
                            "loss_rate": stats["loss_rate"]
                        }))

                    buffer.clear()

        finally:
            track.stop()
            await media_blackhole.start()

    # 6) intercambio SDP
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    # 7) espera a que ICE termine de gather
    while pc.iceGatheringState != "complete":
        await asyncio.sleep(0.1)

    return JSONResponse({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })

