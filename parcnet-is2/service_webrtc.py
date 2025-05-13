from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import numpy as np
import librosa
import yaml
import asyncio
import torch

from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole

from parcnet import PARCnet

SR = 44100
LOSS_THRESHOLD = 0.10  # 10%

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

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

@app.post("/offer")
async def offer(request: Request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    dc = pc.createDataChannel("control")
    media_blackhole = MediaBlackhole()

    stats = {"packetsLost": 0, "packetsReceived": 0, "loss_rate": 0.0}

    async def stats_loop():
        """Cada segundo consulta getStats() y actualiza stats['loss_rate']."""
        while True:
            report = await pc.getStats()
            for s in report.values():
                if s.type == "inbound-rtp" and s.kind == "audio":
                    # Actualiza recuentos
                    stats["packetsLost"] = s.packetsLost
                    stats["packetsReceived"] = s.packetsReceived
                    if stats["packetsReceived"] > 0:
                        stats["loss_rate"] = stats["packetsLost"] / stats["packetsReceived"]
            await asyncio.sleep(1.0)

    asyncio.create_task(stats_loop())

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

                pcm = frame.to_ndarray()[0].astype(np.float32) / 32768.0
                if frame.sample_rate != SR:
                    pcm = librosa.resample(pcm, orig_sr=frame.sample_rate, target_sr=SR)

                buffer.append(pcm)
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

                if applied_parcnet:
                    print(f"[LOSS_RATE: {stats['loss_rate']*100:.1f}%] "
                          f"[PARCnet applied: True] "
                          f"[Detected note: {note_name} @ {frequency:.1f} Hz]")

                dc.send(json.dumps({
                    "note": note_name,
                    "frequency": frequency,
                    "loss_rate": stats["loss_rate"]
                }))

        finally:
            track.stop()
            await media_blackhole.start()

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return JSONResponse({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })

