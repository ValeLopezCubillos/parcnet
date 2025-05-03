from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from parcnet import PARCnet
import numpy as np
import librosa
import io
import soundfile as sf
import torch
from pathlib import Path
import logging
from scipy.signal import butter, lfilter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PARCnet Audio Reconstruction API",
             description="API for audio reconstruction with real packet loss")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
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
    logger.info(f"Model loaded on {'CUDA' if torch.cuda.is_available() else 'CPU'}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise RuntimeError("Could not initialize model") from e

def apply_lowpass_filter(audio, sr, cutoff=18000, order=5):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, audio)

@app.get("/")
def read_root():
    return {
        "message": "PARCnet API working",
        "endpoints": {
            "POST /parcnet": "Reconstruct audio with real losses",
            "POST /parcnet-with-trace": "Reconstruct audio with provided loss mask"
        }
    }

@app.post("/parcnet2")
async def enhance_audio_simple(file: UploadFile = File(...)):
    try:
        logger.info(f"Processing file: {file.filename}")
        file_content = await file.read()
        try:
            audio, sr = librosa.load(io.BytesIO(file_content), sr=44100, mono=True)
        except Exception as e:
            logger.error(f"Error loading audio: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail="Unsupported audio format. Use mono WAV files at 44.1kHz"
            )

        packet_size = 512
        original_length = len(audio)
        if len(audio) % packet_size != 0:
            audio = audio[:-(len(audio) % packet_size)]
            logger.warning(f"Adjusted length from {original_length} to {len(audio)} samples")
        trace = (np.abs(audio) > 1e-6).astype(np.float32)
        loss_percentage = np.mean(trace == 0) * 100
        
        if len(audio) == 0:
            logger.error("Empty audio after preprocessing")
            raise HTTPException(
                status_code=400,
                detail="Audio was empty after preprocessing"
            )
        try:
            logger.info(f"Starting reconstruction: {len(audio)} samples, {loss_percentage:.1f}% losses")
            
            trace = trace[:len(audio) // packet_size * packet_size]
            trace = trace.reshape(-1, packet_size).any(axis=1).astype(np.float32)
            
            enhanced = model(audio, trace)
            
            if len(enhanced) != len(audio):
                logger.error(f"Dimensional error: input {len(audio)} vs output {len(enhanced)}")
                raise ValueError("Reconstructed audio has different length than original")
                
        except Exception as e:
            logger.error(f"Reconstruction error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error during reconstruction: {str(e)}"
            )
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, enhanced, sr, format='WAV', subtype='PCM_16')
        audio_buffer.seek(0)

        logger.info("Reconstruction successful")
        return StreamingResponse(
            audio_buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=reconstructed_{file.filename}",
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal error processing request"
        )
    
@app.post("/detect_loss")
async def detect_packet_loss(file: UploadFile = File(...)):
    try:
        file_content = await file.read()
        audio, sr = librosa.load(io.BytesIO(file_content), sr=44100, mono=True)

        packet_size = 512
        if len(audio) % packet_size != 0:
            audio = audio[:-(len(audio) % packet_size)]
        
        trace = (np.abs(audio) > 1e-6).astype(np.float32)
        loss_percentage = np.mean(trace == 0) * 100

        return {"loss_percentage": loss_percentage}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/parcnet")
async def enhance_audio(file: UploadFile = File(...)):
    try:
        logger.info(f"Processing file: {file.filename}")
        file_content = await file.read()
        
        try:
            audio, sr = librosa.load(io.BytesIO(file_content), sr=44100, mono=True)
        except Exception as e:
            logger.error(f"Error loading audio: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail="Unsupported audio format. Use mono WAV files at 44.1kHz"
            )

        packet_size = 512
        original_length = len(audio)
        
        if len(audio) % packet_size != 0:
            audio = audio[:-(len(audio) % packet_size)]
            logger.warning(f"Adjusted length from {original_length} to {len(audio)} samples")

        trace = (np.abs(audio) > 1e-6).astype(np.float32)
        loss_percentage = np.mean(trace == 0) * 100
        
        if loss_percentage < 0.1:
            logger.warning("Audio does not contain significant losses (<0.1%)")
            raise HTTPException(
                status_code=400,
                detail="Audio does not contain enough losses to reconstruct (<0.1%)"
            )

        if len(audio) == 0:
            logger.error("Empty audio after preprocessing")
            raise HTTPException(
                status_code=400,
                detail="Audio was empty after preprocessing"
            )

        try:
            logger.info(f"Starting reconstruction: {len(audio)} samples, {loss_percentage:.1f}% losses")
            
            trace = trace[:len(audio) // packet_size * packet_size]
            trace = trace.reshape(-1, packet_size).any(axis=1).astype(np.float32)
            
            enhanced = model(audio, trace)
            
            if len(enhanced) != len(audio):
                logger.error(f"Dimensional error: input {len(audio)} vs output {len(enhanced)}")
                raise ValueError("Reconstructed audio has different length than original")
            
            logger.info("Applying 18 kHz lowpass filter")
            enhanced_filtered = apply_lowpass_filter(enhanced, sr)
                
        except Exception as e:
            logger.error(f"Reconstruction error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error during reconstruction: {str(e)}"
            )

        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, enhanced_filtered, sr, format='WAV', subtype='PCM_16')
        audio_buffer.seek(0)

        import matplotlib.pyplot as plt

        logger.info(f"📊 Comparison: Original audio (with losses) vs Reconstructed audio")
        logger.info(f"ℹ️ Detected input loss: {loss_percentage:.2f}%")

        plt.figure(figsize=(14, 6))
        samples_to_plot = min(len(audio), sr * 5)
        time = np.arange(samples_to_plot) / sr

        plt.plot(time, audio[:samples_to_plot], label='Audio with losses', alpha=0.6, color='red')
        plt.plot(time, enhanced_filtered[:samples_to_plot], label='Reconstructed + filtered', alpha=0.7, color='green')

        plt.title(f"Audio Comparison | Loss: {loss_percentage:.2f}%")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout()

        comparison_path = Path("comparisons")
        comparison_path.mkdir(exist_ok=True)

        plot_file = comparison_path / f"comparison_{Path(file.filename).stem}.png"
        plt.savefig(plot_file, dpi=150)
        plt.close()

        logger.info(f"🖼️ Comparison plot saved: {plot_file}")

        logger.info("Reconstruction and filtering successful")
        return StreamingResponse(
            audio_buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=reconstructed_{file.filename}",
                "X-Original-Length": str(original_length),
                "X-Processed-Length": str(len(audio)),
                "X-Packet-Loss": f"{loss_percentage:.1f}%",
                "X-Reconstruction-Status": "success"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal error processing request"
        )

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 10000))  # Usa el puerto de Render o 10000 por defecto
    uvicorn.run(app, host="0.0.0.0", port=port)