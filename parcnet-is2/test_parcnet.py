import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from parcnet import PARCnet

def run_test():
    # Configuración
    audio_file = "test_audio.wav"
    output_dir = "test_results"
    loss_percent = 0.3
    
    # 1. Preparar entorno
    print("🔊 Preparando prueba...")
    Path(output_dir).mkdir(exist_ok=True)
    
    # 2. Cargar y convertir audio a MONO y float32 explícitamente
    print(f"🎵 Cargando {audio_file}...")
    try:
        # Cargar y convertir a mono manualmente si es estéreo
        audio, sr = sf.read(audio_file)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)  # Convertir a mono
        audio = librosa.resample(audio, orig_sr=sr, target_sr=44100)
        audio = audio.astype(np.float32)  # Conversión EXPLÍCITA a float32
        sr = 44100
        print(f"✅ Audio cargado: {len(audio)/sr:.2f} segundos a {sr}Hz, Mono")
    except Exception as e:
        print(f"❌ Error cargando audio: {e}")
        return

    # 3. Configurar modelo
    print("⚙️ Inicializando modelo PARCnet...")
    model = PARCnet(
        model_checkpoint="pretrained_models/parcnet-is2_baseline_checkpoint.ckpt",
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

    # 4. Ajustar y preparar audio
    packet_size = 512
    audio = audio[:len(audio) // packet_size * packet_size]  # Ajustar longitud
    
    # 5. Simular pérdida (asegurando mismo tipo)
    trace = np.random.choice([0, 1], size=len(audio)//packet_size, 
                           p=[loss_percent, 1-loss_percent]).astype(np.float32)
    lossy_audio = (audio * np.repeat(trace, packet_size)).astype(np.float32)
    
    # 6. Procesar con PARCnet
    print("🔄 Procesando audio...")
    enhanced = model(lossy_audio, trace)
    
    # 7. Guardar resultados
    sf.write(f"{output_dir}/original.wav", audio, sr)
    sf.write(f"{output_dir}/con_perdidas.wav", lossy_audio, sr)
    sf.write(f"{output_dir}/mejorado.wav", enhanced, sr)
    
    # 8. Visualización
    plot_results(audio, lossy_audio, enhanced, sr, output_dir)
    print(f"💾 Resultados guardados en: {output_dir}/")

def plot_results(original, lossy, enhanced, sr, output_dir):
    """Visualización mejorada con áreas de pérdida resaltadas"""
    plt.figure(figsize=(15, 8))
    samples = min(len(original), 5 * sr)  # Mostrar primeros 5 segundos o menos
    time = np.arange(samples)/sr
    
    # 1. Original vs Con pérdidas
    plt.subplot(2,1,1)
    plt.plot(time, original[:samples], label="Original", alpha=0.7)
    plt.plot(time, lossy[:samples], 'r', label="Con pérdidas", alpha=0.6)
    
    # Resaltar áreas perdidas
    lost_samples = np.where(lossy[:samples] == 0)[0]
    if len(lost_samples) > 0:
        plt.scatter(time[lost_samples], original[lost_samples], 
                   color='red', s=10, label="Muestras perdidas")
    
    plt.title(f"Comparación (Primeros {samples/sr:.1f}s) - {len(lost_samples)} muestras perdidas")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.legend()
    
    # 2. Original vs Mejorado
    plt.subplot(2,1,2)
    plt.plot(time, original[:samples], label="Original", alpha=0.7)
    plt.plot(time, enhanced[:samples], 'g', label="Mejorado", alpha=0.6)
    plt.title("Audio mejorado vs Original")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparacion.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    run_test()