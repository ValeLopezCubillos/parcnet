import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from parcnet import PARCnet
from scipy.signal import butter, lfilter, freqz, firwin, kaiser
from scipy import signal
import pesq  # Necesitarás instalar pypesq: pip install pypesq

def design_butterworth_lowpass(cutoff, fs, order=5):
    """Diseña un filtro Butterworth de paso bajo"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def design_auditory_filter(cutoff, fs, numtaps=255):
    """Diseña un filtro FIR optimizado para audio"""
    nyq = 0.5 * fs
    width = 500/nyq  # Anchura de transición
    beta = 8.0       # Parámetro de Kaiser
    taps = firwin(numtaps, cutoff/nyq, width=width, 
                 window=('kaiser', beta), pass_zero=True)
    return taps

def apply_iir_filter(data, b, a):
    """Aplica filtro IIR (Butterworth)"""
    return lfilter(b, a, data)

def apply_fir_filter(data, taps):
    """Aplica filtro FIR"""
    return lfilter(taps, 1.0, data)

def calculate_advanced_metrics(original, lossy, enhanced, sr):
    """Calcula métricas avanzadas de calidad"""
    # Identificar muestras perdidas
    lost_samples = np.where(lossy == 0)[0]
    present_samples = np.where(lossy != 0)[0]
    
    # 1. Métricas básicas
    mse_all = np.mean((enhanced - original)**2)
    mse_lost = np.mean((enhanced[lost_samples] - original[lost_samples])**2) if len(lost_samples) > 0 else 0
    mse_present = np.mean((enhanced[present_samples] - original[present_samples])**2) if len(present_samples) > 0 else 0
    
    # 2. Correlación en muestras perdidas
    if len(lost_samples) > 1:
        correlation = np.corrcoef(original[lost_samples], enhanced[lost_samples])[0,1]
    else:
        correlation = 0
    
    # 3. Error relativo
    relative_error = np.mean(np.abs(enhanced[lost_samples] - original[lost_samples]) / 
                          (np.abs(original[lost_samples]) + 1e-6)) if len(lost_samples) > 0 else 0
    
    # 4. PESQ (Perceptual Evaluation of Speech Quality)
    try:
        pesq_score = pesq.pesq(sr, original, enhanced, 'wb')
    except:
        pesq_score = -1  # En caso de error
    
    return {
        'total_samples': len(original),
        'lost_samples': len(lost_samples),
        'present_samples': len(present_samples),
        'mse_all': mse_all,
        'mse_lost': mse_lost,
        'mse_present': mse_present,
        'correlation_recovered': correlation,
        'relative_error': relative_error,
        'pesq_score': pesq_score,
        'max_diff': np.max(np.abs(enhanced - original))
    }

def plot_time_domain(original, lossy, enhanced, enhanced_filtered, sr, output_dir):
    """Visualización en dominio del tiempo"""
    samples = min(len(original), 5 * sr)  # Primeros 5 segundos
    time = np.arange(samples)/sr
    
    plt.figure(figsize=(15, 10))
    
    # 1. Señal original vs con pérdidas
    plt.subplot(3,1,1)
    plt.plot(time, original[:samples], label="Original", alpha=0.7)
    plt.plot(time, lossy[:samples], 'r', label="Con pérdidas", alpha=0.4)
    plt.title("Señal Original vs Con Pérdidas")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.legend()
    
    # 2. Señal original vs mejorada
    plt.subplot(3,1,2)
    plt.plot(time, original[:samples], label="Original", alpha=0.7)
    plt.plot(time, enhanced[:samples], 'g', label="Mejorada", alpha=0.6)
    plt.title("Señal Mejorada vs Original")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.legend()
    
    # 3. Señal original vs mejorada y filtrada
    plt.subplot(3,1,3)
    plt.plot(time, original[:samples], label="Original", alpha=0.7)
    plt.plot(time, enhanced_filtered[:samples], 'm', label="Mejorada+Filtrada", alpha=0.6)
    plt.title("Señal Mejorada y Filtrada vs Original")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/time_domain_comparison.png", dpi=150)
    plt.close()

def plot_frequency_domain(original, enhanced, enhanced_filtered, sr, output_dir):
    """Visualización en dominio de la frecuencia"""
    n = len(original)
    freqs = np.fft.rfftfreq(n, d=1/sr)
    
    # Calcular espectros
    orig_fft = np.abs(np.fft.rfft(original))
    enh_fft = np.abs(np.fft.rfft(enhanced))
    enh_filt_fft = np.abs(np.fft.rfft(enhanced_filtered))
    
    plt.figure(figsize=(15, 10))
    
    # 1. Espectros completos
    plt.subplot(2,1,1)
    plt.semilogy(freqs, orig_fft, label="Original")
    plt.semilogy(freqs, enh_fft, 'g', label="Mejorada", alpha=0.7)
    plt.semilogy(freqs, enh_filt_fft, 'm', label="Mejorada+Filtrada", alpha=0.7)
    plt.title("Comparación de Espectros")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud (log)")
    plt.legend()
    plt.grid()
    
    # 2. Relación de espectros
    plt.subplot(2,1,2)
    plt.plot(freqs, enh_fft/(orig_fft+1e-6), 'g', label="Mejorada/Original", alpha=0.7)
    plt.plot(freqs, enh_filt_fft/(orig_fft+1e-6), 'm', label="Mejorada+Filtrada/Original", alpha=0.7)
    plt.axhline(1, color='k', linestyle='--')
    plt.title("Relación de Espectros")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Relación de Magnitud")
    plt.legend()
    plt.grid()
    plt.ylim([0, 2])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/frequency_domain_comparison.png", dpi=150)
    plt.close()

def plot_critical_bands(original, enhanced, sr, output_dir):
    """Análisis por bandas críticas"""
    n = len(original)
    freqs = np.fft.rfftfreq(n, d=1/sr)
    
    # Bandas críticas (aproximación Bark scale)
    critical_bands = [0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 
                     1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 
                     4400, 5300, 6400, 7700, 9500, 12000, 15500, 20000]
    
    # Calcular energía por banda crítica
    band_energy_ratio = []
    for i in range(len(critical_bands)-1):
        mask = (freqs >= critical_bands[i]) & (freqs < critical_bands[i+1])
        orig_energy = np.sum(np.abs(np.fft.rfft(original)[mask]))
        enh_energy = np.sum(np.abs(np.fft.rfft(enhanced)[mask]))
        band_energy_ratio.append(enh_energy / (orig_energy + 1e-6))
    
    # Graficar
    plt.figure(figsize=(15, 6))
    x_pos = np.arange(len(band_energy_ratio))
    plt.bar(x_pos, band_energy_ratio, color=['r' if x < 0.7 else 'g' for x in band_energy_ratio])
    plt.axhline(1, color='k', linestyle='--')
    plt.title("Relación de Energía por Banda Crítica (Mejorada/Original)")
    plt.xlabel("Banda Crítica (Hz)")
    plt.ylabel("Relación de Energía")
    plt.xticks(x_pos, [f"{critical_bands[i]}-{critical_bands[i+1]}" for i in range(len(critical_bands)-1)], rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/critical_band_analysis.png", dpi=150)
    plt.close()

def generate_abx_test_samples(original, enhanced, sr, output_dir):
    """Genera muestras para prueba auditiva A/B/X"""
    duration = 5  # segundos
    samples = min(len(original), duration * sr)
    
    # 1. Original (A)
    sf.write(f"{output_dir}/test_A_original.wav", original[:samples], sr)
    
    # 2. Mejorada (B)
    sf.write(f"{output_dir}/test_B_enhanced.wav", enhanced[:samples], sr)
    
    # 3. Muestra X (aleatoria A o B)
    if np.random.rand() > 0.5:
        sf.write(f"{output_dir}/test_X.wav", original[:samples], sr)
    else:
        sf.write(f"{output_dir}/test_X.wav", enhanced[:samples], sr)
    
    print("\n🔊 Muestras para prueba A/B/X generadas:")
    print(f"- A: {output_dir}/test_A_original.wav")
    print(f"- B: {output_dir}/test_B_enhanced.wav")
    print(f"- X: {output_dir}/test_X.wav (desconocido)")

def run_test():
    # Configuración
    audio_file = "test_audio.wav"
    output_dir = "test_results"
    loss_percent = 0.5
    cutoff_freq = 18000  # Frecuencia de corte del filtro
    
    # 1. Preparar entorno
    print("🔊 Preparando prueba...")
    Path(output_dir).mkdir(exist_ok=True)
    
    # 2. Cargar audio
    print(f"🎵 Cargando {audio_file}...")
    try:
        audio, sr = sf.read(audio_file)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=44100)
        audio = audio.astype(np.float32)
        sr = 44100
        print(f"✅ Audio cargado: {len(audio)/sr:.2f} segundos a {sr}Hz, Mono")
    except Exception as e:
        print(f"❌ Error cargando audio: {e}")
        return

    # 3. Configurar modelo
    print("⚙️ Inicializando modelo PARCnet...")
    try:
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
        print(f"✅ Modelo cargado en {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        print(f"⚙️ Parámetros del modelo: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"❌ Error inicializando modelo: {e}")
        return

    # 4. Ajustar audio
    packet_size = 512
    audio = audio[:len(audio) // packet_size * packet_size]
    
    # 5. Simular pérdida
    trace = np.random.choice([0, 1], size=len(audio)//packet_size, 
                           p=[loss_percent, 1-loss_percent]).astype(np.float32)
    lossy_audio = (audio * np.repeat(trace, packet_size)).astype(np.float32)
    
    # 6. Procesar con PARCnet
    print("🔄 Procesando audio...")
    try:
        enhanced = model(lossy_audio, trace)
    except Exception as e:
        print(f"❌ Error procesando audio: {e}")
        return
    
    # 7. Diseñar y aplicar filtros
    print("🔧 Aplicando filtros...")
    try:
        # Filtro IIR (Butterworth)
        b, a = design_butterworth_lowpass(cutoff_freq, sr, order=6)
        enhanced_iir = apply_iir_filter(enhanced, b, a)
        
        # Filtro FIR (Kaiser)
        fir_taps = design_auditory_filter(cutoff_freq, sr, numtaps=255)
        enhanced_fir = apply_fir_filter(enhanced, fir_taps)
        
        # Usamos el filtro FIR por defecto (mejor fase lineal)
        enhanced_filtered = enhanced_fir
    except Exception as e:
        print(f"❌ Error aplicando filtros: {e}")
        enhanced_filtered = enhanced
    
    # 8. Calcular métricas avanzadas
    print("📊 Calculando métricas...")
    metrics = calculate_advanced_metrics(audio, lossy_audio, enhanced_filtered, sr)
    
    # 9. Visualizaciones
    print("📈 Generando visualizaciones...")
    plot_time_domain(audio, lossy_audio, enhanced, enhanced_filtered, sr, output_dir)
    plot_frequency_domain(audio, enhanced, enhanced_filtered, sr, output_dir)
    plot_critical_bands(audio, enhanced_filtered, sr, output_dir)
    
    # 10. Generar muestras para prueba auditiva
    generate_abx_test_samples(audio, enhanced_filtered, sr, output_dir)
    
    # 11. Guardar resultados
    print("💾 Guardando archivos de audio...")
    sf.write(f"{output_dir}/original.wav", audio, sr)
    sf.write(f"{output_dir}/lossy.wav", lossy_audio, sr)
    sf.write(f"{output_dir}/enhanced.wav", enhanced, sr)
    sf.write(f"{output_dir}/enhanced_filtered.wav", enhanced_filtered, sr)
    sf.write(f"{output_dir}/enhanced_iir.wav", enhanced_iir, sr)
    sf.write(f"{output_dir}/enhanced_fir.wav", enhanced_fir, sr)
    
    # 12. Mostrar resultados
    print("\n📌 Resultados Finales:")
    print(f"- Muestras totales: {metrics['total_samples']:,}")
    print(f"- Muestras perdidas: {metrics['lost_samples']:,} ({loss_percent*100:.1f}%)")
    print(f"- Correlación en muestras perdidas: {metrics['correlation_recovered']:.3f}")
    print(f"- Error relativo: {metrics['relative_error']:.3f}")
    print(f"- MSE (total): {metrics['mse_all']:.6f}")
    print(f"- MSE (muestras perdidas): {metrics['mse_lost']:.6f}")
    print(f"- MSE (muestras presentes): {metrics['mse_present']:.6f}")
    print(f"- PESQ Score: {metrics['pesq_score']:.2f}")
    print(f"- Máxima diferencia: {metrics['max_diff']:.4f}")
    
    print(f"\n💾 Todos los resultados guardados en: {output_dir}/")

if __name__ == "__main__":
    run_test()