import os
import asyncio
import edge_tts
import torchaudio

# Frases típicas de atención al cliente bancario (semejante a los audios reales de Altur)
PHRASES = [
    "Hola, me gustaría consultar el saldo de mi tarjeta de crédito.",
    "No reconozco un cargo de quinientos pesos en mi estado de cuenta.",
    "Quiero reportar mi tarjeta de débito como robada.",
    "¿Cuáles son los requisitos para un préstamo personal?",
    "Mi transferencia SPEI no ha pasado, ¿pueden ayudarme?",
    "Necesito cancelar mi seguro de auto.",
    "Sí, confirmo que yo realicé esa compra ayer en la noche.",
    "¿Me puede comunicar con un ejecutivo por favor?",
    "Olvidé el NIP del cajero, ¿cómo lo recupero?",
    "Quisiera saber el estatus de mi aclaración."
]

# Voces en Español Mexicano de Microsoft Edge Neural
VOICES = ["es-MX-JorgeNeural", "es-MX-DaliaNeural"]

# Carpeta de salida (el script la creará automáticamente)
OUTPUT_DIR = "datasets_externos/Synthetic_MX_8kHz"

async def generate_audio():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=== Iniciando generación de audios sintéticos (Edge-TTS) ===")
    
    count = 0
    for voice in VOICES:
        for i, text in enumerate(PHRASES):
            temp_file = f"{OUTPUT_DIR}/temp_{voice}_{i}.wav"
            final_file = f"{OUTPUT_DIR}/synth_{voice}_{i}.wav"
            
            print(f"Generando: {voice} -> '{text[:30]}...'")
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(temp_file)
            
            # Resample a 8kHz para simular teléfono (Condiciones del Altur Challenge)
            waveform, sr = torchaudio.load(temp_file)
            if sr != 8000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=8000)
                waveform = resampler(waveform)
            
            # Guardar el audio final a 8kHz PCM 16-bit
            torchaudio.save(final_file, waveform, 8000, encoding="PCM_S", bits_per_sample=16)
            
            # Limpiar archivo original temporal
            os.remove(temp_file)
            count += 1
            print(f" [OK] Guardado: {final_file}")
            
    print(f"\n¡Éxito! Total de audios generados a 8kHz: {count}")

if __name__ == '__main__':
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(generate_audio())
