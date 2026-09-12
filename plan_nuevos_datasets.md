# Plan de Adquisición y Generación de Datasets (Altur Challenge)

## Goal Description
El servidor interrumpió las descargas previas. El objetivo de este plan es triple:
1. **Reanudar de forma segura** las descargas de ASVspoof y WaveFake.
2. **Incorporar nuevos datasets externos** de alta calidad (FakeAVCeleb y VCC).
3. **Estrategia "Secret Weapon"**: Crear un pipeline automatizado para generar miles de audios sintéticos en **español mexicano (telefonía 8kHz)** usando modelos TTS modernos (Microsoft Edge Neural TTS) de forma gratuita. Esto es vital porque los modelos entrenados solo en inglés sufren caídas de precisión de hasta un 30% al inferir en español debido al "Domain Gap".

## User Review Required
> [!IMPORTANT]
> - **Almacenamiento:** ¿Tienes espacio suficiente en disco? Descargar todos los datasets externos adicionales puede tomar >40GB. 
> - **Alineación de Estrategia:** ¿Estás de acuerdo con la estrategia de generar nuestro propio dataset sintético en español mexicano usando `edge-tts`? Es muy rápido y no consume recursos de GPU, pero requiere instalar una pequeña librería de Python.

## Open Questions
> [!WARNING]
> 1. Para los audios reales de nuestro dataset generado, propongo descargar el subset de español de **Mozilla Common Voice**, ¿te parece bien?
> 2. ¿Prefieres priorizar la generación del dataset propio (que tarda minutos) antes de descargar los datasets externos gigantes (que tardan horas)?

---

## Proposed Changes

### 1. Reanudación de Descargas Interrumpidas
Las descargas anteriores se detuvieron a la mitad. El script original no manejaba descargas parciales, por lo que el archivo ZIP podría estar corrupto si solo se descargó una parte.
#### [MODIFY] `C:\Users\jesus\Proyectos\Altur\download_datasets.py`
Se actualizará el script para verificar si el archivo `.zip` es válido. Si está corrupto o incompleto, lo eliminará y reiniciará la descarga. Se usarán librerías más robustas como `requests` con streaming.

### 2. Integración de Nuevos Datasets Externos
#### [NEW] `C:\Users\jesus\Proyectos\Altur\src\data\download_extra_datasets.py`
Agregaremos un script para descargar dos fuentes clave adicionales:
- **FakeAVCeleb (Solo Audio)**: Voces clonadas en tiempo real (SV2TTS), excelente para detectar clones de voz.
- **Voice Conversion Challenge (VCC 2020)**: Para que el modelo aprenda a detectar audios donde un humano habla pero su voz es convertida a otra identidad (una técnica distinta al TTS clásico).

### 3. Generador de Datasets Sintéticos (Español Mexicano 8kHz)
#### [NEW] `C:\Users\jesus\Proyectos\Altur\src\data\generate_synthetic_mx.py`
Un script de Python que utilizará `edge-tts` (la API de voces neuronales de Microsoft Edge, que es de acceso libre) para generar miles de locuciones.
- Usará voces como `es-MX-JorgeNeural` y `es-MX-DaliaNeural`.
- Convertirá automáticamente el resultado a 8kHz y añadirá un filtro paso-banda telefónico para que la acústica sea idéntica al Altur Challenge.
- El costo computacional es bajísimo (descarga el audio ya generado).

#### [MODIFY] Requerimientos
Ejecución de:
```bash
pip install edge-tts torchaudio soundfile requests
```

---

## Verification Plan

### Automated Tests
- Ejecutaré un test rápido del script generador de TTS: `python src/data/generate_synthetic_mx.py --test-run`. Verificaremos que el archivo de salida sea efectivamente WAV de 8kHz, Mono.
- Ejecutaré una validación de los ZIPs de ASVspoof con el módulo `zipfile` para asegurar que las descargas estén íntegras.

### Manual Verification
- Te pediré que escuches un archivo generado `fake_es_MX_001.wav` para confirmar si la acústica del "teléfono" suena similar a las llamadas del banco que provee el reto.
