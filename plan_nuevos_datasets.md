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

---

## ⚠️ Advertencias de estrategia de datos (LEER antes de generar/entrenar)

### 1. Edge-TTS solo da SINTÉTICOS — nos faltan HUMANOS reales
`generate_synthetic_mx.py` produce **solo voz sintética** (clase `synthetic`). **No genera ni un humano real.**
Entrenar con puros sintéticos de Edge-TTS **desbalancea** el dataset y puede **empeorar** el modelo.
Para que el aumento sirva, hay que agregar **humanos reales** en la MISMA proporción y condiciones:
- Humanos externos: subset **Common Voice español** pasado por el mismo pipeline telefónico.
- **Humanos grabados por nosotros** (ver punto 4).

### 2. Higiene de canal (bug que puede tirar todo)
**Humanos y sintéticos deben pasar por EXACTAMENTE el mismo procesado** (resample 8 kHz + filtro banda 300–3400 Hz + códec/ruido, vía `src/features/telephony_aug.py`).
Si los sintéticos van filtrados y los humanos no (o al revés), el modelo aprende **el canal, no la voz** → 99% en pruebas y **colapso en la evaluación oculta**.

### 3. Variedad de motores TTS
Altur usó *algún* motor de voz (posible ElevenLabs/Azure/Google), **no** Edge-TTS.
Aumentar solo con Edge-TTS enseña "sinteticidad de Microsoft", que puede no parecerse al set oculto.
Si se aumenta, usar **varios motores** para aprender sinteticidad genérica, no un motor específico.

### 4. Podemos GRABAR nuestro propio audio (opción válida y recomendada)
Nos faltan humanos reales en-dominio. **Podemos grabarlos nosotros:**
- Varias personas (equipo + conocidos) leyendo/improvisando frases de atención bancaria en **español mexicano**.
- Grabar por teléfono/mic, luego pasar por `telephony_aug.py` (mismo 8 kHz/filtro que los sintéticos).
- Idealmente diálogos (caller + agente) para replicar la dinámica conversacional de los turnos.
- Beneficio: voces **nuevas y reales** in-domain → mejora la robustez ante las voces nuevas del set oculto, que es justo la preocupación principal.
- Límite: serán pocas voces; sirve como complemento, no como dataset masivo.

### 5. ¿Vale la pena todo esto?
El modelo de audio (B) ya da **0.9976 en val**. El aumento **no es para subir val**, es para **robustez ante voces nuevas**.
Hacerlo **mal** (solo Edge-TTS, sin humanos, sin canal idéntico) tiene más riesgo que beneficio.
Hacerlo **bien** (sintéticos variados + humanos reales/grabados + canal idéntico) es la ruta correcta si el modelo no generaliza al set oculto.
