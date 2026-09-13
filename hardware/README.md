# Guía de Conexión a Centinel Altur (Raspberry Pi 5) para el Equipo

Esta guía permite a cualquier integrante del equipo (**Jesús, Emilio o Alonso**) conectarse por SSH a la Raspberry Pi 5 desde su laptop (Windows, Mac o Linux) para probar hardware, correr scripts o enlazar el backend en vivo.

---

## 1. Requisito de Red (Crucial)

Tu laptop y la Raspberry Pi deben estar en la **misma red**:
- Si la Pi está conectada al hotspot de Alonso, conecta tu laptop a la red **`AlonsoHotspot`** (password: `REVOLUTIONHCMTY`).
- Si la Pi está en la red del Hackathon, asegúrate de estar en esa misma red.

---

## 2. Conectarse por SSH

Abre tu terminal (**PowerShell** en Windows, o **Terminal** en Mac/Linux) y corre:

```bash
ssh revolution@altur.local
```

- **Usuario:** `revolution`
- **Contraseña:** `revolution` (o la contraseña acordada)

### Si `altur.local` no resuelve en tu laptop:
En algunas laptops Windows el nombre mDNS `.local` puede tardar en resolver. Pídele a Alonso la IP de la Pi (se consulta con `hostname -I` en la Pi) y conéctate directo:

```bash
ssh revolution@<IP_DE_LA_PI>
# Ejemplo: ssh revolution@172.20.10.4
```

---

## 3. Dónde está el código y el entorno en la Pi

Una vez dentro de la sesión SSH:

1. **Ubicación del repositorio:**
   ```bash
   cd ~/altur/hardware
   ```

2. **Entorno Virtual de Python:**
   El entorno se activa automáticamente. Si no ves `(centinela-env)` en tu prompt, actívalo con:
   ```bash
   source ~/centinela-env/bin/activate
   ```

3. **Sincronizar cambios de Git:**
   ```bash
   git fetch origin
   git pull origin hardware
   ```

---

## 4. Ejecución de Pruebas de Hardware

Desde `~/altur/hardware`:

```bash
# Probar la Matriz LED (MAX7219)
python3 matrix/test_matrix.py

# Probar la Pantalla TFT (ILI9341)
python3 display/test_tft.py

# Probar la Bocina I2S y el Micrófono USB
python3 audio/test_audio.py
```

---

## 5. Conectar la Pi al Backend de tu Laptop

Cuando tengas corriendo tu servidor FastAPI en tu laptop (`uvicorn api.main:app --host 0.0.0.0 --port 8000`), la Pi puede enviarle audio en vivo apuntando a la IP local de tu laptop:

1. Obtén tu IP local en tu laptop (`ipconfig` en Windows, busca la IPv4 de Wi-Fi).
2. En la Pi, el cliente Centinel se conectará a:
   `ws://<TU_IP_LOCAL>:8000/ws/call?source=mic`

---

## 6. Modo Centinel LOCAL en vivo (compu = IA, Pi = pantalla)

Flujo nuevo y **solo local**: la **computadora** captura el micrófono, corre el modelo
(`altur_100`) y envía únicamente el resultado a la **Pi**, que solo lo muestra bonito.
Es **opcional y no intrusivo**: si la Pi no está, la compu igual funciona (muestra en consola);
la versión en línea (Modal/HF) no se toca.

Ambos equipos en la **misma WiFi**.

### En la Pi (solo pantalla)
```bash
cd ~/altur/hardware
python3 centinela_display.py          # escucha en 0.0.0.0:8080, pinta TFT + matriz
```
Consigue la IP de la Pi con `hostname -I`.

### En la computadora (mic + IA)
Primero levanta el modelo local (una vez):
```bash
cd altur_100
python -m uvicorn serve:app --port 8010
```
Luego el centinela local, apuntando a la Pi:
```bash
cd hardware
# Windows PowerShell:
$env:ALTUR_PI_URL="http://<IP_DE_LA_PI>:8080/state"; python centinela_local.py
# Linux/Mac:
ALTUR_PI_URL="http://<IP_DE_LA_PI>:8080/state" python centinela_local.py
```

- Queda en reposo hasta que detecta voz por el micrófono; entonces graba el segmento,
  lo clasifica en local y actualiza la Pi (y la consola) con HUMANO / VOZ IA.
- Sin servidor local: usa `python centinela_local.py --direct` (carga el modelo en el mismo proceso).
- Variables útiles: `ALTUR_LOCAL_URL` (default `http://127.0.0.1:8010`), `ALTUR_MIC_DEVICE`,
  `ALTUR_VAD_THR` (sensibilidad de voz), `ALTUR_PI_URL` (si se omite, solo consola).
