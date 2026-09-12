# Vultr — backend GPU siempre encendido

Alternativa a Modal: una instancia **Vultr Cloud GPU** corriendo el contenedor 24/7.
Ventaja: **sin cold start nunca** → respuesta ~1-2s siempre. Costo: pagas la GPU de forma continua
(no por uso). Útil para la hora del juez / demo en vivo.

## Pasos

1. En vultr.com crea una instancia **Cloud GPU** (NVIDIA, p.ej. A16/A40/L40S) con **Ubuntu 22.04**.
   Abre el puerto **80** en el firewall.

2. Sube la carpeta `altur_100/` al servidor:

   ```bash
   scp -r altur_100 root@<IP>:/root/
   ```

3. Entra y corre el setup (instala Docker + NVIDIA toolkit y levanta el servicio):

   ```bash
   ssh root@<IP>
   cd /root/altur_100
   bash vultr/setup.sh
   ```

4. Listo:
   - API: `POST http://<IP>/detect`  body `{"audio_base64": "..."}` → `{"is_synthetic", "confidence"}`
   - Consola: `http://<IP>/console/`

## Notas

- El primer arranque descarga XLS-R + WavLM de Hugging Face (se cachean en el volumen `hfcache`);
  reinicios posteriores son rápidos.
- `restart: unless-stopped` mantiene el servicio vivo tras reinicios del servidor.
- Para HTTPS/URL bonita, pon un reverse proxy (Caddy/Nginx) delante, o usa Cloudflare.
- Si la Space de Hugging Face debe apuntar aquí en vez de a Modal, cambia `window.API_BASE`
  en `hf_space/index.html` por `http://<IP>` (o el dominio con HTTPS) y vuelve a subirla.
