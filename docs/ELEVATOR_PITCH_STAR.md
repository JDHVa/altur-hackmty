# Elevator pitch — Centinel Altur (metodología STAR)

> Basado únicamente en `altur_100/`. Duración objetivo: **60–90 s**. Cuatro slides para Canva (S · T · A · R) + cierre.
> Visual: fondo `#0B0F14`, acento `#22D3EE`, humano `#34D399`, IA `#F43F5E`.

---

## Slide 1 — S · Situación
**Un bot con voz clonada marca al banco.**
- Suena humano, contesta preguntas de seguridad y no se cansa.
- El operador tiene segundos para decidir si es una persona real.
- Reto Altur: una llamada telefónica (WAV 8 kHz, canal 0 = quien llama) → `human` o `synthetic`.

*Guion (15 s):* "Hoy un atacante clona una voz en minutos y llama al banco. El operador no tiene forma de saber que del otro lado no hay nadie."

---

## Slide 2 — T · Tarea
**Decidir humano o IA con voces que nunca vimos.**
- Entregar `POST /detect` → `{is_synthetic, confidence}`.
- El set del juez es oculto y con hablantes nuevos: no sirve memorizar voces.
- Tiene que responder en segundos y estar en línea a la hora de la demo.

*Guion (15 s):* "Nuestra tarea: un endpoint que diga si la llamada es sintética, con una confianza en la que se pueda confiar, y que funcione con voces que nunca escuchó."

---

## Slide 3 — A · Acción
**Un detector A+B autocontenido (`altur_100`).**
- **A · Conversacional (42 features):** latencias, solapes, silencios, tiempo al primer turno, ratios de habla, nº de turnos — cómo *se comporta* quien llama.
- **B · Acústico (3 señales SSL congeladas):** XLS-R 300M + cabeza SLS · WavLM-base-plus + cabeza · Flow-LLR — cómo *suena* la voz.
- **Fusión:** HistGradientBoosting sobre A+B, umbral 0.5.
- **Versión rápida:** XLS-R a 3 ventanas, sin flow → ~2.7× más veloz.
- **Desplegado 3 veces:** Modal (GPU A10G, por uso) · Vultr GPU 24/7 (sin cold start, ~1–2 s) · consola pública en Hugging Face Space.

*Guion (30 s):* "Miramos dos cosas: cómo se comporta la llamada y cómo suena la voz. Usamos modelos de voz auto-supervisados congelados — aprenden artefactos de síntesis, no la identidad de la persona — y los fusionamos con las señales conversacionales en un solo clasificador. Todo está aislado en una carpeta, dockerizado y corriendo en GPU en la nube."

---

## Slide 4 — R · Resultado
**Precisión de casi 100 %, en producción.**
- Split oficial train → val: **100 %** (0 falsos positivos, 0 falsos negativos), AUC 1.0.
- Validación cruzada 5×5: **99.26 % ± 0.23**, AUC 0.9992.
- Holdout independiente (71 llamadas): **98.6 %**.
- API viva con contrato del reto + consola web donde subes un audio y ves el veredicto.

*Guion (20 s):* "Sobre el split oficial acertamos el 100 %; en validación cruzada, 99.3 %; en un holdout aparte, 98.6 %. Y no es un notebook: es un endpoint en GPU al que cualquiera le puede pegar ahora mismo."

---

## Slide 5 — Cierre
**Centinel Altur: detecta la voz que no existe.**
- `POST /detect` → `{is_synthetic, confidence}`
- Demo: `<URL Modal o Vultr>/console/`

*Guion (5 s):* "Centinel Altur. Detecta la voz que no existe."

---

## Versión de un solo párrafo (para decir de corrido, ~60 s)

"**Situación:** hoy cualquiera clona una voz y llama al banco; el operador no puede saber que no hay nadie del otro lado. **Tarea:** construir un endpoint que diga si la llamada es sintética, con confianza calibrada, y que funcione con voces que nunca vio. **Acción:** fusionamos 42 señales de *cómo se comporta* la llamada — latencias, silencios, turnos — con tres modelos de voz auto-supervisados congelados que detectan *cómo suena* una síntesis, en un solo clasificador desplegado en GPU en Modal, Vultr y Hugging Face. **Resultado:** 100 % en el split oficial, 99.3 % en validación cruzada, 98.6 % en holdout, y una API en producción con el contrato del reto. Centinel Altur: detecta la voz que no existe."
