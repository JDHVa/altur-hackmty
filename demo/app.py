# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from src.features.telephony_aug import augment_chain
from features.audio import audio_score
from features.prosody import prosody_features

THRESHOLD = float(os.environ.get('DEMO_THRESHOLD', '0.2'))

REF = {
    'hnr': (9.8, 14.2, 'Relación armónico-ruido: los sintéticos suelen ser MÁS limpios (HNR alto)'),
    'shimmer_local': (0.101, 0.091, 'Variación de amplitud: los humanos varían MÁS (shimmer alto)'),
    'jitter_local': (0.019, 0.024, 'Variación de tono ciclo a ciclo'),
    'voiced_frac': (0.45, 0.57, 'Fracción sonora'),
}


def analizar(audio):
    if audio is None:
        return {'Sin audio': 1.0}, "Graba o sube un audio primero."
    sr, data = audio
    x = np.asarray(data)
    if x.ndim > 1:
        x = x[:, 0]
    x = x.astype(np.float32)
    if np.issubdtype(np.asarray(data).dtype, np.integer):
        x = x / 32768.0
    if x.size < sr // 2:
        return {'Audio muy corto': 1.0}, "Graba al menos ~1 segundo."

    proc, out_sr = augment_chain(x, sr, codec='g711_ulaw')
    p = float(audio_score(proc, out_sr))
    bio = prosody_features(proc, out_sr)

    label = {'HUMANO 🧑': round(1 - p, 3), 'IA / sintético 🤖': round(p, 3)}

    veredicto = 'IA (sintético)' if p >= THRESHOLD else 'HUMANO'
    lines = [f"### Veredicto: **{veredicto}**",
             f"Probabilidad de sintético: **{p*100:.1f}%**  (umbral {THRESHOLD})",
             "",
             "**Desglose biológico (voz real vs IA):**",
             "",
             "| Métrica | Tu voz | Típico humano | Típico IA |",
             "|---|---|---|---|"]
    for k, (h, s, _desc) in REF.items():
        lines.append(f"| {k} | {bio.get(k, 0):.3f} | {h} | {s} |")
    lines.append("")
    lines.append("_Los sintéticos tienden a sonar 'demasiado limpios' (HNR alto) y con menos variación natural (shimmer bajo)._")
    return label, "\n".join(lines)


with gr.Blocks(title="¿Humano o IA?") as demo:
    gr.Markdown("# 🎙️ ¿Humano o IA?\nGraba tu voz (o sube un audio) y el modelo de anti-spoofing decide si eres una persona real o una voz sintética.\n\nEl audio se procesa como una llamada telefónica (8 kHz, μ-law) igual que en el entrenamiento.")
    with gr.Row():
        inp = gr.Audio(sources=['microphone', 'upload'], type='numpy', label='Tu voz')
    btn = gr.Button('Analizar', variant='primary')
    with gr.Row():
        out_label = gr.Label(label='Veredicto', num_top_classes=2)
        out_md = gr.Markdown()
    btn.click(analizar, inputs=inp, outputs=[out_label, out_md])


if __name__ == '__main__':
    demo.launch(server_name='127.0.0.1', server_port=7860, inbrowser=True)
