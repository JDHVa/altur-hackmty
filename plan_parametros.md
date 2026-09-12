# Plan de parámetros — barrido colaborativo (fase crítica)

> Objetivo: encontrar los **pesos de las 4 señales de audio + umbral** que mejor separen
> HUMANO de IA en el set de demos. Dos personas prueban 5 configs cada una.
> **No requiere GPU ni modelos:** todo corre sobre `demo_scores.csv` (ya commiteado).

## Cómo funciona

`demo_scores.csv` tiene, por audio, las 4 señales ya calculadas (`wavlm, xlsr, prosody, flow`)
y su `label` (0=humano real, 1=IA/clon). Se evalúa una config al instante:

```
python scripts/eval_config.py --wavlm 0.25 --xlsr 0.40 --prosody 0.25 --flow 0.10 --threshold 0.30
```
Score final = promedio ponderado de las señales (pesos renormalizados). `pred = score >= threshold`.
El script imprime **ACC**, **FN** (IA que pasa como humano — el error grave), **FP** (humano marcado IA) y los errores.

## Criterio de "mejor" (prioridad banco)
1. **Minimizar FN** (dejar pasar una IA es lo más caro).
2. Luego minimizar FP (molestar humanos).
3. Empate → mayor ACC.
Nota: hay 2 clones casi imposibles (`clon_emilio`, `clon_grabacion_27`: todas las señales ~0) →
probablemente sean FN en toda config; no te obsesiones con ellos.

## Datos actuales (baseline)
Config actual `0.25/0.40/0.25/0.10 @0.30` → ACC 0.75, FN 3, FP 3.
Problema conocido: **prosody sube a humanos limpios** (emilio_real 0.94) y hay reales nuevos difíciles.

## Reparto — Persona A (5 configs)

| # | wavlm | xlsr | prosody | flow | thr | idea |
|---|---|---|---|---|---|---|
| A1 | 0.25 | 0.40 | 0.25 | 0.10 | 0.30 | baseline |
| A2 | 0.20 | 0.60 | 0.10 | 0.10 | 0.30 | xlsr fuerte, prosody bajo |
| A3 | 0.40 | 0.50 | 0.00 | 0.10 | 0.30 | sin prosody |
| A4 | 0.40 | 0.50 | 0.00 | 0.10 | 0.22 | sin prosody, umbral bajo |
| A5 | 0.00 | 1.00 | 0.00 | 0.00 | 0.40 | solo xlsr |

## Reparto — Persona B (5 configs)

| # | wavlm | xlsr | prosody | flow | thr | idea |
|---|---|---|---|---|---|---|
| B1 | 0.15 | 0.35 | 0.40 | 0.10 | 0.35 | prosody protagonista |
| B2 | 0.30 | 0.30 | 0.25 | 0.15 | 0.30 | balanceado |
| B3 | 0.45 | 0.45 | 0.00 | 0.10 | 0.28 | wavlm+xlsr, sin prosody |
| B4 | 0.30 | 0.55 | 0.05 | 0.10 | 0.35 | xlsr manda, umbral alto |
| B5 | 0.25 | 0.45 | 0.20 | 0.10 | 0.25 | umbral bajo |

## Qué reportar por cada config
Copiar la línea de resultados, p.ej.:
```
A2 -> ACC=0.83  FN=2  FP=1
```
Comparar y quedarnos con la de **menor FN** (desempate por FP y ACC).

## Después de elegir la ganadora
- Fijar esos pesos en `scripts/analyze_audio.py` (`WEIGHTS`) y el `--threshold` por defecto.
- Y en el endpoint: `api/inference.py` (`AUDIO_WEIGHTS`) — coordinar con Jesús.
- Revalidar con **voces nuevas** (no las del CSV) para no sobreajustar.

## Atajo opcional (automático)
Si quieren el óptimo directo, se puede hacer un grid corriendo `eval_config` sobre muchas combos;
avisen y lo agrego. El reparto manual sirve para entender los trade-offs.
