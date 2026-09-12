# -*- coding: utf-8 -*-
import re
import numpy as np

FILLERS = {
    'eh', 'ehh', 'eeh', 'em', 'emm', 'este', 'esteee', 'mmm', 'mm', 'mmh', 'aja', 'ajá',
    'pues', 'osea', 'bueno', 'verdad', 'digamos', 'haber', 'okey', 'ok', 'ajam', 'eee',
    'hmm', 'uff', 'ah', 'oh', 'ea',
}
MULTI_FILLERS = ['o sea', 'a ver', 'no sé', 'es que', 'como que', 'por así decirlo', 'la verdad']

LING_KEYS = ['ling_n_words', 'ling_speaking_rate', 'ling_filler_ratio', 'ling_ttr',
             'ling_repeat_ratio', 'ling_mean_word_len', 'ling_n_pauses', 'ling_mean_pause',
             'ling_pause_ratio', 'ling_articulation_rate']


def _tokens(text):
    return re.findall(r"[a-záéíóúñü]+", text.lower())


def linguistic_features(stt_result: dict) -> dict:
    text = stt_result.get('text', '') or ''
    words_ts = stt_result.get('words', []) or []
    dur = float(stt_result.get('duration_s', 0.0)) or 0.0
    toks = _tokens(text)
    n = len(toks)
    out = {k: 0.0 for k in LING_KEYS}
    if n == 0:
        return out

    low = ' ' + text.lower() + ' '
    fill = sum(1 for t in toks if t in FILLERS)
    fill += sum(low.count(' ' + mf + ' ') for mf in MULTI_FILLERS)

    repeats = sum(1 for i in range(1, n) if toks[i] == toks[i - 1])
    uniq = len(set(toks))

    out['ling_n_words'] = float(n)
    out['ling_speaking_rate'] = float(n / dur) if dur > 0 else 0.0
    out['ling_filler_ratio'] = float(fill / n)
    out['ling_ttr'] = float(uniq / n)
    out['ling_repeat_ratio'] = float(repeats / n)
    out['ling_mean_word_len'] = float(np.mean([len(t) for t in toks]))

    if words_ts:
        gaps = []
        speech = 0.0
        for i in range(len(words_ts)):
            w = words_ts[i]
            if w.get('start') is not None and w.get('end') is not None:
                speech += max(0.0, w['end'] - w['start'])
            if i > 0:
                g = words_ts[i]['start'] - words_ts[i - 1]['end']
                if g is not None and g > 0:
                    gaps.append(g)
        pauses = [g for g in gaps if g > 0.3]
        out['ling_n_pauses'] = float(len(pauses))
        out['ling_mean_pause'] = float(np.mean(pauses)) if pauses else 0.0
        out['ling_pause_ratio'] = float(sum(pauses) / dur) if dur > 0 else 0.0
        out['ling_articulation_rate'] = float(len(words_ts) / speech) if speech > 0 else 0.0

    return out
