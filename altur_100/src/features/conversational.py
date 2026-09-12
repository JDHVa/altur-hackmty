import numpy as np

FEATURE_ORDER = [
    'duration_s', 'n_turns_total', 'n_turns_caller', 'n_turns_agent',
    'ratio_caller_agent_turns', 'caller_dur_mean', 'caller_dur_std',
    'caller_dur_min', 'caller_dur_max', 'caller_dur_median', 'caller_dur_cv',
    'agent_dur_mean', 'agent_dur_std', 'agent_dur_min', 'agent_dur_max',
    'agent_dur_median', 'caller_total_speech_s', 'agent_total_speech_s',
    'caller_speech_ratio', 'agent_speech_ratio', 'caller_latency_mean',
    'caller_latency_std', 'caller_latency_min', 'caller_latency_max',
    'caller_latency_median', 'n_caller_responses', 'agent_latency_mean',
    'agent_latency_std', 'n_overlaps', 'overlap_total_s', 'overlap_mean',
    'n_silences', 'silence_total_s', 'silence_mean', 'silence_std',
    'silence_ratio', 'caller_consecutive_segments', 'agent_consecutive_segments',
    'caller_turns_per_min', 'first_speaker', 'last_speaker',
    'time_to_first_caller_s',
]


def _stats(values):
    if not values:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0}
    arr = np.asarray(values, dtype=float)
    return {
        'mean': float(arr.mean()),
        'std': float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        'min': float(arr.min()),
        'max': float(arr.max()),
        'median': float(np.median(arr)),
    }


def extract_features_from_turns(turns, duration_s):
    turns = sorted(turns, key=lambda t: t['start'])
    feats = {k: 0.0 for k in FEATURE_ORDER}
    feats['duration_s'] = float(duration_s)

    if not turns:
        return feats

    caller = [t for t in turns if t['channel'] == 0]
    agent = [t for t in turns if t['channel'] == 1]
    caller_durs = [t['end'] - t['start'] for t in caller]
    agent_durs = [t['end'] - t['start'] for t in agent]

    feats['n_turns_total'] = len(turns)
    feats['n_turns_caller'] = len(caller)
    feats['n_turns_agent'] = len(agent)
    feats['ratio_caller_agent_turns'] = len(caller) / len(agent) if agent else 0.0

    cs = _stats(caller_durs)
    feats['caller_dur_mean'] = cs['mean']
    feats['caller_dur_std'] = cs['std']
    feats['caller_dur_min'] = cs['min']
    feats['caller_dur_max'] = cs['max']
    feats['caller_dur_median'] = cs['median']
    feats['caller_dur_cv'] = cs['std'] / cs['mean'] if cs['mean'] else 0.0

    ag = _stats(agent_durs)
    feats['agent_dur_mean'] = ag['mean']
    feats['agent_dur_std'] = ag['std']
    feats['agent_dur_min'] = ag['min']
    feats['agent_dur_max'] = ag['max']
    feats['agent_dur_median'] = ag['median']

    feats['caller_total_speech_s'] = float(sum(caller_durs))
    feats['agent_total_speech_s'] = float(sum(agent_durs))
    feats['caller_speech_ratio'] = feats['caller_total_speech_s'] / duration_s if duration_s else 0.0
    feats['agent_speech_ratio'] = feats['agent_total_speech_s'] / duration_s if duration_s else 0.0

    caller_latencies = []
    agent_latencies = []
    silences = []
    overlaps = []
    caller_consecutive = 0
    agent_consecutive = 0

    for i in range(1, len(turns)):
        prev = turns[i - 1]
        curr = turns[i]
        gap = curr['start'] - prev['end']

        if gap > 0:
            silences.append(gap)
        elif gap < 0:
            overlaps.append(-gap)

        if prev['channel'] == 1 and curr['channel'] == 0:
            caller_latencies.append(gap)
        elif prev['channel'] == 0 and curr['channel'] == 1:
            agent_latencies.append(gap)

        if prev['channel'] == 0 and curr['channel'] == 0:
            caller_consecutive += 1
        if prev['channel'] == 1 and curr['channel'] == 1:
            agent_consecutive += 1

    cl = _stats(caller_latencies)
    feats['caller_latency_mean'] = cl['mean']
    feats['caller_latency_std'] = cl['std']
    feats['caller_latency_min'] = cl['min']
    feats['caller_latency_max'] = cl['max']
    feats['caller_latency_median'] = cl['median']
    feats['n_caller_responses'] = len(caller_latencies)

    al = _stats(agent_latencies)
    feats['agent_latency_mean'] = al['mean']
    feats['agent_latency_std'] = al['std']

    feats['n_overlaps'] = len(overlaps)
    feats['overlap_total_s'] = float(sum(overlaps))
    feats['overlap_mean'] = float(np.mean(overlaps)) if overlaps else 0.0

    feats['n_silences'] = len(silences)
    feats['silence_total_s'] = float(sum(silences))
    feats['silence_mean'] = float(np.mean(silences)) if silences else 0.0
    feats['silence_std'] = float(np.std(silences, ddof=1)) if len(silences) > 1 else 0.0
    feats['silence_ratio'] = feats['silence_total_s'] / duration_s if duration_s else 0.0

    feats['caller_consecutive_segments'] = caller_consecutive
    feats['agent_consecutive_segments'] = agent_consecutive
    feats['caller_turns_per_min'] = len(caller) / (duration_s / 60.0) if duration_s else 0.0

    feats['first_speaker'] = turns[0]['channel']
    feats['last_speaker'] = turns[-1]['channel']

    first_caller = next((t['start'] for t in turns if t['channel'] == 0), 0.0)
    feats['time_to_first_caller_s'] = float(first_caller)

    return feats


def _energy_vad(signal, sr, frame_ms=30, threshold_ratio=0.05, min_gap_s=0.2):
    frame = int(sr * frame_ms / 1000)
    if frame < 1 or len(signal) < frame:
        return []
    n = len(signal) // frame
    energies = np.array([
        np.sqrt(np.mean(signal[i * frame:(i + 1) * frame] ** 2))
        for i in range(n)
    ])
    if energies.max() <= 0:
        return []
    thr = energies.max() * threshold_ratio
    active = energies > thr

    segments = []
    start = None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        elif not a and start is not None:
            segments.append((start * frame_ms / 1000.0, i * frame_ms / 1000.0))
            start = None
    if start is not None:
        segments.append((start * frame_ms / 1000.0, n * frame_ms / 1000.0))

    merged = []
    for s, e in segments:
        if merged and s - merged[-1][1] < min_gap_s:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def _to_8k_int16(signal, sr):
    sig = np.asarray(signal, dtype=np.float32)
    if sr != 8000:
        n = int(round(len(sig) * 8000 / sr))
        if n < 1:
            return np.zeros(0, dtype=np.int16), 8000
        sig = np.interp(np.linspace(0, len(sig), n, endpoint=False),
                        np.arange(len(sig)), sig)
    sig = np.clip(sig, -1.0, 1.0)
    return (sig * 32767).astype(np.int16), 8000


def _webrtc_vad(signal, sr, aggressiveness=3, frame_ms=30, min_gap_s=0.2):
    import webrtcvad
    pcm, sr8 = _to_8k_int16(signal, sr)
    frame = int(sr8 * frame_ms / 1000)
    if frame < 1 or len(pcm) < frame:
        return []
    vad = webrtcvad.Vad(aggressiveness)
    n = len(pcm) // frame
    active = [vad.is_speech(pcm[i * frame:(i + 1) * frame].tobytes(), sr8) for i in range(n)]
    segments = []
    start = None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        elif not a and start is not None:
            segments.append((start * frame_ms / 1000.0, i * frame_ms / 1000.0))
            start = None
    if start is not None:
        segments.append((start * frame_ms / 1000.0, n * frame_ms / 1000.0))
    merged = []
    for s, e in segments:
        if merged and s - merged[-1][1] < min_gap_s:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def _segments(signal, sr):
    try:
        return _webrtc_vad(signal, sr)
    except Exception:
        return _energy_vad(signal, sr)


def turns_from_audio(caller_signal, agent_signal, sr):
    turns = []
    for start, end in _segments(caller_signal, sr):
        turns.append({'channel': 0, 'start': start, 'end': end})
    for start, end in _segments(agent_signal, sr):
        turns.append({'channel': 1, 'start': start, 'end': end})
    turns.sort(key=lambda t: t['start'])
    return turns
