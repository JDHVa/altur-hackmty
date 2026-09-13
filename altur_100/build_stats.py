import os
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'feats_cache.csv')
OUT = os.path.join(HERE, 'src', 'models', 'saved', 'feature_stats.json')

FEATURES = [
    'time_to_first_caller_s', 'caller_latency_mean', 'caller_latency_median',
    'n_overlaps', 'overlap_total_s', 'n_turns_caller', 'caller_turns_per_min',
    'silence_mean', 'xlsr', 'wavlm',
]


def main():
    df = pd.read_csv(CACHE).fillna(0.0)
    y = df['y'].values
    stats = {}
    for f in FEATURES:
        if f not in df.columns:
            continue
        h = df[f].values[y == 0]
        s = df[f].values[y == 1]
        stats[f] = {
            'human_mean': float(np.mean(h)), 'human_std': float(np.std(h) + 1e-6),
            'synth_mean': float(np.mean(s)), 'synth_std': float(np.std(s) + 1e-6),
        }
    json.dump(stats, open(OUT, 'w'), indent=1)
    print('guardado', OUT, 'con', len(stats), 'features')


if __name__ == '__main__':
    main()
