# -*- coding: utf-8 -*-
import os
import csv
import argparse

ROOT = os.path.join(os.path.dirname(__file__), '..')


def main():
    ap = argparse.ArgumentParser(description='Evalua una config de pesos+umbral sobre demo_scores.csv')
    ap.add_argument('--wavlm', type=float, default=0.25)
    ap.add_argument('--xlsr', type=float, default=0.40)
    ap.add_argument('--prosody', type=float, default=0.25)
    ap.add_argument('--flow', type=float, default=0.10)
    ap.add_argument('--threshold', type=float, default=0.30)
    ap.add_argument('--csv', default=os.path.join(ROOT, 'demo_scores.csv'))
    ap.add_argument('--show', action='store_true', help='listar cada audio')
    args = ap.parse_args()

    w = {'wavlm': args.wavlm, 'xlsr': args.xlsr, 'prosody': args.prosody, 'flow': args.flow}
    tw = sum(w.values())
    rows = list(csv.DictReader(open(args.csv, encoding='utf-8')))

    fp, fn, tp, tn = 0, 0, 0, 0
    mistakes = []
    for r in rows:
        s = sum(w[k] * float(r[k]) for k in w) / tw
        pred = 1 if s >= args.threshold else 0
        lab = int(r['label'])
        if pred == 1 and lab == 1:
            tp += 1
        elif pred == 0 and lab == 0:
            tn += 1
        elif pred == 1 and lab == 0:
            fp += 1
            mistakes.append((r['file'], 'REAL->IA', round(s, 3)))
        else:
            fn += 1
            mistakes.append((r['file'], 'IA->humano', round(s, 3)))
        if args.show:
            print(f"  {r['file']:32s} lab={lab} score={s:.3f} pred={'IA' if pred else 'HUM'}")

    n = len(rows)
    acc = (tp + tn) / n if n else 0
    print(f'\ncfg wavlm={args.wavlm} xlsr={args.xlsr} prosody={args.prosody} flow={args.flow} thr={args.threshold}')
    print(f'  N={n}  ACC={acc:.3f}  | IA cazadas={tp} falladas(FN)={fn}  | humanos ok={tn} mal(FP)={fp}')
    print(f'  FALSO NEGATIVO (IA pasa como humano) = {fn}  <- el error grave para el banco')
    if mistakes:
        print('  errores:')
        for m in mistakes:
            print(f'    {m[0]:32s} {m[1]:12s} score={m[2]}')


if __name__ == '__main__':
    main()
