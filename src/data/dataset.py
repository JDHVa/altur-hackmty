import os
import json
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset

class AlturAudioDataset(Dataset):
    def __init__(self, manifest_path, audio_dir, turns_dir, split='train', target_sr=16000, max_duration_s=10):
        self.audio_dir = audio_dir
        self.turns_dir = turns_dir
        self.target_sr = target_sr
        self.max_frames = target_sr * max_duration_s
        
        df = pd.read_csv(manifest_path)
        if split:
            df = df[df['split'] == split].reset_index(drop=True)
            
        self.data = df
        self.label_map = {'human': 0, 'synthetic': 1}

    def __len__(self):
        return len(self.data)
        
    def _extract_tabular_features(self, anon_id):
        json_path = os.path.join(self.turns_dir, f"{anon_id}.json")
        if not os.path.exists(json_path):
            return torch.zeros(4)
            
        with open(json_path, 'r') as f:
            turns = json.load(f).get('turns', [])
            
        if not turns:
             return torch.zeros(4)
             
        latencies, overlaps, caller_durations = [], [], []
        
        for i in range(1, len(turns)):
            prev_turn = turns[i-1]
            curr_turn = turns[i]
            
            if curr_turn['channel'] == 0 and prev_turn['channel'] == 1:
                latency = curr_turn['start'] - prev_turn['end']
                if latency < 0:
                    overlaps.append(abs(latency))
                    latencies.append(0)
                else:
                    latencies.append(latency)
                    overlaps.append(0)
            
            if curr_turn['channel'] == 0:
                caller_durations.append(curr_turn['end'] - curr_turn['start'])
                
        avg_latency = sum(latencies)/len(latencies) if latencies else 0
        avg_overlap = sum(overlaps)/len(overlaps) if overlaps else 0
        avg_duration = sum(caller_durations)/len(caller_durations) if caller_durations else 0
        num_turns = len([t for t in turns if t['channel'] == 0])
        
        return torch.tensor([avg_latency, avg_overlap, avg_duration, num_turns], dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        anon_id = row['anon_id']
        label_str = row['label']
        label = self.label_map.get(label_str, -1)
        
        wav_path = os.path.join(self.audio_dir, f"{anon_id}.wav")
        waveform, sr = torchaudio.load(wav_path)
        
        caller_audio = waveform[0:1, :]
        
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            caller_audio = resampler(caller_audio)
            
        if caller_audio.shape[1] > self.max_frames:
            caller_audio = caller_audio[:, :self.max_frames]
        else:
            padding = self.max_frames - caller_audio.shape[1]
            caller_audio = torch.nn.functional.pad(caller_audio, (0, padding))
            
        tabular_features = self._extract_tabular_features(anon_id)
            
        return {
            'id': anon_id,
            'audio': caller_audio,
            'features': tabular_features,
            'label': torch.tensor(label, dtype=torch.long)
        }
