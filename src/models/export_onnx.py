# -*- coding: utf-8 -*-
import os
import torch
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.audio_model import VoiceSpoofResNet

def export_to_onnx():
    print('[B3] Iniciando exportacion a ONNX...')
    
    device = torch.device('cpu')
    model = VoiceSpoofResNet(pretrained=False).to(device)
    
    model_path = os.path.join(os.path.dirname(__file__), 'saved', 'best_audio_resnet.pth')
    
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    
    # Compatibilidad con la refactorizacion (Dropout)
    if 'resnet.fc.weight' in state_dict:
        state_dict['resnet.fc.1.weight'] = state_dict.pop('resnet.fc.weight')
        state_dict['resnet.fc.1.bias'] = state_dict.pop('resnet.fc.bias')
        
    model.load_state_dict(state_dict)
    model.eval()
    
    # Batch size=1, Channels=1, Mel-bins=64, Time-frames=251 (aprox 8 seg a 16kHz)
    dummy_spectrogram = torch.randn(1, 1, 64, 251, device=device)
    onnx_path = os.path.join(os.path.dirname(__file__), 'saved', 'audio_resnet.onnx')
    
    print('Trazando la red neuronal (ResNet18 backbone)...')
    try:
        torch.onnx.export(
            model.resnet, # Solo exportamos el procesador de imagenes
            dummy_spectrogram, 
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['spectrogram_input'],
            output_names=['spoof_prob_logit'],
            dynamic_axes={
                'spectrogram_input': {0: 'batch_size', 3: 'time_frames'},
                'spoof_prob_logit': {0: 'batch_size'}
            }
        )
        print('\n[EXITO] ONNX guardado en:', onnx_path)
    except Exception as e:
        print('Error en exportacion:', e)

if __name__ == '__main__':
    export_to_onnx()
