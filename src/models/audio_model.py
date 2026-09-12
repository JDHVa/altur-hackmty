import torch
import torch.nn as nn
import torchaudio
import torchvision.models as models

class VoiceSpoofResNet(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=64, pretrained=True, dropout=0.3):
        super(VoiceSpoofResNet, self).__init__()
        
        # Capa para convertir el audio crudo en un espectrograma de Mel
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=n_mels
        )
        
        # Usamos ResNet18 como backbone (muy rapido para inferencia)
        # torchvision.models.resnet18 -> cambiado a 'weights' para versiones nuevas
        if pretrained:
            self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            self.resnet = models.resnet18(weights=None)
            
        # ResNet18 espera 3 canales (RGB), pero el espectrograma tiene 1 canal
        # Modificamos la primera capa convolucional
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Modificamos la ultima capa (fc) para clasificacion binaria (Humano vs IA).
        # Anadimos Dropout para reducir el sobreajuste (dataset chico: 282 calls).
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_ftrs, 1),  # Salida cruda (logit) para BCEWithLogitsLoss
        )

    def forward(self, x):
        # x shape: (batch, 1, time_samples)
        x = self.mel_spectrogram(x)
        # x shape: (batch, 1, n_mels, frames)
        
        # Convertimos a decibelios (escala logaritmica) para que la red vea mejor los detalles
        x = torchaudio.functional.amplitude_to_DB(x, multiplier=10.0, amin=1e-10, db_multiplier=0.0)
        
        # Pasamos por ResNet18
        out = self.resnet(x)
        return out
