import os
import urllib.request
import zipfile
import tarfile
from pathlib import Path

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"[{dest_path.name}] Ya existe. Omitiendo descarga.")
        return True
        
    print(f"Descargando {dest_path.name} desde {url}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"[{dest_path.name}] Descarga completada.")
        return True
    except Exception as e:
        print(f"Error descargando {dest_path.name}: {e}")
        return False

def extract_file(file_path, extract_to):
    print(f"Extrayendo {file_path.name}...")
    try:
        if file_path.name.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif file_path.name.endswith('.tar.gz') or file_path.name.endswith('.tgz'):
            with tarfile.open(file_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_to)
        print(f"[{file_path.name}] Extracción completada en {extract_to}")
    except Exception as e:
        print(f"Error extrayendo {file_path.name}: {e}")

def main():
    base_dir = Path("C:/Users/jesus/Proyectos/Altur/datasets_externos")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # URLs de los datasets
    datasets = {
        "WaveFake": {
            "url": "https://zenodo.org/record/5642694/files/WaveFake.zip",
            "filename": "WaveFake.zip"
        },
        "ASVspoof_2019_LA": {
            # URL oficial en Edinburgh DataShare
            "url": "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip",
            "filename": "ASVspoof2019_LA.zip"
        }
    }
    
    print("=== Iniciando descarga de Datasets Externos ===")
    
    for name, info in datasets.items():
        print(f"\n--- Procesando {name} ---")
        dest_file = base_dir / info["filename"]
        extract_dir = base_dir / name
        
        if download_file(info["url"], dest_file):
            if not extract_dir.exists():
                extract_dir.mkdir(parents=True, exist_ok=True)
                extract_file(dest_file, extract_dir)
            else:
                print(f"El directorio {extract_dir.name} ya existe. Omitiendo extracción.")

if __name__ == '__main__':
    main()
