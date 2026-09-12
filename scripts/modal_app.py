import modal

app = modal.App("altur-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch", "torchaudio", "torchvision",
        "transformers", "faster-whisper",
        "librosa", "soundfile", "praat-parselmouth",
        "webrtcvad-wheels", "imageio-ffmpeg",
        "scikit-learn", "lightgbm", "xgboost", "joblib",
        "pandas", "numpy", "tqdm", "datasets",
    )
    .add_local_dir("src", "/app/src")
    .add_local_dir("scripts", "/app/scripts")
)

data_vol = modal.Volume.from_name("altur-data", create_if_missing=True)
models_vol = modal.Volume.from_name("altur-models", create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=7200,
              volumes={"/data": data_vol, "/models": models_vol})
def train(script: str, args: list[str]):
    import os
    import shutil
    import subprocess

    shutil.copytree("/app", "/work", dirs_exist_ok=True)
    os.chdir("/work")
    for d in ("hackmty26", "datasets_externos"):
        src = f"/data/{d}"
        dst = f"/work/{d}"
        if os.path.isdir(src) and not os.path.exists(dst):
            os.symlink(src, dst)

    subprocess.run(["python", script, *args], check=True)

    saved = "/work/src/models/saved"
    if os.path.isdir(saved):
        shutil.copytree(saved, "/models/saved", dirs_exist_ok=True)
        models_vol.commit()
        print("modelos copiados a volume altur-models:/saved")


@app.local_entrypoint()
def main(script: str, args: str = ""):
    train.remote(script, args.split() if args else [])
