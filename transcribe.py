#!/usr/bin/env python3
"""Transcription d'audios en texte (hors-ligne) avec faster-whisper.

Exemples :
    python transcribe.py                          # transcrit tout le dossier Files/
    python transcribe.py Files/A.ogg              # un seul fichier
    python transcribe.py Files -m small -l fr     # modèle "small", langue française
    python transcribe.py Files --srt              # génère aussi des sous-titres .srt

Les transcriptions sont écrites dans le dossier Transcriptions/ (un .txt par audio).
Aucun ffmpeg requis : le décodage audio est géré par PyAV (installé avec faster-whisper).
Installation : pip install -r requirements.txt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Téléchargement HTTP classique (et non « xet ») : permet d'annuler proprement un téléchargement
# de modèle en cours et de le reprendre ensuite. À définir avant tout import de huggingface_hub.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

AUDIO_EXTS = {".ogg", ".mp3", ".wav", ".m4a", ".flac", ".opus", ".aac", ".wma", ".webm", ".mp4", ".mkv"}
# Dossier de travail : à côté du .exe une fois compilé, à côté du script sinon.
BASE_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) \
    else Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "Files"
DEFAULT_OUTPUT = BASE_DIR / "Transcriptions"


def collect_audio_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in AUDIO_EXTS)
        elif p.is_file():
            files.append(p)
        else:
            print(f"[!] Introuvable : {p}", file=sys.stderr)
    return files


def fmt_ts(seconds: float, sep: str = ",") -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02}{sep}{ms:03}"


# Modèles proposés : nom -> (taille affichée, taille approximative en octets).
MODEL_INFO = {
    "tiny": ("75 Mo", 78_000_000),
    "base": ("145 Mo", 148_000_000),
    "small": ("485 Mo", 488_000_000),
    "medium": ("1,5 Go", 1_530_000_000),
    "large-v3-turbo": ("1,6 Go", 1_620_000_000),
    "large-v3": ("3,1 Go", 3_090_000_000),
}
DEFAULT_MODEL = "large-v3"
_MODEL_FILES = ["config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*"]


class DownloadCancelled(Exception):
    """Levée quand l'utilisateur annule un téléchargement de modèle."""


def models_dir() -> Path:
    """Dossier où sont stockés les modèles téléchargés par l'application."""
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "AudioTranscriber" / "models"


def model_path(name: str) -> Path | None:
    """Dossier local du modèle s'il est entièrement téléchargé, sinon None (aucun accès réseau).

    Cherche d'abord dans le dossier de l'application, puis dans le cache Hugging Face
    (modèles déjà téléchargés auparavant par faster-whisper).
    """
    own = models_dir() / name
    if (own / "model.bin").exists():          # model.bin n'est renommé qu'une fois complet
        return own
    try:
        from faster_whisper.utils import download_model
        p = Path(download_model(name, local_files_only=True))
        return p if (p / "model.bin").exists() else None
    except Exception:
        return None


def _http_get(url: str, headers: dict | None = None, timeout: int = 30):
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "AudioTranscriber/1.0", **(headers or {})})
    return urllib.request.urlopen(req, timeout=timeout)


def download_model_with_progress(name: str, on_progress=None, cancel=None) -> Path:
    """Télécharge un modèle faster-whisper depuis Hugging Face (reprise possible).

    on_progress(octets_reçus, octets_total) est appelé régulièrement. `cancel` est un
    threading.Event : une fois activé, le téléchargement s'arrête aussitôt (DownloadCancelled)
    en conservant ce qui a été reçu, pour une reprise ultérieure au même endroit.
    """
    import fnmatch
    import json
    import urllib.error
    from faster_whisper.utils import _MODELS

    if name not in _MODELS:
        raise ValueError(f"Modèle inconnu : {name}")
    repo = _MODELS[name]
    cancelled = lambda: cancel is not None and cancel.is_set()  # noqa: E731

    try:
        with _http_get(f"https://huggingface.co/api/models/{repo}?blobs=true") as r:
            siblings = json.load(r)["siblings"]
    except Exception as e:
        raise ConnectionError("Impossible de joindre Hugging Face : vérifiez votre connexion "
                              f"internet ({e}).") from e
    files = [(f["rfilename"], int(f.get("size") or 0)) for f in siblings
             if any(fnmatch.fnmatch(f["rfilename"], pat) for pat in _MODEL_FILES)]
    files.sort(key=lambda f: f[0] == "model.bin")      # model.bin en dernier
    total = sum(size for _, size in files)

    dest_dir = models_dir() / name
    dest_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    last = [0.0]

    def report(force: bool = False) -> None:
        now = time.time()
        if on_progress and (force or now - last[0] > 0.2):
            last[0] = now
            on_progress(done, total)

    for filename, size in files:
        dest, part = dest_dir / filename, dest_dir / (filename + ".part")
        if dest.exists() and (not size or dest.stat().st_size == size):
            done += dest.stat().st_size
            report(True)
            continue
        base_done = done
        for attempt in range(6):                      # reprise automatique en cas de coupure réseau
            if cancelled():
                raise DownloadCancelled()
            have = part.stat().st_size if part.exists() else 0
            try:
                headers = {"Range": f"bytes={have}-"} if have else {}
                with _http_get(f"https://huggingface.co/{repo}/resolve/main/{filename}", headers) as r:
                    if have and r.status != 206:      # le serveur ne gère pas la reprise : on repart de 0
                        have = 0
                    with open(part, "ab" if have else "wb") as out:
                        got = have
                        while True:
                            if cancelled():
                                raise DownloadCancelled()
                            chunk = r.read(1024 * 1024)
                            if not chunk:
                                break
                            out.write(chunk)
                            got += len(chunk)
                            done = base_done + got
                            report()
                if size and part.stat().st_size != size:
                    raise ConnectionError("fichier incomplet")
                break
            except DownloadCancelled:
                raise
            except (urllib.error.URLError, OSError, ConnectionError) as e:
                if attempt == 5:
                    raise ConnectionError(f"Téléchargement interrompu ({e}). Relancez-le pour reprendre.") from e
                time.sleep(2)
        part.replace(dest)
        done = base_done + dest.stat().st_size
        report(True)
    return dest_dir


def setup_cuda_dlls() -> None:
    """Rend visibles les DLL CUDA (cuBLAS/cuDNN) au programme, y compris une fois compilé en .exe."""
    candidates = []
    for var in ("CUDA_PATH", "CUDA_HOME"):
        if os.environ.get(var):
            candidates.append(Path(os.environ[var]) / "bin")
    toolkit = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA GPU Computing Toolkit" / "CUDA"
    if toolkit.is_dir():
        candidates += sorted(toolkit.glob("v*/bin"), reverse=True)
    for d in candidates:
        if d.is_dir() and hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(str(d))
                os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
            except OSError:
                pass


def load_model(name: str, device: str = "cuda", compute_type: str = "default", log=print):
    """Charge le modèle, sur le GPU NVIDIA par défaut, avec repli automatique sur le CPU.

    Retourne (modèle, périphérique_utilisé, type_de_calcul).
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("faster-whisper n'est pas installé : pip install -r requirements.txt")

    src = str(model_path(name) or name)   # modèle local si présent : aucun accès réseau

    if device in ("cuda", "auto"):
        setup_cuda_dlls()
        ct = "float16" if compute_type == "default" else compute_type
        try:
            log(f"Chargement du modèle '{name}' sur le GPU NVIDIA ({ct})...")
            model = WhisperModel(src, device="cuda", compute_type=ct)
            # Test à blanc : les DLL CUDA ne sont chargées qu'au premier calcul.
            import numpy as np
            segs, _ = model.transcribe(np.zeros(16000, dtype=np.float32))
            list(segs)
            return model, "cuda", ct
        except Exception as e:
            if device == "cuda":
                log(f"[!] GPU indisponible ({e}) -> repli sur le CPU.")
            else:
                log(f"GPU non utilisable ({e}) -> CPU.")

    ct = "int8" if compute_type in ("default", "float16") else compute_type
    log(f"Chargement du modèle '{name}' sur le CPU ({ct})...")
    return WhisperModel(src, device="cpu", compute_type=ct), "cpu", ct


def transcribe_file(model, audio: Path, out_dir: Path, language: str | None,
                    timestamps: bool, srt: bool, beam_size: int, vad: bool) -> None:
    t0 = time.time()
    segments, info = model.transcribe(
        str(audio),
        language=language,
        beam_size=beam_size,
        vad_filter=vad,
    )
    print(f"\n=== {audio.name} === langue: {info.language} "
          f"(confiance {info.language_probability:.0%}), durée: {info.duration:.1f}s")

    lines, srt_blocks = [], []
    for i, seg in enumerate(segments, 1):
        text = seg.text.strip()
        line = f"[{fmt_ts(seg.start, '.')} -> {fmt_ts(seg.end, '.')}] {text}" if timestamps else text
        print(line)
        lines.append(line)
        srt_blocks.append(f"{i}\n{fmt_ts(seg.start)} --> {fmt_ts(seg.end)}\n{text}\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / f"{audio.stem}.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    written = [txt_path]
    if srt:
        srt_path = out_dir / f"{audio.stem}.srt"
        srt_path.write_text("\n".join(srt_blocks), encoding="utf-8")
        written.append(srt_path)
    print(f"-> {', '.join(str(w) for w in written)} ({time.time() - t0:.1f}s)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcrit des fichiers audio en texte (faster-whisper).")
    ap.add_argument("inputs", nargs="*", type=Path, default=[DEFAULT_INPUT],
                    help="fichiers ou dossiers audio (défaut : Files/)")
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT, help="dossier de sortie")
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL,
                    help="tiny, base, small, medium, large-v3, ... (défaut : large-v3)")
    ap.add_argument("-l", "--language", default=None,
                    help="code langue (fr, en, ...) ; détection automatique si omis")
    ap.add_argument("--device", default="cuda", choices=["cuda", "auto", "cpu"],
                    help="cuda = GPU NVIDIA avec repli CPU (défaut), cpu = CPU uniquement")
    ap.add_argument("--compute-type", default="default",
                    help="default (float16 sur GPU, int8 sur CPU), int8_float16, float32...")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--no-vad", action="store_true", help="désactive le filtre de détection de voix")
    ap.add_argument("-t", "--timestamps", action="store_true", help="horodatage dans le .txt")
    ap.add_argument("--srt", action="store_true", help="génère aussi un fichier .srt")
    args = ap.parse_args()

    files = collect_audio_files(args.inputs)
    if not files:
        sys.exit("Aucun fichier audio trouvé.")

    if model_path(args.model) is None and args.model in MODEL_INFO:
        print(f"Modèle '{args.model}' absent : téléchargement ({MODEL_INFO[args.model][0]})...")

        def show(done: int, total: int) -> None:
            if total:
                print(f"\r  {done / 1e6:,.0f} / {total / 1e6:,.0f} Mo ({done / total:.0%})", end="", flush=True)
        download_model_with_progress(args.model, show)
        print()
    model, _device, _compute = load_model(args.model, args.device, args.compute_type)

    for f in files:
        try:
            transcribe_file(model, f, args.output, args.language,
                            args.timestamps, args.srt, args.beam_size, not args.no_vad)
        except Exception as e:  # un fichier défaillant ne bloque pas les suivants
            print(f"[!] Échec sur {f.name} : {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
