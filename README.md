# Audio_Extract — Transcription d'audios en texte

Application Windows (interface graphique moderne, thème sombre, en [PySide6 / Qt 6](https://doc.qt.io/qtforpython-6/)) qui transcrit des fichiers audio (`.ogg`, `.mp3`, `.wav`, `.m4a`, `.flac`, ...) en texte, **hors-ligne**, grâce à [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (modèle Whisper d'OpenAI).

- Lancement par double-clic sur **`dist/AudioTranscriber.exe`** (rien à installer).
- Ajout des audios par **glisser-déposer**, par **chemin d'accès** ou en cliquant sur la zone de dépôt.
- Aucune clé API, aucun envoi de données (sauf le téléchargement du modèle au premier usage).
- Langue détectée automatiquement (ou choisie dans la liste).

## Structure du projet

```
Audio_Extract/
├── dist/AudioTranscriber.exe   # application prête à l'emploi (générée par build_exe.bat)
├── app.py                      # interface graphique (PySide6)
├── transcribe.py               # moteur de transcription + version ligne de commande
├── build_exe.bat               # compile app.py en .exe
├── requirements.txt            # dépendances
├── Files/                      # exemples d'audios (A.ogg, B.ogg)
└── Transcriptions/             # résultats générés (.txt, .srt)
```

## Utiliser l'application (.exe)

1. Double-cliquez sur `dist/AudioTranscriber.exe` (le démarrage prend quelques secondes).
2. Ajoutez un ou plusieurs audios de l'une de ces façons :
   - **glissez-déposez** les fichiers (ou un dossier) n'importe où dans la fenêtre (la zone en pointillés s'illumine) ;
   - **collez le chemin** dans le champ « Chemin » (les guillemets de « Copier en tant que chemin d'accès » sont acceptés) puis `Entrée` ou **Ajouter** (un chemin tapé mais pas ajouté est pris en compte au clic sur Transcrire) ;
   - **cliquez sur la zone de dépôt** pour ouvrir l'explorateur de fichiers.
3. Choisissez le modèle (par défaut **`large-v3`**, le plus précis) et la langue, puis cliquez sur **▶ Transcrire**.
4. Le texte s'affiche au fur et à mesure. Boutons **Copier le texte** et **Ouvrir le dossier**.

Pour chaque audio `X.ogg`, l'application écrit `X.txt` (et `X.srt` si l'option est cochée) dans le dossier de sortie, par défaut `Transcriptions/` à côté du `.exe`.

### Choisir et télécharger un modèle

La liste **Modèle** indique pour chacun sa taille et son état : `✓ installé` ou `à télécharger`.

- **Télécharger à l'avance :** sélectionnez un modèle « à télécharger » puis cliquez sur **⬇ Télécharger**.
- **Télécharger à la demande :** cliquez sur **Transcrire** avec un modèle non installé ; l'application propose de le télécharger, puis lance la transcription toute seule à la fin.
- **Suivi en direct :** la barre de progression et la ligne d'état affichent l'avancement, par exemple
  `Téléchargement de « large-v3 » : 1,24 Go / 3,09 Go (40%) · 3,8 Mo/s · reste 8 min 05 s`.
- **Annuler / reprendre :** **Annuler** arrête le téléchargement immédiatement. En le relançant, il **reprend là où il s'était arrêté** (reprise aussi automatique en cas de coupure réseau).
- Un modèle téléchargé reste disponible **hors-ligne**. Il est stocké dans `%LOCALAPPDATA%\AudioTranscriber\models\` (les modèles déjà présents dans le cache Hugging Face sont aussi reconnus). Pour libérer de la place, supprimez le dossier du modèle voulu.

### Options de l'interface

| Option | Description |
|---|---|
| Modèle | `tiny`, `base`, `small`, `medium`, `large-v3-turbo`, `large-v3` (défaut) |
| Langue | Détection automatique ou langue précise (le français est présélectionné) |
| Horodatage dans le texte | Ajoute `[début -> fin]` devant chaque phrase |
| Générer aussi un .srt | Fichier de sous-titres horodatés |
| Dossier de sortie | Où enregistrer les fichiers |

Le bouton **Annuler** arrête la transcription après le segment en cours, ou le téléchargement du modèle s'il y en a un.

### Calcul sur le GPU NVIDIA

À chaque exécution, l'application charge le modèle **sur le GPU NVIDIA** (précision `float16`) ; la barre d'état affiche « GPU NVIDIA » pendant la transcription. Si le GPU ne peut pas être utilisé, elle bascule automatiquement sur le CPU (plus lent) et l'indique à la place.

Prérequis pour le GPU : pilote NVIDIA à jour et **CUDA Toolkit 12.x** installé (fournit `cublas64_12.dll`, détecté via `CUDA_PATH`). Testé avec une RTX 4070 Laptop (8 Go) et CUDA 12.9. Sur un autre PC sans CUDA Toolkit, l'application fonctionne mais sur le CPU.

Avec 8 Go de VRAM, tous les modèles proposés devraient tenir sur le GPU, y compris `large-v3` (le plus précis, ~3 Go en `float16`).

### Quel modèle choisir ?

| Modèle | Taille | Vitesse | Qualité |
|---|---|---|---|
| `tiny` | ~75 Mo | très rapide | faible |
| `base` | ~145 Mo | rapide | moyenne |
| `small` | ~485 Mo | correcte | bonne |
| `medium` | ~1,5 Go | rapide sur GPU, lente sur CPU | très bonne |
| `large-v3-turbo` | ~1,6 Go | rapide sur GPU | proche de `large-v3` |
| `large-v3` | ~3,1 Go | correcte sur GPU, très lente sur CPU | la meilleure (défaut) |

Pour du français parlé familier ou bruité, `medium` et surtout `large-v3` sont nettement meilleurs que `base`.

## Lancer depuis les sources (sans .exe)

Prérequis : Python 3.9 ou plus récent.

```bash
pip install -r requirements.txt
python app.py
```

## Recompiler le .exe

```bash
pip install pyinstaller
build_exe.bat
```

Le résultat est `dist/AudioTranscriber.exe` (~165 Mo : il embarque Python, Qt et le moteur de transcription, mais pas les modèles). Il peut être copié sur un autre PC Windows sans rien installer.

## Mode ligne de commande

Si on passe des arguments, le programme (script ou `.exe`) fonctionne sans fenêtre :

```bash
python transcribe.py                          # tout le dossier Files/
python transcribe.py Files/A.ogg -m medium -l fr -t --srt
dist\AudioTranscriber.exe Files -m large-v3-turbo -o resultats
```

| Option | Description | Défaut |
|---|---|---|
| `inputs` | Fichiers ou dossiers audio | `Files/` |
| `-o`, `--output` | Dossier de sortie | `Transcriptions/` |
| `-m`, `--model` | Modèle Whisper (téléchargé automatiquement avec une barre de progression en console s'il manque) | `large-v3` |
| `-l`, `--language` | Code langue (`fr`, `en`, ...) | auto |
| `-t`, `--timestamps` | Horodatage dans le `.txt` | non |
| `--srt` | Génère aussi un `.srt` | non |
| `--device` | `cuda` (GPU NVIDIA, repli CPU), `cpu` (CPU uniquement) | `cuda` |
| `--compute-type` | `float16`, `int8_float16`, `int8`, ... | `float16` (GPU), `int8` (CPU) |
| `--beam-size` | Précision du décodage | `5` |
| `--no-vad` | Désactive le filtre des silences | filtre actif |

Avec le `.exe` (sans console), l'avancement n'est pas affiché : seuls les fichiers de sortie sont créés.

## Problèmes courants

- **Windows SmartScreen / antivirus bloque l'exe** : normal pour un programme non signé compilé avec PyInstaller. « Informations complémentaires » puis « Exécuter quand même ».
- **Démarrage lent de l'exe** : c'est un exécutable « tout-en-un » qui se décompresse à chaque lancement.
- **Le glisser-déposer ne fait rien** : si l'application est lancée « en tant qu'administrateur », Windows bloque le glisser-déposer depuis l'Explorateur ; relancez-la normalement ou utilisez le chemin ou un clic sur la zone de dépôt.
- **`open() got an unexpected keyword argument 'metadata_errors'`** (depuis les sources) : `pip install "av>=15,<17"`.
- **Transcription de mauvaise qualité** : utilisez un modèle plus gros et précisez la langue.
- **« Impossible de joindre Hugging Face »** : le téléchargement d'un modèle nécessite internet (et que `huggingface.co` ne soit pas bloqué par un pare-feu). Relancez-le : il reprend où il s'était arrêté.
