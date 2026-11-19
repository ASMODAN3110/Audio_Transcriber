#!/usr/bin/env python3
"""Interface graphique (PySide6 / Qt 6) de transcription audio -> texte (faster-whisper).

Lancement : python app.py   (ou double-clic sur AudioTranscriber.exe)
Si des arguments sont passés (ex. AudioTranscriber.exe Files -m base), le programme
se comporte comme le script en ligne de commande transcribe.py.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

# En mode fenêtré (.exe sans console), stdout/stderr valent None : on les redirige
# pour éviter les plantages des bibliothèques qui y écrivent (tqdm, huggingface_hub).
for _name in ("stdout", "stderr"):
    if getattr(sys, _name) is None:
        setattr(sys, _name, open(os.devnull, "w", encoding="utf-8"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import transcribe as core  # noqa: E402

LANGUAGES = {
    "Détection automatique": None, "Français": "fr", "Anglais": "en", "Espagnol": "es",
    "Allemand": "de", "Italien": "it", "Portugais": "pt", "Arabe": "ar",
}
MODELS = list(core.MODEL_INFO)          # tiny, base, small, medium, large-v3-turbo, large-v3

STYLE = """
* { font-family: "Segoe UI", sans-serif; font-size: 10pt; color: #e6e9f0; }
QMainWindow, QWidget#root { background: #12151c; }
QLabel#title { font-size: 18pt; font-weight: 700; }
QLabel#subtitle, QLabel#hint { color: #8b93a7; }
QFrame#card { background: #1a1f2b; border: 1px solid #272e3d; border-radius: 12px; }
QLabel#cardTitle { font-weight: 600; color: #aab3c8; }
QLineEdit, QComboBox, QListWidget, QPlainTextEdit {
    background: #12151c; border: 1px solid #2c3446; border-radius: 8px; padding: 6px 10px;
    selection-background-color: #3b6fe0;
}
QLineEdit, QComboBox { min-height: 24px; }
QPushButton { min-height: 22px; }
QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus { border-color: #4f8cff; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox QAbstractItemView { background: #1a1f2b; border: 1px solid #2c3446;
    selection-background-color: #3b6fe0; outline: none; }
QListWidget { padding: 4px; }
QListWidget::item { padding: 5px 6px; border-radius: 6px; }
QListWidget::item:selected { background: #26324d; }
QPushButton { background: #232a3a; border: 1px solid #2c3446; border-radius: 8px; padding: 8px 14px; }
QPushButton:hover { background: #2b3448; }
QPushButton:disabled { color: #5c6478; background: #1a1f2b; }
QPushButton#primary { background: #4f8cff; border: none; font-weight: 600; color: white; padding: 10px 22px; }
QPushButton#primary:hover { background: #6a9dff; }
QPushButton#primary:disabled { background: #2a3550; color: #7d89a6; }
QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 18px; height: 18px; border-radius: 5px; border: 1px solid #3a4560; background: #12151c; }
QCheckBox::indicator:checked { background: #4f8cff; border-color: #4f8cff; }
QProgressBar { background: #1a1f2b; border: none; border-radius: 4px; height: 8px; text-align: center; color: transparent; }
QProgressBar::chunk { background: #4f8cff; border-radius: 4px; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: #333c52; border-radius: 4px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""

from PySide6.QtCore import Qt, QThread, QUrl, Signal  # noqa: E402
from PySide6.QtGui import QDesktopServices, QGuiApplication, QTextCursor  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QVBoxLayout, QWidget,
)


class DropZone(QFrame):
    """Zone de dépôt : glisser des fichiers dessus, ou cliquer pour parcourir."""
    clicked = Signal()

    IDLE = ("QFrame#drop { background: #161b27; border: 2px dashed #34405c; border-radius: 14px; }"
            "QFrame#drop:hover { border-color: #4f8cff; background: #182033; }")
    ACTIVE = "QFrame#drop { background: #1b2a4d; border: 2px solid #4f8cff; border-radius: 14px; }"

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("drop")
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(96)
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        self.icon = QLabel("🎙")
        self.icon.setAlignment(Qt.AlignCenter)
        self.icon.setStyleSheet("font-size: 20pt; background: transparent;")
        self.label = QLabel("Glissez-déposez vos fichiers audio ici")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-size: 12pt; font-weight: 600; background: transparent;")
        sub = QLabel("ou cliquez pour parcourir  ·  .ogg .mp3 .wav .m4a .flac …")
        sub.setObjectName("hint")
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet("background: transparent; color: #8b93a7;")
        for w in (self.icon, self.label, sub):
            lay.addWidget(w)
        self.set_active(False)

    def set_active(self, active: bool) -> None:
        self.setStyleSheet(self.ACTIVE if active else self.IDLE)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self.clicked.emit()


class Worker(QThread):
    """Charge le modèle (GPU) et transcrit les fichiers dans un thread séparé."""
    busy = Signal(str)
    status = Signal(str)
    progress = Signal(float)
    text = Signal(str)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, settings: dict, cache: dict) -> None:
        super().__init__()
        self.s = settings
        self.cache = cache          # {"model", "name", "device"} partagé entre les exécutions
        self.cancel_event = threading.Event()

    def run(self) -> None:
        s = self.s
        try:
            if self.cache.get("model") is None or self.cache.get("name") != s["model"]:
                self.busy.emit(f"Chargement du modèle « {s['model']} » sur le GPU…")
                model, device, _ = core.load_model(s["model"], "cuda", log=self.busy.emit)
                self.cache.update(model=model, name=s["model"], device=device)
            self.progress.emit(0)
            s["out_dir"].mkdir(parents=True, exist_ok=True)
            n = len(s["files"])
            for idx, audio in enumerate(s["files"]):
                if self.cancel_event.is_set():
                    break
                try:
                    self._transcribe_one(audio, idx, n)
                except Exception as e:  # un fichier en échec ne bloque pas les suivants
                    self.text.emit(f"[!] Échec sur {audio.name} : {e}\n\n")
            self.finished_ok.emit("Annulé." if self.cancel_event.is_set()
                                  else f"Terminé. Fichiers enregistrés dans : {s['out_dir']}")
        except Exception as e:
            self.failed.emit(f"Erreur : {e}")

    def _transcribe_one(self, audio: Path, idx: int, n: int) -> None:
        s = self.s
        hw = "GPU NVIDIA" if self.cache["device"] == "cuda" else "CPU — GPU indisponible"
        self.status.emit(f"[{idx + 1}/{n}] Transcription de {audio.name} ({hw})…")
        segments, info = self.cache["model"].transcribe(
            str(audio), language=s["language"], beam_size=5, vad_filter=True)
        self.text.emit(f"=== {audio.name} (langue : {info.language}) ===\n")
        lines, srt_blocks = [], []
        for i, seg in enumerate(segments, 1):
            if self.cancel_event.is_set():
                return
            text = seg.text.strip()
            line = (f"[{core.fmt_ts(seg.start, '.')} -> {core.fmt_ts(seg.end, '.')}] {text}"
                    if s["timestamps"] else text)
            lines.append(line)
            srt_blocks.append(f"{i}\n{core.fmt_ts(seg.start)} --> {core.fmt_ts(seg.end)}\n{text}\n")
            self.text.emit(line + "\n")
            frac = min(seg.end / info.duration, 1.0) if info.duration else 0
            self.progress.emit((idx + frac) / n * 100)
        (s["out_dir"] / f"{audio.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if s["srt"]:
            (s["out_dir"] / f"{audio.stem}.srt").write_text("\n".join(srt_blocks), encoding="utf-8")
        self.text.emit("\n")
        self.progress.emit((idx + 1) / n * 100)


def fmt_size(n: float) -> str:
    return (f"{n / 1e9:.2f} Go" if n >= 1e9 else f"{n / 1e6:.0f} Mo").replace(".", ",")


def fmt_eta(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds} s"
    if seconds < 3600:
        return f"{seconds // 60} min {seconds % 60:02d} s"
    return f"{seconds // 3600} h {seconds % 3600 // 60:02d} min"


class DownloadWorker(QThread):
    """Télécharge un modèle dans un thread séparé (progression, annulation, reprise)."""
    progress = Signal(float, float)     # octets reçus, octets total (float : pas de limite à 32 bits)
    finished_ok = Signal()
    cancelled = Signal()
    failed = Signal(str)

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.cancel_event = threading.Event()

    def run(self) -> None:
        try:
            core.download_model_with_progress(
                self.name, lambda done, total: self.progress.emit(done, total), self.cancel_event)
            self.finished_ok.emit()
        except core.DownloadCancelled:
            self.cancelled.emit()
        except Exception as e:
            self.failed.emit(str(e))


def card(title: str | None = None) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("card")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(14, 10, 14, 10)
    lay.setSpacing(7)
    if title:
        t = QLabel(title.upper())
        t.setObjectName("cardTitle")
        lay.addWidget(t)
    return frame, lay


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Audio Transcriber")
        avail = QGuiApplication.primaryScreen().availableGeometry()
        self.resize(920, min(820, avail.height() - 40))
        self.setAcceptDrops(True)

        self.files: list[Path] = []
        self.cache: dict = {}
        self.worker: Worker | None = None
        self.dl: DownloadWorker | None = None
        self._dl_chain = False                # lancer la transcription une fois le modèle téléchargé
        self._dl_samples: deque = deque()
        self._busy = False                    # transcription ou téléchargement en cours
        self._build()

    # ------------------------------------------------------------------ UI
    def _build(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        main = QVBoxLayout(root)
        main.setContentsMargins(22, 14, 22, 14)
        main.setSpacing(9)

        title = QLabel("Audio Transcriber")
        title.setObjectName("title")
        sub = QLabel("Transcrivez vos audios en texte, hors-ligne, sur votre GPU NVIDIA.")
        sub.setObjectName("subtitle")
        main.addWidget(title)
        main.addWidget(sub)

        # Zone de dépôt + chemin
        self.drop = DropZone()
        self.drop.clicked.connect(self._browse)
        main.addWidget(self.drop)

        row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Ou collez ici le chemin d'accès d'un fichier / dossier audio…")
        self.path_edit.setAcceptDrops(False)   # un fichier déposé sur le champ est ajouté à la liste
        self.path_edit.returnPressed.connect(self._add_typed_path)
        add_btn = QPushButton("Ajouter")
        add_btn.clicked.connect(self._add_typed_path)
        row.addWidget(self.path_edit, 1)
        row.addWidget(add_btn)
        main.addLayout(row)

        # Liste des fichiers
        files_card, fl = card("Fichiers à transcrire")
        self.listw = QListWidget()
        self.listw.setSelectionMode(QListWidget.ExtendedSelection)
        self.listw.setFixedHeight(76)
        rm = QPushButton("Retirer")
        rm.clicked.connect(self._remove_selected)
        clr = QPushButton("Vider")
        clr.clicked.connect(self._clear)
        fr = QHBoxLayout()
        fr.addWidget(self.listw, 1)
        col = QVBoxLayout()
        col.addWidget(rm)
        col.addWidget(clr)
        fr.addLayout(col)
        fl.addLayout(fr)
        main.addWidget(files_card)

        # Options
        opt_card, ol = card("Options")
        r1 = QHBoxLayout()
        self.model_box = QComboBox()
        self.model_box.setMinimumWidth(360)
        self.model_box.setToolTip("Plus le modèle est gros, plus la transcription est précise (et plus il est "
                                  "lourd à télécharger). « large-v3-turbo » est un très bon compromis vitesse/qualité.")
        self.model_box.currentIndexChanged.connect(lambda _i: self._update_dl_button())
        self.dl_btn = QPushButton("⬇  Télécharger")
        self.dl_btn.clicked.connect(lambda: self._begin_download(self.model_box.currentData(), False))
        r1.addWidget(QLabel("Modèle"))
        r1.addWidget(self.model_box, 1)
        r1.addWidget(self.dl_btn)
        ol.addLayout(r1)
        r2 = QHBoxLayout()
        self.lang_box = QComboBox()
        self.lang_box.addItems(list(LANGUAGES))
        self.lang_box.setCurrentText("Français")
        self.ts_check = QCheckBox("Horodatage dans le texte")
        self.srt_check = QCheckBox("Générer aussi un .srt")
        r2.addWidget(QLabel("Langue"))
        r2.addWidget(self.lang_box)
        r2.addSpacing(16)
        r2.addWidget(self.ts_check)
        r2.addSpacing(8)
        r2.addWidget(self.srt_check)
        r2.addStretch(1)
        ol.addLayout(r2)
        r3 = QHBoxLayout()
        self.out_edit = QLineEdit(str(core.DEFAULT_OUTPUT))
        self.out_edit.setAcceptDrops(False)
        out_btn = QPushButton("Changer…")
        out_btn.clicked.connect(self._choose_output)
        r3.addWidget(QLabel("Dossier de sortie"))
        r3.addWidget(self.out_edit, 1)
        r3.addWidget(out_btn)
        ol.addLayout(r3)
        main.addWidget(opt_card)

        # Actions
        act = QHBoxLayout()
        self.start_btn = QPushButton("▶  Transcrire")
        self.start_btn.setObjectName("primary")
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.clicked.connect(self._start)
        self.cancel_btn = QPushButton("Annuler")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        copy_btn = QPushButton("Copier le texte")
        copy_btn.clicked.connect(self._copy)
        open_btn = QPushButton("Ouvrir le dossier")
        open_btn.clicked.connect(self._open_output)
        act.addWidget(self.start_btn)
        act.addWidget(self.cancel_btn)
        act.addStretch(1)
        act.addWidget(copy_btn)
        act.addWidget(open_btn)
        main.addLayout(act)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(8)
        main.addWidget(self.progress)
        self.status = QLabel("Prêt.")
        self.status.setObjectName("hint")
        main.addWidget(self.status)

        # Résultat
        self.result = QPlainTextEdit()
        self.result.setReadOnly(True)
        self.result.setPlaceholderText("La transcription apparaîtra ici…")
        self.result.setMinimumHeight(110)
        main.addWidget(self.result, 1)

        # Hauteurs minimales explicites (évite que les rangées se chevauchent quand la fenêtre est petite)
        for kind in (QLineEdit, QComboBox, QPushButton):
            for w in root.findChildren(kind):
                w.setMinimumHeight(34)
        self._refresh_models(core.DEFAULT_MODEL)

    # ------------------------------------------------- glisser-déposer (fenêtre)
    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.drop.set_active(True)

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        self.drop.set_active(False)

    def dropEvent(self, event) -> None:  # noqa: N802
        self.drop.set_active(False)
        self._add_paths([u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()])
        event.acceptProposedAction()

    # --------------------------------------------------------- gestion fichiers
    def _add_paths(self, raw_paths) -> None:
        added, ignored = 0, []
        for raw in raw_paths:
            p = Path(str(raw).strip().strip('"').strip("'"))
            found = core.collect_audio_files([p]) if p.exists() else []
            found = [f for f in found if f.suffix.lower() in core.AUDIO_EXTS]
            if not found:
                ignored.append(str(p))
            for f in found:
                if f not in self.files:
                    self.files.append(f)
                    self.listw.addItem(str(f))
                    added += 1
        if ignored:
            QMessageBox.warning(self, "Fichier introuvable ou non audio",
                                "Ignoré :\n" + "\n".join(ignored[:10]))
        if added:
            self.status.setText(f"{len(self.files)} fichier(s) prêt(s) à être transcrit(s).")

    def _add_typed_path(self) -> None:
        value = self.path_edit.text().strip()
        if value:
            self._add_paths([value])
            self.path_edit.clear()

    def _browse(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(core.AUDIO_EXTS))
        chosen, _ = QFileDialog.getOpenFileNames(
            self, "Choisir des fichiers audio", "", f"Fichiers audio ({exts});;Tous les fichiers (*.*)")
        if chosen:
            self._add_paths(chosen)

    def _remove_selected(self) -> None:
        for item in self.listw.selectedItems():
            row = self.listw.row(item)
            self.listw.takeItem(row)
            del self.files[row]

    def _clear(self) -> None:
        self.listw.clear()
        self.files.clear()

    def _choose_output(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Dossier de sortie", self.out_edit.text())
        if d:
            self.out_edit.setText(d)

    def _open_output(self) -> None:
        out = Path(self.out_edit.text())
        out.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(out)))

    def _copy(self) -> None:
        QGuiApplication.clipboard().setText(self.result.toPlainText().strip())
        self.status.setText("Texte copié dans le presse-papiers.")

    # ------------------------------------------------------------------ modèles
    @staticmethod
    def _model_label(name: str) -> str:
        state = "✓ installé" if core.model_path(name) else "à télécharger"
        return f"{name}   ·   {core.MODEL_INFO[name][0]}   ·   {state}"

    def _refresh_models(self, select: str | None = None) -> None:
        current = select or self.model_box.currentData() or core.DEFAULT_MODEL
        self.model_box.blockSignals(True)
        self.model_box.clear()
        for name in MODELS:
            self.model_box.addItem(self._model_label(name), name)
        self.model_box.setCurrentIndex(MODELS.index(current) if current in MODELS else 0)
        self.model_box.blockSignals(False)
        self._update_dl_button()

    def _update_dl_button(self) -> None:
        name = self.model_box.currentData()
        installed = bool(name) and core.model_path(name) is not None
        self.dl_btn.setText("✓  Installé" if installed else "⬇  Télécharger")
        self.dl_btn.setEnabled(not installed and not self._busy)

    def _set_busy_ui(self, busy: bool) -> None:
        self._busy = busy
        self.start_btn.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)
        self.model_box.setEnabled(not busy)
        self._update_dl_button()

    # ---------------------------------------------------------------- téléchargement
    def _begin_download(self, name: str, chain: bool) -> None:
        self._dl_chain = chain
        self._dl_samples.clear()
        self._set_busy_ui(True)
        self.progress.setRange(0, 0)
        self.status.setText(f"Téléchargement de « {name} » : connexion à Hugging Face…")
        d = self.dl = DownloadWorker(name)
        d.progress.connect(self._on_dl_progress)
        d.finished_ok.connect(self._on_dl_done)
        d.cancelled.connect(self._on_dl_cancelled)
        d.failed.connect(self._on_dl_failed)
        d.start()

    def _on_dl_progress(self, done: float, total: float) -> None:
        name = self.dl.name if self.dl else ""
        total = total or core.MODEL_INFO.get(name, ("", 0))[1]
        now = time.time()
        self._dl_samples.append((now, done))
        while self._dl_samples and now - self._dl_samples[0][0] > 6:    # débit moyen sur ~6 s
            self._dl_samples.popleft()
        t0, d0 = self._dl_samples[0]
        speed = (done - d0) / (now - t0) if now - t0 > 0.8 else 0
        pct = min(done / total, 1.0) if total else 0
        self.progress.setRange(0, 1000)
        self.progress.setValue(int(pct * 1000))
        txt = f"Téléchargement de « {name} » : {fmt_size(done)} / {fmt_size(total)} ({pct:.0%})"
        if done >= total > 0:
            txt += "  ·  finalisation…"
        elif speed > 0:
            txt += f"  ·  {speed / 1e6:.1f} Mo/s  ·  reste {fmt_eta((total - done) / speed)}"
        self.status.setText(txt)

    def _end_download(self) -> tuple[str, bool]:
        """Termine proprement le thread de téléchargement ; retourne (nom, enchaîner_transcription)."""
        name = self.dl.name if self.dl else ""
        if self.dl:
            self.dl.wait(3000)
        chain, self._dl_chain = self._dl_chain, False
        self._refresh_models(name)
        return name, chain

    def _on_dl_done(self) -> None:
        name, chain = self._end_download()
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.status.setText(f"✓ Modèle « {name} » téléchargé et prêt.")
        if chain:
            self._run_transcription()
        else:
            self._set_busy_ui(False)

    def _on_dl_cancelled(self) -> None:
        self._end_download()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status.setText("Téléchargement annulé — il reprendra là où il s'est arrêté.")
        self._set_busy_ui(False)

    def _on_dl_failed(self, msg: str) -> None:
        self._end_download()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status.setText("Échec du téléchargement.")
        self._set_busy_ui(False)
        QMessageBox.critical(self, "Téléchargement impossible", msg)

    # ------------------------------------------------------------ transcription
    def _start(self) -> None:
        self._add_typed_path()          # un chemin tapé mais pas encore ajouté est pris en compte
        if not self.files:
            QMessageBox.information(self, "Aucun fichier", "Ajoutez un fichier audio "
                                    "(glisser-déposer, chemin ou clic sur la zone de dépôt).")
            return
        name = self.model_box.currentData()
        if core.model_path(name) is None:
            ans = QMessageBox.question(
                self, "Modèle non installé",
                f"Le modèle « {name} » n'est pas encore installé ({core.MODEL_INFO[name][0]}).\n\n"
                "Le télécharger maintenant, puis lancer la transcription ?")
            if ans == QMessageBox.Yes:
                self._begin_download(name, True)
            return
        self._run_transcription()

    def _run_transcription(self) -> None:
        self.result.clear()
        self._set_busy_ui(True)
        settings = dict(
            files=list(self.files), model=self.model_box.currentData(),
            language=LANGUAGES[self.lang_box.currentText()], timestamps=self.ts_check.isChecked(),
            srt=self.srt_check.isChecked(), out_dir=Path(self.out_edit.text()),
        )
        w = self.worker = Worker(settings, self.cache)
        w.busy.connect(self._on_busy)
        w.status.connect(self._on_status)
        w.progress.connect(self._on_progress)
        w.text.connect(self._on_text)
        w.finished_ok.connect(lambda m: self._on_end(m, False))
        w.failed.connect(lambda m: self._on_end(m, True))
        w.start()

    def _cancel(self) -> None:
        if self.dl and self.dl.isRunning():
            self.dl.cancel_event.set()
            self.status.setText("Annulation du téléchargement…")
        elif self.worker:
            self.worker.cancel_event.set()
            self.status.setText("Annulation en cours…")

    def _on_busy(self, msg: str) -> None:
        self.status.setText(msg)
        self.progress.setRange(0, 0)        # barre « en attente »

    def _on_status(self, msg: str) -> None:
        self.status.setText(msg)
        self.progress.setRange(0, 100)

    def _on_progress(self, value: float) -> None:
        self.progress.setRange(0, 100)
        self.progress.setValue(int(value))

    def _on_text(self, chunk: str) -> None:
        self.result.moveCursor(QTextCursor.MoveOperation.End)
        self.result.insertPlainText(chunk)
        self.result.ensureCursorVisible()

    def _on_end(self, msg: str, error: bool) -> None:
        self.progress.setRange(0, 100)
        if not error and not (self.worker and self.worker.cancel_event.is_set()):
            self.progress.setValue(100)
        self.status.setText(msg)
        if self.worker:
            self.worker.wait(3000)
        self._set_busy_ui(False)
        if error:
            QMessageBox.critical(self, "Erreur", msg)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.dl and self.dl.isRunning():
            self.dl.cancel_event.set()
            self.dl.wait(5000)
        if self.worker and self.worker.isRunning():
            self.worker.cancel_event.set()
            self.worker.wait(3000)
        super().closeEvent(event)


def main() -> None:
    if len(sys.argv) > 1:      # mode ligne de commande (utile aussi avec le .exe)
        core.main()
        return
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
