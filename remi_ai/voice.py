"""Offline speech recognition for Ask Remi.

The app window (Edge WebView2) has no working speech-to-text of its own, so Remi listens
through the computer's microphone here in Python with Vosk, an offline recogniser. The small
Indian English model (about 36 MB) is downloaded once into the data folder; after that no
audio ever leaves the computer.

The window drives it with start / poll / stop:
    once       listen for one command, stop at the first pause
    handsfree  keep listening; the window acts on phrases that start with "Remi"
"""
import json
import os
import queue
import threading
import time
import urllib.request
import zipfile

MODEL_NAME = "vosk-model-small-en-in-0.4"
MODEL_URL = f"https://alphacephei.com/vosk/models/{MODEL_NAME}.zip"
RATE = 16000
ONCE_TIMEOUT = 10.0   # seconds of no speech before a single command gives up


class Voice:
    def __init__(self, data_dir):
        self.dir = os.path.join(data_dir, "voice")
        self._lock = threading.Lock()
        self._model = None
        self._download = None      # 0..100 while downloading
        self._error = ""
        self._stream = None
        self._worker = None
        self._audio = queue.Queue()
        self._finals = []
        self._partial = ""
        self._level = 0.0
        self._mode = None
        self._stop = threading.Event()

    # ---- setup ---------------------------------------------------------------------
    @staticmethod
    def _deps():
        try:
            import sounddevice  # noqa: F401
            import vosk  # noqa: F401
            return ""
        except Exception as e:  # missing package, or no audio library on this computer
            return f"{type(e).__name__}: {e}"

    def model_path(self):
        custom = os.environ.get("REMI_VOICE_MODEL")
        if custom and os.path.isdir(custom):
            return custom
        return os.path.join(self.dir, MODEL_NAME)

    def status(self):
        missing = self._deps()
        ready = os.path.isfile(os.path.join(self.model_path(), "conf", "model.conf")) or \
            os.path.isdir(os.path.join(self.model_path(), "am"))
        return {"available": not missing, "problem": missing or self._error, "model_ready": ready,
                "downloading": self._download, "listening": self._mode is not None, "mode": self._mode,
                "engine": "Offline speech recognition (Vosk, Indian English)"}

    def prepare(self):
        """Download the speech model in the background if it isn't there yet."""
        st = self.status()
        if not st["available"] or st["model_ready"] or self._download is not None:
            return st
        self._error = ""
        self._download = 0
        threading.Thread(target=self._fetch_model, daemon=True).start()
        return self.status()

    def _fetch_model(self):
        os.makedirs(self.dir, exist_ok=True)
        tmp = os.path.join(self.dir, MODEL_NAME + ".zip.part")
        try:
            with urllib.request.urlopen(MODEL_URL, timeout=30) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        self._download = min(99, int(done * 100 / total))
            with zipfile.ZipFile(tmp) as z:
                z.extractall(self.dir)
        except Exception as e:
            self._error = f"Could not download the speech model ({type(e).__name__}). Check the internet connection."
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
            self._download = None

    def _load_model(self):
        if self._model is None:
            import vosk
            vosk.SetLogLevel(-1)
            self._model = vosk.Model(self.model_path())
        return self._model

    # ---- listening -----------------------------------------------------------------
    def start(self, mode="once"):
        with self._lock:
            if self._mode is not None:
                self._mode = mode
                return self.status()
            st = self.status()
            if not st["available"]:
                raise ValueError("Voice isn't installed on this computer yet. Restart Remi with run_remi.bat to add it.")
            if not st["model_ready"]:
                self.prepare()
                raise ValueError("The speech model is still downloading. Try again in a moment.")
            import sounddevice as sd
            import vosk
            rec = vosk.KaldiRecognizer(self._load_model(), RATE)
            self._audio = queue.Queue()
            self._finals, self._partial, self._level = [], "", 0.0
            self._stop.clear()
            try:
                self._stream = sd.RawInputStream(samplerate=RATE, blocksize=4000, dtype="int16", channels=1,
                                                 callback=lambda data, frames, t, status: self._audio.put(bytes(data)))
                self._stream.start()
            except Exception as e:
                self._stream = None
                raise ValueError(f"Remi couldn't open the microphone ({type(e).__name__}). "
                                 "Check a microphone is connected and allowed in Windows privacy settings.")
            self._mode = mode
            self._worker = threading.Thread(target=self._listen, args=(rec,), daemon=True)
            self._worker.start()
            return self.status()

    def _listen(self, rec):
        import numpy as np
        started = time.time()
        heard = False
        while not self._stop.is_set():
            try:
                data = self._audio.get(timeout=0.3)
            except queue.Empty:
                continue
            a = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            self._level = float(min(1.0, np.sqrt(np.mean(a * a)) / 6000)) if a.size else 0.0
            if rec.AcceptWaveform(data):
                text = json.loads(rec.Result()).get("text", "").strip()
                self._partial = ""
                if text:
                    heard = True
                    self._finals.append(text)
                    if self._mode == "once":
                        break
            else:
                self._partial = json.loads(rec.PartialResult()).get("partial", "")
                if self._partial:
                    heard = True
            if self._mode == "once" and not heard and time.time() - started > ONCE_TIMEOUT:
                break
        if self._mode == "once" and not self._stop.is_set():
            text = json.loads(rec.FinalResult()).get("text", "").strip()
            if text and not self._finals:
                self._finals.append(text)
        self._close()

    def poll(self):
        finals, self._finals = self._finals, []
        return {"listening": self._mode is not None, "mode": self._mode, "partial": self._partial,
                "finals": finals, "level": round(self._level, 3), "downloading": self._download,
                "problem": self._error}

    def stop(self):
        self._stop.set()
        if self._worker is None or not self._worker.is_alive():
            self._close()
        return {"ok": True}

    def _close(self):
        s, self._stream = self._stream, None
        if s is not None:
            try:
                s.stop()
                s.close()
            except Exception:
                pass
        self._mode = None
        self._level = 0.0
