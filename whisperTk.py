# - Mic capture → VAD → realtime Faster-Whisper → low-confidence phrases re-transcribed at higher beam.
# - Tk UI shows volume, stats, sliders, realtime text, and final transcript (bold if changed after finalize).

import os, time, queue, threading, collections
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import numpy as np, pyaudio, webrtcvad, tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from faster_whisper import WhisperModel
import noisereduce as nr
MODEL_SIZE, DEVICE, COMPUTE_TYPE = "Systran/faster-distil-whisper-small.en", "cuda", "bfloat16"
CUSTOM_VOCAB_PROMPT = "Sujit Vasanth. Faster-Whisper. GPU Acceleration. CUDA. Quantization."
SAMPLE_RATE, CHUNK_MS, VAD_AGGR = 16000, 30, 3
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)
REALTIME_CTX_SEC = 5
DEFAULTS = dict(silence_threshold_s=0.5, word_confidence_threshold=0.40,
                speech_confirmation_ms=120, pre_speech_buffer_ms=500)

# --- Small utilities ---
def safe_denoise(y):
    try: return nr.reduce_noise(y=y, sr=SAMPLE_RATE, stationary=True, prop_decrease=0.8)
    except Exception: return y

# --- Event bus ---
class Bus:
    def __init__(self): self.q, self.flag = queue.Queue(), threading.Event()
    def pub(self, kind): self.q.put(kind); self.flag.set()
    def drain(self, n=100):
        kinds=set()
        for _ in range(n):
            try: kinds.add(self.q.get_nowait())
            except queue.Empty: break
        if self.q.empty(): self.flag.clear()
        return kinds
    def pending(self): return self.flag.is_set()

# --- Settings holder ---
class Settings:
    def __init__(self): self._l=threading.Lock(); self.__dict__.update(DEFAULTS)
    def get(self,k):  # thread-safe
        with self._l: return getattr(self,k)
    def set(self,k,v):
        with self._l: setattr(self,k,v)

# --- UI (Tk) ---
class UI:
    def __init__(self, settings: Settings, bus: Bus):
        self.settings, self.bus, self._alive = settings, bus, True
        r = self.root = tk.Tk()
        r.title("Whisper Realtime (GPU)"); r.geometry("700x660")

        self.vars = { 'confidence': tk.StringVar(value="Min Word Prob: N/A"),
            'unchanged':  tk.StringVar(value="Unchanged: 0"),
            'changed':    tk.StringVar(value="Changed: 0"),
            'volume':     tk.DoubleVar(value=0),}
        ttk.Label(r, text="Input Volume").pack(pady=(10,0))
        s=ttk.Style(); s.configure("green.Horizontal.TProgressbar", background='green')
        ttk.Progressbar(r, variable=self.vars['volume'], maximum=100, style="green.Horizontal.TProgressbar").pack(fill='x', padx=20, pady=5)
        ttk.Label(r, textvariable=self.vars['confidence']).pack(pady=(5,0))
        row=ttk.Frame(r); row.pack(pady=5)
        ttk.Label(row, textvariable=self.vars['unchanged']).pack(side="left", padx=10)
        ttk.Label(row, textvariable=self.vars['changed']).pack(side="left", padx=10)
        self._labels={}
        def add_slider(name, rng, fmt):
            self._labels[name]=ttk.Label(r, text=fmt.format(self.settings.get(name)))
            self._labels[name].pack(pady=(10,0))
            s=ttk.Scale(r, from_=rng[0], to=rng[1], orient="horizontal",
                        command=lambda v,n=name,f=fmt:self._set(n,v,f))
            s.set(self.settings.get(name)); s.pack(fill='x', padx=20)
        add_slider('silence_threshold_s', (0.1, 2.0), "End of Utterance (s): {:.2f}")
        add_slider('word_confidence_threshold', (0.0, 1.0), "Word Conf. Threshold: {:.2f}")
        add_slider('speech_confirmation_ms', (30, 500), "Start of Utterance (ms): {}")
        add_slider('pre_speech_buffer_ms', (100, 1000), "Pre-Roll Buffer (ms): {}")

        lf=ttk.LabelFrame(r, text="Transcript"); lf.pack(fill='both', expand=True, padx=12, pady=12)
        ttk.Label(lf, text="Realtime (in-progress):").pack(anchor="w", padx=10, pady=(8,2))
        self.realtime = ScrolledText(lf, height=4, wrap="word", state="disabled"); self.realtime.pack(fill="x", padx=10, pady=(0,8))
        ttk.Label(lf, text="Final transcript:").pack(anchor="w", padx=10, pady=(6,2))
        self.final = ScrolledText(lf, height=12, wrap="word", state="disabled"); self.final.pack(fill="both", expand=True, padx=10, pady=(0,10))
        self.final.tag_config("changed_final", font=("TkDefaultFont", 10, "bold"))

    # set slider
    def _set(self, key, v, fmt):
        val = round(float(v),2) if '.' in fmt else int(float(v))
        self.settings.set(key, val); self._labels[key].config(text=fmt.format(val))

    # write realtime
    def show_realtime(self, text:str):
        if not self._alive: return
        t=self.realtime; t.config(state="normal"); t.delete("1.0","end")
        if text: t.insert("end", text); t.config(state="disabled")

    # write final list with bold on changed finals
    def show_final(self, sentences):
        if not self._alive: return
        near_bottom=False
        try:
            last=float(self.final.index("@0,999999").split(".")[0])
            tot=float(self.final.index("end-1c").split(".")[0])
            near_bottom=(tot-last)<3.0
        except Exception: pass
        t=self.final; t.config(state="normal"); t.delete("1.0","end")
        for s in sentences:
            f=(s.get('final_text') or s.get('original_text') or "").strip()
            if not f: continue
            if not f.endswith(('.','!','?')): f+="."
            if s.get('status')=='finalized' and (s.get('final_text') or "")!=(s.get('original_text') or ""):
                start=t.index("end-1c"); t.insert("end", f+" "); end=t.index("end-1c"); t.tag_add("changed_final", start, end)
            else:
                t.insert("end", f+" ")
        t.config(state="disabled")
        if near_bottom: t.see("end")

    def on_close(self, cb): self.root.protocol("WM_DELETE_WINDOW", lambda: (setattr(self,'_alive',False), cb(), self.root.destroy()))
    def run(self): self.root.mainloop()

# --- Transcriber ---
class Transcriber:
    def __init__(self, settings: Settings, ui_vars: dict, bus: Bus):
        self.settings, self.ui, self.bus = settings, ui_vars, bus
        self.running, self.lock = True, threading.Lock()
        print(f"Loading model '{MODEL_SIZE}' on {DEVICE} with {COMPUTE_TYPE}...")
        self.model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        self.vad = webrtcvad.Vad(VAD_AGGR)
        self.full, self.is_speaking, self.speech_start_time = [], False, None
        self.cur_text, self.cur_conf = "", 1.0
        self.unchanged, self.changed = 0, 0
        self.new_audio, self.final_q = threading.Event(), queue.Queue()

        self.pre = collections.deque(maxlen=int(self.settings.get('pre_speech_buffer_ms')/CHUNK_MS))
        self.ctx = collections.deque(maxlen=int(REALTIME_CTX_SEC*1000/CHUNK_MS))
        self.buff = []

        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                                   input=True, frames_per_buffer=CHUNK_SAMPLES)

    # thread: mic
    def t_capture(self):
        cur_pre_len = self.pre.maxlen
        while self.running:
            try:
                # pre-roll resize
                new_len = int(self.settings.get('pre_speech_buffer_ms')/CHUNK_MS)
                if new_len!=cur_pre_len:
                    with self.lock: self.pre = collections.deque(list(self.pre), maxlen=new_len)
                    cur_pre_len=new_len

                try: chunk = self.stream.read(CHUNK_SAMPLES, exception_on_overflow=False)
                except Exception: time.sleep(0.01); continue

                a = np.frombuffer(chunk, dtype=np.int16)
                if a.size==0: continue
                rms = float(np.sqrt(np.mean(a.astype(np.float32)**2))); self.ui['volume'].set(min(100, int(rms/50)))
                with self.lock:
                    self.ctx.append(chunk)
                    (self.buff if self.is_speaking else self.pre).append(chunk)
                self.new_audio.set()
            except Exception: time.sleep(0.005)

    # thread: realtime ASR
    def t_realtime(self):
        while self.running:
            self.new_audio.wait(); self.new_audio.clear()
            with self.lock:
                if not self.is_speaking or not self.buff: continue
                data = b"".join(self.buff)
            try:
                y = np.frombuffer(data, dtype=np.int16).astype(np.float32)/32768.0
                if y.size==0: continue
                y = safe_denoise(y)
                segs,_ = self.model.transcribe(y, beam_size=1, initial_prompt=CUSTOM_VOCAB_PROMPT, word_timestamps=True)
                segs=list(segs)
                text = "".join(s.text for s in segs).strip()
                probs=[w.probability for s in segs if s.words for w in s.words]
                conf = min(probs) if probs else 0.0
                with self.lock: self.cur_text, self.cur_conf = text, conf
                self.bus.pub("realtime")
            except Exception: time.sleep(0.005)

    # thread: finalize ASR
    def t_finalize(self):
        while self.running:
            try: audio, idx, orig = self.final_q.get(timeout=1)
            except queue.Empty: continue
            try:
                y = np.frombuffer(b"".join(audio), dtype=np.int16).astype(np.float32)/32768.0
                if y.size==0: continue
                y = safe_denoise(y)
                segs,_ = self.model.transcribe(y, beam_size=5, initial_prompt=CUSTOM_VOCAB_PROMPT)
                final = self._cap("".join(s.text for s in segs))
                if final and 0<=idx<len(self.full):
                    with self.lock: self.full[idx].update(original_text=orig, final_text=final, status='finalized')
            except Exception: pass
            self.bus.pub("final")

    # thread: vad/phrase control
    def t_loop(self):
        last=time.time()
        while self.running:
            if not self.ctx: time.sleep(0.01); continue
            try: speech = self.vad.is_speech(self.ctx[-1], SAMPLE_RATE)
            except Exception: time.sleep(0.005); continue
            if speech:
                last=time.time(); self._on_speech_start()
            else:
                self._on_speech_end(last)
            time.sleep(0.01)

    # helpers
    def _on_speech_start(self):
        with self.lock:
            if self.is_speaking: return
            ms = self.settings.get('speech_confirmation_ms')
            if self.speech_start_time is None: self.speech_start_time = time.time(); return
            if time.time()-self.speech_start_time >= (ms/1000.0):
                self.is_speaking, self.cur_text, self.cur_conf, self.buff = True, "", 1.0, list(self.pre)
                self.speech_start_time=None; self.bus.pub("realtime")

    def _on_speech_end(self, last):
        with self.lock:
            if self.speech_start_time: self.speech_start_time=None
            if self.is_speaking and time.time()-last > self.settings.get('silence_threshold_s'):
                self._end_phrase()

    def _end_phrase(self):
        tmp = self._cap(self.cur_text)
        if tmp:
            self.ui['confidence'].set(f"Min Word Prob: {self.cur_conf:.2f}")
            obj={'original_text':tmp,'final_text':tmp,'status':'awaiting_finalization'}
            self.full.append(obj)
            if self.cur_conf < self.settings.get('word_confidence_threshold'):
                self.final_q.put((list(self.buff), len(self.full)-1, tmp))
                self.changed+=1; self.ui['changed'].set(f"Changed: {self.changed}")
            else:
                obj['status']='high_confidence'
                self.unchanged+=1; self.ui['unchanged'].set(f"Unchanged: {self.unchanged}")
            self.bus.pub("final")
        self.is_speaking, self.buff, self.cur_text = False, [], ""
        self.bus.pub("realtime")

    def snapshot(self):
        with self.lock: return [dict(s) for s in self.full]
    def current(self):
        with self.lock: return self._cap(self.cur_text)
    @staticmethod
    def _cap(t:str): t=(t or "").strip(); return (t[:1].upper()+t[1:]) if t else ""

    def stop(self):
        self.running=False; self.new_audio.set(); time.sleep(0.1)
        try:
            if self.stream.is_active(): self.stream.stop_stream()
        except Exception: pass
        for fn in (self.stream.close, self.pa.terminate):
            try: fn()
            except Exception: pass
        print("Audio resources released.")

# --- Main ---
settings, bus = Settings(), Bus()
ui = UI(settings, bus)
tr = Transcriber(settings, ui.vars, bus)
ui.on_close(tr.stop)

# workers
for target in (tr.t_capture, tr.t_realtime, tr.t_finalize, tr.t_loop):
    threading.Thread(target=target, daemon=True).start()

# event-driven UI pump (coalesced)
def pump():
    if not ui._alive or not tr.running: return
    kinds = bus.drain()
    if "final" in kinds: ui.show_final(tr.snapshot())
    if "realtime" in kinds: ui.show_realtime(tr.current())
    ui.root.after(0 if (kinds or bus.pending()) else 150, pump)
ui.root.after(0, pump)
ui.run()
