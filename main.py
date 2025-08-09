"""
Berlin × Tri-Harmonic — Sequencer + Spheres + Cymatics
------------------------------------------------------
- 16-step Berlin-style sequencer (bass/lead + kick/snare/hats)
- Full Tri-Harmonic routing/visuals (S1/S2/S3 spheres, torus, pulse rings)
- Waveforms: sine, triangle, saw, square, pulse (PWM), detuned saw
- Cymatics panel: animated standing-wave patterns driven by current freqs
- Smoothed fractional delay (wet/dry/feedback/time) to avoid zipper/jitter

Requires: PyQt5, vispy, sounddevice (auto-installs)
"""

import sys, time, logging, threading
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Dict, Any, List

import numpy as np

# -----------------------------------------------------------------------------#
# Dependency bootstrap
# -----------------------------------------------------------------------------#
def _ensure_pkg(mod_name: str, pip_name: Optional[str] = None):
    try:
        __import__(mod_name)
    except ImportError:
        print(f"[setup] {mod_name} not found. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name or mod_name])

_ensure_pkg("PyQt5", "PyQt5")
_ensure_pkg("vispy", "vispy")
_ensure_pkg("sounddevice", "sounddevice")

from PyQt5 import QtWidgets, QtCore
from vispy import app, scene
import sounddevice as sd

app.use_app("pyqt5")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -----------------------------------------------------------------------------#
# Data structures shared with the tri-harmonic core
# -----------------------------------------------------------------------------#
class SphereFunction(Enum):
    TEMPORAL = "temporal_memory"
    SEMANTIC = "semantic_drift"
    HARMONIC = "harmonic_feedback"

@dataclass
class DataPoint:
    value: float          # frequency (Hz) or value
    timestamp: float
    symbol: str
    phase: float          # routing phase (0..2π)
    spectrum: Optional[np.ndarray]
    energy: float         # 0..1
    duration: float
    metadata: Dict[str, Any] = None
    def clone(self):
        return DataPoint(
            value=self.value, timestamp=self.timestamp, symbol=self.symbol,
            phase=self.phase, spectrum=self.spectrum.copy() if self.spectrum is not None else None,
            energy=self.energy, duration=self.duration,
            metadata=self.metadata.copy() if self.metadata else {}
        )

# -----------------------------------------------------------------------------#
# Tri-Harmonic core (routing + visuals plumbing)
# -----------------------------------------------------------------------------#
class SymbolicSphere:
    def __init__(self, radius, plane, function, capacity=100, decay_lambda=0.95, rotation_speed=0.5):
        self.radius = radius; self.plane = plane; self.function = function
        self.capacity = capacity; self.decay_lambda = decay_lambda; self.rotation_speed = rotation_speed
        self.memory_ring = deque(maxlen=capacity)
        self.phase_offset = np.random.random()*2*np.pi
        self.symbol_set = set()
        self.output_queue = deque()
        self.total_residence_time = 0.0; self.exit_count = 0

    def _pos(self, phase):
        a = phase + self.phase_offset
        if self.plane == "XY": return (self.radius*np.cos(a), self.radius*np.sin(a), 0.0)
        if self.plane == "YZ": return (0.0, self.radius*np.cos(a), self.radius*np.sin(a))
        if self.plane == "XZ": return (self.radius*np.cos(a), 0.0, self.radius*np.sin(a))
        return (0.0, 0.0, 0.0)

    def update_rotation(self):
        self.phase_offset = (self.phase_offset + self.rotation_speed) % (2*np.pi)
        for dp in self.memory_ring: dp.energy *= self.decay_lambda

    def can_accept(self, dp: DataPoint):
        # lightweight accept score
        s = 0.2 + 0.6*np.clip(dp.energy,0,1)
        if dp.symbol in self.symbol_set: s += 0.2
        elif len(self.symbol_set) < 20: s += 0.1
        return (s > 0.25, s)

    def inject(self, dp: DataPoint):
        d = dp.clone(); d.metadata = d.metadata or {}
        d.metadata.update({"sphere_position": self._pos(d.phase), "entry_time": time.time(),
                           "sphere_plane": self.plane, "sphere_function": self.function.value})
        self.memory_ring.append(d); self.symbol_set.add(d.symbol)

    def extract_ready(self):
        ready, rem = [], deque(); now = time.time()
        for d in self.memory_ring:
            dt = now - d.metadata.get("entry_time", now)
            exit_cond = (dt > 1.2) or (d.energy < 0.05) or (len(self.memory_ring) > self.capacity-2)
            if exit_cond:
                d.metadata["exit_reason"] = "timeout"; d.metadata["residence_time"] = dt
                ready.append(d); self.total_residence_time += dt; self.exit_count += 1
            else: rem.append(d)
        self.memory_ring = rem
        return ready

    def apply_transformations(self):
        for d in self.memory_ring:
            d.phase = (d.phase + self.rotation_speed) % (2*np.pi)
            d.metadata["sphere_position"] = self._pos(d.phase)

class ResonanceGate:
    def __init__(self, phase_tolerance=0.6, energy_threshold=0.08):
        self.phase_tolerance = phase_tolerance; self.energy_threshold = energy_threshold
    def check_resonance(self, dp: DataPoint, sphere: SymbolicSphere):
        if dp.energy < self.energy_threshold: return (False, 0.0)
        pd = abs(dp.phase - sphere.phase_offset) % (2*np.pi)
        align = 1.0 - min(pd, 2*np.pi-pd)/max(1e-6, self.phase_tolerance)
        s = 0.3*np.clip(align, 0, 1)
        ok, sc = sphere.can_accept(dp); s += (0.5 if ok else 0.2)*sc
        return (s > 0.25, s)

class TriHarmonicCore:
    def __init__(self, major_radius=10.0, minor_radius=3.0, enable_s2=True, enable_s3=True):
        self.major_radius = major_radius; self.minor_radius = minor_radius
        r = minor_radius * 0.8
        self.sphere_s1 = SymbolicSphere(r, "XY", SphereFunction.TEMPORAL, rotation_speed=0.45)
        self.sphere_s2 = SymbolicSphere(r, "YZ", SphereFunction.SEMANTIC, rotation_speed=0.33, decay_lambda=0.98) if enable_s2 else None
        self.sphere_s3 = SymbolicSphere(r, "XZ", SphereFunction.HARMONIC, rotation_speed=0.7,  decay_lambda=0.92) if enable_s3 else None
        self.spheres = [self.sphere_s1] + ([self.sphere_s2] if self.sphere_s2 else []) + ([self.sphere_s3] if self.sphere_s3 else [])
        self.gate = ResonanceGate()
        self.torus_buffer = deque(maxlen=1000)
        self.metrics = {"total_processed":0,"s1_entries":0,"s1_exits":0,"s2_entries":0,"s2_exits":0,"s3_entries":0,"s3_exits":0,
                        "torus_direct":0,"total_resonance_checks":0,"avg_resonance_score":0.0}
        self.resonance_scores: List[float] = []

    def _torus_pos(self, phase: float, energy: float):
        R, rr = self.major_radius, self.minor_radius * 0.9
        u = phase; v = energy * 2 * np.pi
        x = (R + rr*np.cos(v))*np.cos(u); y = (R + rr*np.cos(v))*np.sin(u); z = rr*np.sin(v)
        return [float(x), float(y), float(z)]

    def process(self, dp: DataPoint):
        self.metrics["total_processed"] += 1
        best, best_sc = None, 0.0
        for s in self.spheres:
            ok, sc = self.gate.check_resonance(dp, s)
            self.metrics["total_resonance_checks"] += 1; self.resonance_scores.append(sc)
            if ok and sc > best_sc: best, best_sc = s, sc
        if best:
            best.inject(dp); key = {self.sphere_s1:"s1_entries", self.sphere_s2:"s2_entries", self.sphere_s3:"s3_entries"}[best]
            self.metrics[key] += 1
        else:
            self.torus_buffer.append(dp); self.metrics["torus_direct"] += 1

        for s in self.spheres:
            s.update_rotation(); s.apply_transformations()
            for ed in s.extract_ready():
                ed.metadata["sphere_processed"] = True
                self.torus_buffer.append(ed)
                k = {self.sphere_s1:"s1_exits", self.sphere_s2:"s2_exits", self.sphere_s3:"s3_exits"}[s]
                self.metrics[k] += 1

        if self.resonance_scores:
            self.metrics["avg_resance"] = float(np.mean(self.resonance_scores[-100:]))

# -----------------------------------------------------------------------------#
# Berlin audio engine (with waveforms + smoothed delay)  — audio-only
# -----------------------------------------------------------------------------#
class ParamSmoother:
    def __init__(self, sr, ramp_ms=25.0):
        self.sr = sr; self.cur = 0.0; self.target = 0.0; self.set_time(ramp_ms)
    def set_time(self, ramp_ms):
        tau = max(1e-3, ramp_ms/1000.0)
        self.alpha = 1.0 - np.exp(-1.0/(self.sr*tau))
    def set_target(self, v): self.target = float(v)
    def next_block(self, n):
        out = np.empty(n, dtype=np.float32); c = self.cur; a = self.alpha; tgt = self.target
        for i in range(n): c += (tgt - c) * a; out[i] = c
        self.cur = c; return out

def tri_from_sine(phase):
    # polynomial triangle from saw approximation (cheap & decent)
    saw = (2.0*phase - 1.0)
    return (2.0/np.pi)*np.arcsin(np.clip(np.sin(2*np.pi*phase), -1, 1))

def osc_block(shape, freq, n, sr, phase0=0.0, pwm=0.5, detune_cents=7.0):
    t = (phase0 + np.arange(n)/sr*freq) % 1.0
    if shape == "sine":
        out = np.sin(2*np.pi*t, dtype=np.float32)
    elif shape == "triangle":
        out = tri_from_sine(t).astype(np.float32)
    elif shape == "square":
        out = np.sign(np.sin(2*np.pi*t)).astype(np.float32)
    elif shape == "pulse":
        duty = float(np.clip(pwm, 0.05, 0.95))
        out = ((t % 1.0) < duty).astype(np.float32)*2 - 1
    elif shape == "supersaw":
        # two detuned saws
        det = 2**(detune_cents/1200.0)
        t2 = (phase0 + np.arange(n)/sr*(freq*det)) % 1.0
        saw1 = (2.0*t - 1.0); saw2 = (2.0*t2 - 1.0)
        out = 0.5*(saw1 + saw2).astype(np.float32)
    else:  # "saw"
        out = (2.0*t - 1.0).astype(np.float32)
    phase1 = (phase0 + n*freq/sr) % 1.0
    return out, phase1

class OnePoleLP:
    def __init__(self, sr, cutoff=800.0):
        self.sr = sr; self.z = 0.0; self.set_cutoff(cutoff)
    def set_cutoff(self, hz):
        hz = max(20.0, min(self.sr*0.45, hz))
        x = np.exp(-2*np.pi*hz/self.sr); self.a = 1.0 - x; self.b = x
    def process(self, x):
        y = np.empty_like(x); z = self.z; a=self.a; b=self.b
        for i in range(x.size): z = a*x[i] + b*z; y[i] = z
        self.z = z; return y

@dataclass
class Voice:
    shape: str = "saw"
    phase: float = 0.0
    pwm: float = 0.5
    detune_cents: float = 7.0
    filt: OnePoleLP = None

class BerlinAudio:
    def __init__(self, sr=44100):
        self.sr = sr; self._lock = threading.Lock()
        # Global
        self.master = 0.8; self.bpm = 120.0; self.swing = 0.0
        self.steps = 16; self.sample_counter = 0; self.step_samples = self._calc_step_samples()
        self.next_step_at = self.step_samples; self.step_idx = 0
        self.time = 0.0
        # Delay
        self.delay_buf = np.zeros(int(sr*2.5), dtype=np.float32); self.dw = 0
        self.delay_ms = 350.0; self.delay_fb = 0.35; self.delay_mix = 0.25
        self._mix_s = ParamSmoother(sr, 25); self._fb_s = ParamSmoother(sr, 25); self._dt_s = ParamSmoother(sr, 60)
        self._mix_s.set_target(self.delay_mix); self._fb_s.set_target(self.delay_fb); self._dt_s.set_target(self.delay_ms/1000.0)
        # Vibrato
        self.vib_cents = 4.0; self.vib_rate = 5.5
        # Voices
        self.bass = Voice(shape="saw", filt=OnePoleLP(sr, 220.0))
        self.lead = Voice(shape="square", filt=OnePoleLP(sr, 1800.0))
        self.drive = 1.2
        # Drums env
        self.k_env = 0.0; self.k_phase = 0.0; self.s_env = 0.0; self.h_env = 0.0
        # Patterns (minor pentatonic around A)
        self.root = 110.0
        self.scale = np.array([0, 3, 5, 7, 10, 12])
        self.bass_pat = [0,0,3,0, 5,5,3,0, 0,7,5,3, 10,7,5,3]
        self.lead_pat = [12,15,12,17, 12,19,17,15, 12,15,12,17, 19,17,15,12]
        self.k_pat =   [1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0]
        self.s_pat =   [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0]
        self.h_pat =   [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0]
        # Targets per step
        self.b_freq = self.root; self.l_freq = self.root*2
        self.b_gain = 0.25; self.l_gain = 0.22
        # Event callback for visuals
        self.on_step_event = None  # fn(kind:str, freq:float, velocity:float, step_idx:int)

        self.stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32", blocksize=512, callback=self._cb)
        self.stream.start()

    # Setters
    def _calc_step_samples(self):
        spb = 60.0 / max(1.0, self.bpm); return int((spb/4.0) * self.sr)  # 4 steps per beat

    def set_bpm(self, bpm):
        with self._lock:
            self.bpm = float(np.clip(bpm, 20, 300)); self.step_samples = self._calc_step_samples()
            self.sample_counter = 0; self.next_step_at = self.step_samples; self.step_idx = 0

    def set_swing(self, pct):  # 0..0.5
        with self._lock: self.swing = float(np.clip(pct, 0.0, 0.5))

    def set_voice(self, which: str, shape: str=None, pwm: float=None, detune: float=None, cutoff: float=None):
        v = self.bass if which=="bass" else self.lead
        with self._lock:
            if shape is not None: v.shape = shape
            if pwm   is not None: v.pwm = float(np.clip(pwm, 0.05, 0.95))
            if detune is not None: v.detune_cents = float(np.clip(detune, 0.0, 25.0))
            if cutoff is not None: v.filt.set_cutoff(float(np.clip(cutoff, 50, 8000)))

    def set_drive(self, drive_x):  # 0.5 .. 3.0
        with self._lock: self.drive = float(np.clip(drive_x, 0.5, 3.0))

    def set_delay(self, ms, fb, mix):
        with self._lock:
            self.delay_ms = float(np.clip(ms, 1.0, 1500.0))
            self.delay_fb = float(np.clip(fb, 0.0, 0.95))
            self.delay_mix = float(np.clip(mix, 0.0, 1.0))
            self._mix_s.set_target(self.delay_mix); self._fb_s.set_target(self.delay_fb); self._dt_s.set_target(self.delay_ms/1000.0)

    def set_vibrato(self, cents, rate_hz):
        with self._lock:
            self.vib_cents = float(np.clip(cents, 0, 50)); self.vib_rate = float(np.clip(rate_hz, 0.1, 12.0))

    def randomize_patterns(self):
        rng = np.random.default_rng()
        self.bass_pat = [int(rng.choice([0,0,3,5,7,10])) for _ in range(16)]
        self.lead_pat = [int(rng.choice([12,15,17,19,24,27])) for _ in range(16)]
        self.k_pat = [1 if i%4==0 else 0 for i in range(16)]
        self.s_pat = [1 if i in (4,12) else 0 for i in range(16)]
        self.h_pat = [1 if i%2==0 else 0 for i in range(16)]

    # Step advance
    def _advance_step(self):
        i = self.step_idx % self.steps
        semi_b = self.scale[0] + self.bass_pat[i]; self.b_freq = self.root * (2**(semi_b/12.0))
        semi_l = self.lead_pat[i];                  self.l_freq = self.root * (2**(semi_l/12.0))
        self.b_gain = 0.28; self.l_gain = 0.24

        # drums
        if self.k_pat[i]: self.k_env = 1.0; self.k_phase = 0.0; 
        if self.s_pat[i]: self.s_env = 1.0
        if self.h_pat[i]: self.h_env = 1.0

        # visual event callback
        if self.on_step_event:
            if self.k_pat[i]: self.on_step_event("kick", 55.0, 1.0, i)
            self.on_step_event("bass", self.b_freq, 0.8, i)
            self.on_step_event("lead", self.l_freq, 0.6, i)

        base = self._calc_step_samples()
        if (i%2)==1: base = int(base * (1.0 + self.swing))  # swing every 8th off-beat
        self.next_step_at += base; self.step_idx += 1

    # Drums
    def _kick(self, n):
        if self.k_env <= 1e-4: return np.zeros(n, dtype=np.float32)
        t = np.arange(n)/self.sr
        f0, f1, dtime = 90.0, 35.0, 0.05
        f = f1 + (f0-f1)*np.exp(-t/dtime)
        self.k_env *= np.exp(-n/(self.sr*0.22))
        ph = self.k_phase + 2*np.pi*np.cumsum(f)/self.sr
        self.k_phase = ph[-1] % (2*np.pi)
        return (np.sin(ph) * self.k_env * 0.95).astype(np.float32)

    def _snare(self, n):
        if self.s_env <= 1e-4: return np.zeros(n, dtype=np.float32)
        noise = (np.random.rand(n).astype(np.float32)*2 - 1)
        d = noise - np.concatenate(([0.0], noise[:-1])) * 0.85
        self.s_env *= np.exp(-n/(self.sr*0.14))
        return d * self.s_env * 0.5

    def _hihat(self, n):
        if self.h_env <= 1e-4: return np.zeros(n, dtype=np.float32)
        noise = (np.random.rand(n).astype(np.float32)*2 - 1)
        y = noise - np.concatenate(([0.0], noise[:-1])) * 0.5
        self.h_env *= np.exp(-n/(self.sr*0.045))
        return y * self.h_env * 0.22

    # Delay with smoothing + fractional interp
    def _delay(self, x):
        n = x.size; out = np.empty_like(x)
        mix = self._mix_s.next_block(n); fb = self._fb_s.next_block(n); dts = self._dt_s.next_block(n)
        dSamp = np.clip(dts*self.sr, 1.0, self.delay_buf.size-2)
        for i in range(n):
            rd_float = (self.dw - dSamp[i]) % self.delay_buf.size
            i0 = int(rd_float); frac = rd_float - i0; i1 = (i0+1) % self.delay_buf.size
            y = (1.0-frac)*self.delay_buf[i0] + frac*self.delay_buf[i1]
            out[i] = x[i]*(1.0 - mix[i]) + y*mix[i]
            self.delay_buf[self.dw] = x[i] + y*fb[i]
            self.dw = (self.dw + 1) % self.delay_buf.size
        return out

    # Audio callback
    def _cb(self, out, frames, time_info, status):
        with self._lock:
            master = self.master; drive = self.drive
            b_shape, l_shape = self.bass.shape, self.lead.shape
            pwm_b, pwm_l = self.bass.pwm, self.lead.pwm
            det_b, det_l = self.bass.detune_cents, self.lead.detune_cents
            vib_c, vib_r = self.vib_cents, self.vib_rate

        # step timing
        start = self.sample_counter; end = start + frames
        while self.next_step_at <= end: self._advance_step()
        self.sample_counter = end

        # vibrato
        t = np.arange(frames)/self.sr
        if vib_c > 0:
            lfo = np.sin(2*np.pi*vib_r*(self.time + t))
            cents = (vib_c/1200.0)*lfo
            bf = self.b_freq * (2.0 ** cents)
            lf = self.l_freq * (2.0 ** (0.5*cents))
        else:
            bf = np.full(frames, self.b_freq, dtype=np.float32)
            lf = np.full(frames, self.l_freq, dtype=np.float32)

        # voices
        b_raw, self.bass.phase = osc_block(b_shape, self.b_freq, frames, self.sr, self.bass.phase, pwm_b, det_b)
        l_raw, self.lead.phase = osc_block(l_shape, self.l_freq, frames, self.sr, self.lead.phase, pwm_l, det_l)
        b_raw *= self.b_gain; l_raw *= self.l_gain
        # soft drive
        b_raw = np.tanh(drive*b_raw); l_raw = np.tanh(drive*l_raw)
        # filter
        b = self.bass.filt.process(b_raw); l = self.lead.filt.process(l_raw)

        # drums
        k = self._kick(frames); s = self._snare(frames); h = self._hihat(frames)
        mix = b + l + k + s + h

        # delay
        if self.delay_mix > 0.001: mix = self._delay(mix)

        # out
        mix *= master; np.clip(mix, -0.98, 0.98, out=mix)
        out[:,0] = mix; self.time += frames/self.sr

    def stop(self): self.stream.stop(); self.stream.close()

# -----------------------------------------------------------------------------#
# Visualizer with torus + spheres + pulse rings + cymatics
# -----------------------------------------------------------------------------#
class BerlinTriVis:
    def __init__(self):
        # Core + Audio
        self.core = TriHarmonicCore(enable_s2=True, enable_s3=True)
        self.audio = BerlinAudio()
        self.audio.on_step_event = self._on_step  # connect events to core & visuals

        # Qt/VisPy
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.win = QtWidgets.QWidget(); self.win.setWindowTitle("Berlin × Tri-Harmonic (Sequencer + Cymatics)")

        root = QtWidgets.QHBoxLayout(self.win); root.setContentsMargins(8,8,8,8); root.setSpacing(8)
        left = QtWidgets.QVBoxLayout(); right = QtWidgets.QVBoxLayout(); root.addLayout(left, 1); root.addLayout(right, 0)

        # Canvas 3D (torus + spheres)
        self.canvas = scene.SceneCanvas(title="TRI-HARMONIC VIS", size=(1280, 900), bgcolor="black")
        left.addWidget(self.canvas.native, stretch=1)
        grid = self.canvas.central_widget.add_grid()
        self.view3d = grid.add_view(row=0, col=0, camera="turntable")
        self.view3d.camera.distance = 40; self.view3d.camera.elevation = 30; self.view3d.camera.azimuth = 45; self.view3d.camera.fov = 60

        # Cymatics panel (right under controls)
        self.cym_canvas = scene.SceneCanvas(size=(520, 520), bgcolor="#101010")
        right.addWidget(self.cym_canvas.native, stretch=0)
        self.cym_view = self.cym_canvas.central_widget.add_view()
        self.cym_view.camera = 'panzoom'; self.cym_view.camera.zoom = (1.1, 1.1)
        self.cym_img = scene.visuals.Image(np.zeros((256,256), dtype=np.float32), cmap='viridis', parent=self.cym_view.scene)
        self.cym_text = QtWidgets.QLabel("Cymatics (mode mix)"); self.cym_text.setStyleSheet("color:#ccc;")
        right.addWidget(self.cym_text)

        # Geometry
        self._create_torus_wireframe(); self._create_sphere_wireframes()

        # Particles & pulse rings
        self.torus_particles = scene.visuals.Markers(parent=self.view3d.scene)
        self.s1_particles = scene.visuals.Markers(parent=self.view3d.scene)
        self.s2_particles = scene.visuals.Markers(parent=self.view3d.scene)
        self.s3_particles = scene.visuals.Markers(parent=self.view3d.scene)
        for p in (self.torus_particles, self.s1_particles, self.s2_particles, self.s3_particles):
            p.set_gl_state("translucent", depth_test=True)
        self.pulse_rings: List[Dict[str, Any]] = []; self.max_pulse_rings = 6

        # Controls
        ctrl = QtWidgets.QGridLayout(); row=0
        # BPM
        ctrl.addWidget(QtWidgets.QLabel("BPM"), row,0); self.bpm = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.bpm.setRange(20,300); self.bpm.setValue(120)
        self.bpm.valueChanged.connect(lambda v: self.audio.set_bpm(float(v))); self.bpm_lbl = QtWidgets.QLabel("120"); self.bpm.valueChanged.connect(lambda v: self.bpm_lbl.setText(str(v)))
        ctrl.addWidget(self.bpm, row,1); ctrl.addWidget(self.bpm_lbl, row,2); row+=1
        # Swing
        ctrl.addWidget(QtWidgets.QLabel("Swing %"), row,0); self.swing = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.swing.setRange(0,50); self.swing.setValue(0)
        self.swing.valueChanged.connect(lambda v: self.audio.set_swing(v/100.0)); self.swing_lbl = QtWidgets.QLabel("0"); self.swing.valueChanged.connect(lambda v: self.swing_lbl.setText(str(v)))
        ctrl.addWidget(self.swing, row,1); ctrl.addWidget(self.swing_lbl, row,2); row+=1
        # Waveforms
        ctrl.addWidget(QtWidgets.QLabel("Bass Wave"), row,0); self.bwav = QtWidgets.QComboBox(); self.bwav.addItems(["saw","square","sine","triangle","pulse","supersaw"]); self.bwav.setCurrentText("saw")
        ctrl.addWidget(self.bwav, row,1); row+=1
        ctrl.addWidget(QtWidgets.QLabel("Lead Wave"), row,0); self.lwav = QtWidgets.QComboBox(); self.lwav.addItems(["square","saw","sine","triangle","pulse","supersaw"]); self.lwav.setCurrentText("square")
        ctrl.addWidget(self.lwav, row,1); row+=1
        self.bwav.currentTextChanged.connect(lambda _: self.audio.set_voice("bass", shape=self.bwav.currentText()))
        self.lwav.currentTextChanged.connect(lambda _: self.audio.set_voice("lead", shape=self.lwav.currentText()))
        # PWM + Detune + Cutoffs
        ctrl.addWidget(QtWidgets.QLabel("PWM (bass/lead)"), row,0); self.pwmb = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.pwml = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        for s in (self.pwmb, self.pwml): s.setRange(5,95); s.setValue(50); s.setTracking(False)
        self.pwmb.valueChanged.connect(lambda v: self.audio.set_voice("bass", pwm=v/100.0))
        self.pwml.valueChanged.connect(lambda v: self.audio.set_voice("lead", pwm=v/100.0))
        ctrl.addWidget(self.pwmb, row,1); ctrl.addWidget(self.pwml, row,2); row+=1
        ctrl.addWidget(QtWidgets.QLabel("Detune (cents) b/l"), row,0); self.detb = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.detl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        for s in (self.detb, self.detl): s.setRange(0,25); s.setValue(7); s.setTracking(False)
        self.detb.valueChanged.connect(lambda v: self.audio.set_voice("bass", detune=float(v)))
        self.detl.valueChanged.connect(lambda v: self.audio.set_voice("lead", detune=float(v)))
        ctrl.addWidget(self.detb, row,1); ctrl.addWidget(self.detl, row,2); row+=1
        ctrl.addWidget(QtWidgets.QLabel("Cutoff (Hz) b/l"), row,0); self.cutb = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.cutl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.cutb.setRange(50,2000); self.cutl.setRange(200,8000); self.cutb.setValue(220); self.cutl.setValue(1800)
        self.cutb.valueChanged.connect(lambda v: self.audio.set_voice("bass", cutoff=float(v)))
        self.cutl.valueChanged.connect(lambda v: self.audio.set_voice("lead", cutoff=float(v)))
        ctrl.addWidget(self.cutb, row,1); ctrl.addWidget(self.cutl, row,2); row+=1
        ctrl.addWidget(QtWidgets.QLabel("Drive (x0.1)"), row,0); self.drive = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.drive.setRange(5,30); self.drive.setValue(12)
        self.drive.valueChanged.connect(lambda v: self.audio.set_drive(v/10.0)); self.drive_lbl = QtWidgets.QLabel("1.2"); self.drive.valueChanged.connect(lambda v: self.drive_lbl.setText(f"{v/10:.1f}"))
        ctrl.addWidget(self.drive, row,1); ctrl.addWidget(self.drive_lbl, row,2); row+=1
        # Vibrato
        ctrl.addWidget(QtWidgets.QLabel("Vibrato (cents / Hz)"), row,0); self.vdep = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.vrate = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.vdep.setRange(0,50); self.vdep.setValue(4); self.vrate.setRange(1,120); self.vrate.setValue(55)
        self.vdep.valueChanged.connect(lambda _: self.audio.set_vibrato(float(self.vdep.value()), float(self.vrate.value()/10)))
        self.vrate.valueChanged.connect(lambda _: self.audio.set_vibrato(float(self.vdep.value()), float(self.vrate.value()/10)))
        ctrl.addWidget(self.vdep, row,1); ctrl.addWidget(self.vrate, row,2); row+=1
        # Delay
        ctrl.addWidget(QtWidgets.QLabel("Delay (ms / fb% / mix%)"), row,0); self.dms = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.dfb = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.dmx = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.dms.setRange(1,1500); self.dfb.setRange(0,95); self.dmx.setRange(0,100); self.dms.setValue(350); self.dfb.setValue(35); self.dmx.setValue(25)
        for s in (self.dms,self.dfb,self.dmx): s.setTracking(False)
        def upd_delay(_=None): self.audio.set_delay(float(self.dms.value()), float(self.dfb.value()/100), float(self.dmx.value()/100))
        self.dms.valueChanged.connect(upd_delay); self.dfb.valueChanged.connect(upd_delay); self.dmx.valueChanged.connect(upd_delay)
        ctrl.addWidget(self.dms, row,1); ctrl.addWidget(self.dfb, row,2); row+=1
        # Randomize
        rand = QtWidgets.QPushButton("Randomize Patterns"); rand.clicked.connect(self.audio.randomize_patterns)
        ctrl.addWidget(rand, row,0,1,3); row+=1
        right.addLayout(ctrl)

        # Stats
        self.stats = QtWidgets.QPlainTextEdit(); self.stats.setReadOnly(True); self.stats.setMinimumWidth(520); self.stats.setMinimumHeight(260)
        right.addWidget(self.stats, 1)

        # Particles style
        self.origin_marker = scene.visuals.Markers(parent=self.view3d.scene)
        self.origin_marker.set_data(pos=np.array([[0,0,0]]), face_color="white", edge_color="yellow", edge_width=2, size=18, symbol="disc")
        self.x_axis = scene.visuals.XYZAxis(parent=self.view3d.scene, width=2)

        # Timers
        self._vis_frame = 0; self._sim_frame = 0
        self.timer = QtCore.QTimer(); self.timer.setInterval(40); self.timer.timeout.connect(self._tick); self.timer.start()

        print("[engine] Berlin × Tri-Harmonic running")

    # Geometry
    def _create_torus_wireframe(self):
        R = self.core.major_radius; r = self.core.minor_radius
        u = np.linspace(0, 2*np.pi, 30); v = np.linspace(0, 2*np.pi, 20)
        for i in range(len(u)):
            x = (R + r*np.cos(v))*np.cos(u[i]); y = (R + r*np.cos(v))*np.sin(u[i]); z = r*np.sin(v)
            scene.visuals.Line(np.column_stack([x,y,z]), color=(0.5,0.5,0.5,0.2), parent=self.view3d.scene, width=0.5)
        for j in range(0, len(v), 2):
            x = (R + r*np.cos(v[j]))*np.cos(u); y = (R + r*np.cos(v[j]))*np.sin(u); z = r*np.sin(v[j])*np.ones_like(u)
            scene.visuals.Line(np.column_stack([x,y,z]), color=(0.5,0.5,0.5,0.2), parent=self.view3d.scene, width=0.5)

    def _create_sphere_wireframes(self):
        th = np.linspace(0, 2*np.pi, 20); r = self.core.sphere_s1.radius
        scene.visuals.Line(np.column_stack([r*np.cos(th), r*np.sin(th), 0*th]), color="orange", parent=self.view3d.scene, width=2)
        if self.core.sphere_s2: scene.visuals.Line(np.column_stack([0*th, r*np.cos(th), r*np.sin(th)]), color="cyan", parent=self.view3d.scene, width=2)
        if self.core.sphere_s3: scene.visuals.Line(np.column_stack([r*np.cos(th), 0*th, r*np.sin(th)]), color="lime", parent=self.view3d.scene, width=2)

    # Event from audio step
    def _on_step(self, kind: str, freq: float, vel: float, step_idx:int):
        # map step to phase around torus
        phase = (step_idx / 16.0) * 2*np.pi
        energy = float(np.clip(vel, 0.05, 1.0))
        sym = f"{kind.upper()}-{int(freq)}Hz"
        dp = DataPoint(value=freq, timestamp=time.time(), symbol=sym, phase=phase,
                       spectrum=None, energy=energy, duration=0.25, metadata={})
        self.core.process(dp)
        # make a pulse ring
        origin = self.core._torus_pos(phase, energy)
        self._make_pulse(freq, energy, origin)

    # Pulse visuals
    def _make_pulse(self, freq, energy, origin):
        try:
            if len(self.pulse_rings) >= self.max_pulse_rings:
                old = self.pulse_rings.pop(0)
                if old.get("visual") and old["visual"].parent: old["visual"].parent = None
            theta = np.linspace(0, 2*np.pi, 40); r0 = 2.0
            f_norm = min(1.0, max(0.0, (freq - 200.0)/800.0))
            col = (1.0, 1.0-f_norm, f_norm, 1.0)
            rx = origin[0] + r0*np.cos(theta); ry = origin[1] + r0*np.sin(theta); rz = origin[2] + 0*theta
            vis = scene.visuals.Line(np.column_stack([rx, ry, rz]), color=col, parent=self.view3d.scene, width=5)
            self.pulse_rings.append({"visual":vis,"start_time":time.time(),"color":col,"origin":origin,"radius":r0})
        except Exception as e:
            print(f"[pulse create] {e}")

    def _update_pulse_rings(self):
        if not self.pulse_rings: return
        now = time.time(); alive=[]
        for ring in self.pulse_rings:
            try:
                age = now - ring["start_time"]; max_age = 3.0
                if age < max_age:
                    prog = age/max_age; radius = ring["radius"] + prog*20.0; alpha = (1.0-prog)*0.9
                    th = np.linspace(0, 2*np.pi, 40); o = ring["origin"]
                    rx = o[0] + radius*np.cos(th); ry = o[1] + radius*np.sin(th); rz = o[2] + 0*th
                    ring["visual"].set_data(np.column_stack([rx,ry,rz]), color=(*ring["color"][:3], alpha))
                    alive.append(ring)
                else:
                    if ring.get("visual") and ring["visual"].parent: ring["visual"].parent = None
            except Exception:
                if ring.get("visual") and ring["visual"].parent: ring["visual"].parent = None
        self.pulse_rings = alive

    # Cymatics (fake-but-fun mode mix)
    def _update_cymatics(self):
        # Use current bass/lead freqs (pull straight from audio)
        b = float(self.audio.b_freq); l = float(self.audio.l_freq)
        # Normalize to a mode index range
        def freq_to_mode(f, base=55.0):
            ratio = max(0.1, f/base); m = int(np.clip(np.round(2 + 6*np.log2(ratio)), 1, 16))
            n = int(np.clip(np.round(2 + 5*np.sqrt(max(0.0, ratio-0.5))), 1, 16))
            return m, n
        m1,n1 = freq_to_mode(b); m2,n2 = freq_to_mode(l)

        N = 256
        x = np.linspace(-1,1,N); y = np.linspace(-1,1,N); X,Y = np.meshgrid(x,y)
        R = np.sqrt(X*X + Y*Y) + 1e-6; TH = np.arctan2(Y,X)
        # Mix of circular (Bessel-ish) and rectangular (Chladni-ish) modes
        pat1 = np.sin(m1*np.pi*X) * np.sin(n1*np.pi*Y)      # rectangular-ish
        pat2 = np.cos(m2*TH) * np.sin(n2*np.pi*R) / (1+3*R) # radial-ish
        img = 0.6*pat1 + 0.8*pat2
        img = (img - img.min()) / (img.ptp() + 1e-9)
        self.cym_img.set_data(img.astype(np.float32))
        self.cym_text.setText(f"Cymatics — modes (rect {m1},{n1}) (rad {m2},{n2})")

    def _update_particles(self):
        if self.core.torus_buffer:
            n = min(40, len(self.core.torus_buffer)); samples = list(self.core.torus_buffer)[-n:]
            pos=[]
            for d in samples:
                u = d.phase + self._vis_frame*0.02; v = d.energy*2*np.pi
                R, rr = self.core.major_radius, self.core.minor_radius*0.9
                x=(R+rr*np.cos(v))*np.cos(u); y=(R+rr*np.cos(v))*np.sin(u); z=rr*np.sin(v)
                pos.append([x,y,z])
            self.torus_particles.set_data(np.array(pos), face_color="yellow", edge_color="white", edge_width=0.5, size=10)
        else:
            self.torus_particles.set_data(pos=np.zeros((0,3)))

        def upd_sphere(s, particles, col):
            pts=[]; 
            for d in s.memory_ring:
                if "sphere_position" in (d.metadata or {}): pts.append(d.metadata["sphere_position"])
            if pts: particles.set_data(np.array(pts), face_color=col, edge_color="white", edge_width=0.5, size=14, symbol="star")
            else: particles.set_data(pos=np.zeros((0,3)))
        upd_sphere(self.core.sphere_s1, self.s1_particles, "orange")
        if self.core.sphere_s2: upd_sphere(self.core.sphere_s2, self.s2_particles, "cyan")
        if self.core.sphere_s3: upd_sphere(self.core.sphere_s3, self.s3_particles, "lime")

    def _update_stats(self):
        st = self.core.metrics; total = st["total_processed"]; tor = st["torus_direct"]
        eff = (100 - (tor/total*100)) if total>0 else 0
        s = "BERLIN × TRI-HARMONIC — STATS\n\n"
        s += f"Processed: {total}\n"
        s += f"S1 entries: {st['s1_entries']}  | S2: {st.get('s2_entries',0)}  | S3: {st.get('s3_entries',0)}\n"
        s += f"Torus direct: {tor}  | Routing efficiency: {eff:.1f}%\n"
        s += f"Freqs: Bass {self.audio.b_freq:.1f} Hz  | Lead {self.audio.l_freq:.1f} Hz\n"
        s += f"Delay: {self.audio.delay_ms:.0f} ms  fb {self.audio.delay_fb*100:.0f}%  mix {self.audio.delay_mix*100:.0f}%\n"
        self.stats.setPlainText(s)

    def _tick(self):
        self._vis_frame += 1; self._sim_frame += 1
        self._update_pulse_rings(); self._update_particles(); self._update_cymatics(); self._update_stats()

    def run(self):
        self.canvas.show(); self.win.resize(1720, 980); self.win.show()
        try:
            self.app.exec_()
        finally:
            self.audio.stop()

# -----------------------------------------------------------------------------#
# Main
# -----------------------------------------------------------------------------#
def main():
    print("\n" + "="*76)
    print("  BERLIN × TRI-HARMONIC — 16-Step Sequencer • Spheres • Cymatics • Delay")
    print("="*76)
    ui = BerlinTriVis()
    ui.run()

if __name__ == "__main__":
    main()
