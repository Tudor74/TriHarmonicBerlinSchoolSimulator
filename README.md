Berlin × Tri-Harmonic — Sequencer + Spheres + Cymatics
An interactive, audiovisual synthesizer environment combining a Berlin-style 16-step sequencer, a Tri-Harmonic routing core with symbolic spheres, and a real-time cymatics visualizer.
Built with PyQt5, VisPy, and SoundDevice, it blends algorithmic music generation, 3D geometry, and procedural graphics into a tightly integrated performance tool.

Features
Berlin-style 16-step sequencer

Bass & lead voices with independent waveforms (saw, square, sine, triangle, pulse/PWM, supersaw)

Kick, snare, and hi-hat patterns with step-level control

Swing timing, random pattern generation, and BPM control

Tri-Harmonic routing core

Three symbolic spheres (S1: Temporal Memory, S2: Semantic Drift, S3: Harmonic Feedback)

Real-time phase/energy-based resonance gating and routing

Toroidal particle buffer for unassigned data pulses

Audio engine

Phase-accurate oscillators with PWM and detune

One-pole low-pass filters for each voice

Soft-clipping drive, smoothed fractional-delay with feedback/mix controls

Vibrato modulation with adjustable depth and rate

3D visualisation

Rotating torus with sphere wireframes

Live particle systems for spheres and torus buffer

Expanding pulse-ring effects for triggered events

Cymatics simulation panel

Real-time standing-wave patterns driven by current bass/lead frequencies

Blends Chladni-style rectangular modes and radial Bessel-like modes

UI controls

BPM, swing, waveform, PWM, detune, cutoff, drive, vibrato, delay controls

Randomize patterns button

Live performance stats (routing efficiency, frequency readouts, delay settings)

Requirements
Python 3.8+

PyQt5

vispy

sounddevice

All dependencies auto-install on first run.

Run
bash
Copy
Edit
python berlin_triharmonic.py
