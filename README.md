# 🌌 N-Body Physics Simulation (High‑Performance Python Engine)

A **high‑performance, CPU‑optimized N‑body gravitational simulation** written in Python. This project started as a simple educational planetary simulation and evolved into a **data‑oriented, multi‑core, physics‑accurate engine** capable of handling hundreds to thousands of bodies in real time.

This README explains **what the project is**, **how it works internally**, and **why specific engineering decisions were made**, from a *senior‑developer / engine‑programmer* perspective.

---

## 📌 Project Overview

This simulation models Newtonian gravity between multiple bodies using real physical equations and numerically stable integration techniques. It is designed to:

* Scale efficiently on modern CPUs
* Remain stable over long simulation times
* Separate physics performance from rendering limitations
* Demonstrate real‑world optimization strategies used in engines and scientific computing

The simulation supports:

* Multi‑body gravitational systems
* Orbit trails
* Collision and body merging
* Camera panning, zooming, and following
* Preset systems (solar‑like, binary stars, stress tests)

---

## 🎯 Design Goals

1. **CPU‑aware performance** (cache, SIMD, parallelism)
2. **Numerical stability** (long‑term energy behavior)
3. **Scalability** (hundreds → thousands of bodies)
4. **Clear separation of concerns** (physics vs rendering)
5. **Educational clarity without sacrificing correctness**

This is *not* a toy demo — it is a small physics engine.

---

## 🧠 Core Engineering Principles Used

### 1️⃣ Data‑Oriented Design (SoA)

Instead of object‑per‑planet (AoS), the engine uses **Struct‑of‑Arrays**:

* Positions → contiguous array
* Velocities → contiguous array
* Masses → contiguous array

This dramatically improves:

* CPU cache locality
* Memory bandwidth usage
* SIMD vectorization

Modern CPUs prefer **data layouts**, not objects.

---

### 2️⃣ Numba JIT Compilation

The physics core is compiled using **Numba**:

* Eliminates Python interpreter overhead
* Enables SIMD (vectorized math)
* Allows safe multi‑core parallelism

```text
Python loop → ~10–50 million ops/sec
Numba loop  → hundreds of millions ops/sec
```

---

### 3️⃣ Parallel Physics (Multi‑Core Scaling)

Physics updates use parallel loops where:

* Each body computes its own net force
* No shared writes between threads
* Minimal cache contention

This allows near‑linear scaling on multi‑core CPUs.

---

### 4️⃣ Verlet Integration (Physics Stability)

Instead of Euler integration, the engine uses **Verlet integration**:

* Better energy conservation
* Fewer floating‑point dependencies
* Larger stable timesteps

This prevents orbit decay and explosion over time.

---

### 5️⃣ Gravitational Softening

A softening constant is applied to distance calculations to:

* Avoid singularities
* Prevent NaNs and denormal floats
* Improve numerical stability
* Protect CPU pipelines from slow paths

This is standard practice in astrophysical simulations.

---

## ⚙️ Physics Model

### Newton’s Law of Gravitation

For each pair of bodies:

[ F = G \frac{m_1 m_2}{r^2} ]

Force is decomposed into x/y components and accumulated per body.

### Collision Handling

When bodies intersect:

* Bodies merge
* Mass is conserved
* Momentum is conserved
* Radius is recomputed

This allows realistic accretion behavior.

---

## 🎮 Rendering & Interaction

Rendering is handled via **Pygame** and intentionally kept separate from physics logic.

### Camera System

* Pan (mouse drag)
* Zoom (scroll wheel)
* Follow body mode

### Visual Features

* Orbit trails with capped history
* HUD with FPS, body count, simulation time
* Cached text rendering to reduce draw cost

Rendering is intentionally not optimized beyond reason — the focus is physics throughput.

---

## 🧪 Performance Characteristics

### What FPS Means (Important)

FPS ≠ physics performance.

Correct performance metric:

```
Bodies × Physics Steps / Second
```

Example:

* 100 bodies @ 240 FPS
* 400 bodies @ 240 FPS

→ **4× real performance improvement**

---

### CPU Scaling Behavior

| CPU Type                   | Expected Behavior       |
| -------------------------- | ----------------------- |
| Low‑end dual core          | Physics‑limited quickly |
| Modern desktop CPU         | Rendering‑limited       |
| Workstation (Threadripper) | Physics nearly free     |

The engine scales with **core count**, **cache size**, and **memory bandwidth**.

---

## 📂 Project Structure

```
main.py
│
├── Physics core (Numba‑accelerated)
├── Data storage (SoA arrays)
├── Collision system
├── Camera system
├── Preset generators
├── HUD / diagnostics
└── Main loop
```

The project is intentionally single‑file for ease of experimentation, but can be modularized easily.

---

## ▶️ Running the Simulation

### Requirements

* Python 3.9+
* pygame
* numpy
* numba

### Run

```bash
python main.py
```

---

## ⌨️ Controls (Typical)

* **Mouse Drag** → Pan camera
* **Scroll Wheel** → Zoom
* **Click / Drag** → Create body with velocity
* **Keyboard shortcuts** → Presets, pause, reset

(Exact bindings depend on the current configuration.)

---

## 🚀 Future Improvements

Planned or possible extensions:

* Barnes–Hut (O(N log N)) gravity
* Headless physics benchmarking mode
* Save / load simulation states
* OpenGL rendering backend
* GPU compute (CUDA / OpenCL)
* Energy and momentum diagnostics

---

## 🧑‍💻 Author Notes

This project demonstrates:

* Real CPU‑level optimization thinking
* Data‑oriented programming
* Numerical simulation principles
* Practical engine architecture

It intentionally avoids "magic" abstractions to keep performance behavior transparent.

---

## 🏁 Final Note

This is not just a simulation — it is a **learning platform for real systems programming concepts**, written in Python but designed with **C/C++ engine principles** in mind.

If you understand this code, you understand **how real engines think**.

---

🔥 Happy simulating. Push the limits.
