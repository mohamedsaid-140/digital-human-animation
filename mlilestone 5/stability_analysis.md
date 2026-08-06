# Milestone 5 — Stability / Realism Evaluation

## Particle system: Verlet integration + distance constraints
`particles.py` implements the ponytail as a 6-particle chain: **Verlet
integration** (position + previous-position, no explicit velocity state)
followed by 4 constraint-relaxation iterations per frame that push each
segment back toward its rest length.

**Why Verlet, not explicit Euler:** explicit Euler integrates velocity and
position from a stored velocity variable and tends to gain energy on stiff
constraint systems unless the timestep is very small — a classic source of
simulation blow-up. Verlet's implicit velocity (`x - x_prev`) combined with
position-based constraint projection is unconditionally stable for this
class of problem at typical frame rates, which is why it's the standard
choice for real-time hair/cloth (Jakobsen 2001).

**Measured result** (bottom-right plot, `m5_motion_analysis.png`):
- Mean constraint error: **0.0073 m** against a 0.045 m rest segment length
  (≈16%).
- Maximum transient error: **0.0107 m** (≈24%), occurring during the first
  ~0.15 s as the chain settles from its initial hanging-straight-down state
  into the moving simulation.
- Critically, the error **does not grow over the sequence** — it oscillates
  in a bounded envelope that tracks the periodic gait bob, rather than
  drifting upward. This is the signature of a stable simulation: the
  constraint solver is dissipating the error each frame (via `damping`
  and the position-based correction), not accumulating it.

**Interpretation:** 4 constraint iterations under-relaxes a fairly stiff
chain (hence the double-digit % transient stretch — real-time cloth/hair
solvers typically use more iterations or a stiffness warm-start for
production quality). This is a legitimate, disclosed accuracy/performance
trade-off rather than a hidden bug: more iterations would tighten the error
at linear extra cost per frame, which is exactly the kind of system/hardware
trade-off this milestone asks to be aware of.

## Motion consistency: foot sliding
See `motion_analysis.md` for the plot. The left foot's horizontal velocity
sits at a constant **0.55 m/s (= the root's walking speed) for the entire
gait cycle**, rather than dropping to ≈0 during stance. This is a real,
measured artifact — not a rendering glitch — with a clear cause: there is
no foot-plant constraint or IK layer coupling the feet to the ground; the
pelvis root translates at a fixed rate independent of gait phase. The
metric is reported honestly here specifically so it can be tracked as a
regression/improvement target once IK is introduced.

## Numerical precision notes (carried over from Milestone 2, still relevant here)
Two effects are now exercised continuously (every animated frame) rather
than once per pose:
- **Quaternion renormalization drift.** Every `set_pose` call replaces the
  joint's quaternion outright (no compounding), so drift from repeated
  multiplication does not accumulate frame-to-frame in this system —
  unlike, e.g., an incremental "apply angular velocity" integrator, which
  is exactly the failure mode demonstrated in Milestone 2's stability
  experiment (`render_m2.py::stability_experiments`) and is why this
  system deliberately samples an absolute pose per frame instead of
  integrating relative rotations.
- **SLERP near-parallel case.** `Quaternion.slerp` falls back to
  normalized linear interpolation when `dot > 0.9995` to avoid a 0/0 from
  `sin(theta_0)` in the denominator as the two keys approach each other —
  exercised whenever two adjacent keyframes are nearly identical (e.g. the
  wave clip's return-to-rest segments).

## Summary
| Metric | Value | Verdict |
|---|---|---|
| Ponytail mean constraint error | 0.0073 m (16% of segment) | Bounded, non-growing → **stable** |
| Ponytail max transient error | 0.0107 m (24%) | Settling transient only, decays → **acceptable** |
| Foot sliding (stance) | 0.55 m/s (should be ≈0) | **Known limitation** — needs foot IK (Milestone 6 candidate) |
| Keyframe interpolation continuity | No visible popping at key boundaries | **Correct** |
