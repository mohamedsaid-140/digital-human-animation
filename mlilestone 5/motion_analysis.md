# Milestone 5 — Motion Analysis

## System under test
The Milestone 1–2 skeleton (17-joint FK hierarchy) and LBS-skinned mesh now
carry two independent temporal drivers, both operating purely through
`skeleton.set_pose(joint, quaternion)` — no changes to the representation
were needed to add time:

1. **`KeyframeClip`** — sparse per-joint quaternion keys, evaluated with
   SLERP between the surrounding keys (`animation.py`). Used for the
   hand-authored wave (`out/m5_wave.gif`).
2. **`procedural_walk_cycle`** — closed-form sinusoidal locomotion driven by
   one phase variable, with anatomically correct left/right and
   contralateral arm/leg phase offsets (`out/m5_walk.gif`).

## Keyframe interpolation quality
The top-left plot (`m5_motion_analysis.png`) tracks the right hand's world-space
height through the wave clip. The curve is continuous and C¹-smooth at every
keyframe boundary — there is no visible "popping" or velocity discontinuity
where tracks hand off between keys. This is the expected behavior of SLERP
interpolation between unit quaternions (see `stability_analysis.md` for why
SLERP is preferred to per-component linear interpolation of Euler angles).

## Gait analysis (procedural walk)
The top-right plot shows vertical trajectories of the pelvis (root bob) and
both feet over two full gait cycles. The feet alternate correctly — each
foot's height oscillates with a half-cycle phase offset from the other,
and the pelvis bob shows the expected double-frequency component (it rises
slightly during **both** the left-stance and right-stance sub-phases, not
once per full stride). This confirms the phase-offset math in
`procedural_walk_cycle` is internally consistent, not just visually
plausible per-limb.

## Foot-sliding (motion consistency) metric
Bottom-left: horizontal velocity of the left foot. A physically correct
walk holds each foot's horizontal velocity at ≈0 while it is planted
(stance phase) and only moves it during swing. Our measured curve is
**flat at ≈0.55 m/s for the entire cycle**, including while the foot is at
its lowest point — i.e. the planted foot is sliding along the ground at
the character's full walking speed.

**Root cause:** the system has no foot-plant/IK correction layer. The root
translates the whole skeleton forward at a constant rate independent of
which foot is on the ground; only the hip/knee angles are animated. This is
a known, honestly-reported limitation of pure forward-kinematic locomotion
without inverse kinematics, and it is exactly the kind of defect this
milestone's evaluation is meant to surface (see
`stability_analysis.md`/realism evaluation for the numeric value and its
interpretation, and the Milestone 6 roadmap for the IK fix).

## Secondary motion (ponytail) response
The ponytail's particle positions were driven by the animated head-joint
position every frame. Qualitatively (visible in `m5_walk.gif`), the chain
lags the head's motion and swings with a damped oscillation on each
step-induced bob — the expected qualitative signature of underdamped
mass-spring/Verlet secondary motion, not simply following the head rigidly.
The quantitative stability of this simulation (constraint error over time)
is analyzed in `stability_analysis.md`.
