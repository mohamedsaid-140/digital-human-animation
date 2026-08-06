"""
animation.py
------------
Milestone 5, part A: temporal control of the skeleton.

Two animation sources are implemented, both driving the SAME skeleton
representation from Milestones 1-2 (no changes to skeleton.py needed --
temporal behavior is layered on top of the static pose system):

1. KeyframeClip -- sparse (time -> per-joint quaternion) keyframes,
   evaluated at arbitrary t via per-joint SLERP between the two
   surrounding keys. This is the standard non-procedural animation
   representation (as authored by hand or from mocap).

2. procedural_walk_cycle -- a phase-driven analytic walk cycle: hips,
   knees and shoulders driven by sinusoids of a single phase variable,
   with the correct phase offsets between left/right limbs and a
   contralateral arm swing. This is "procedural" animation: no stored
   keyframes, just a closed-form function of time.
"""

import math
from math3d import Vec3, Quaternion


class KeyframeClip:
    def __init__(self, duration):
        self.duration = duration
        self.tracks = {}   # joint_name -> list[(time, Quaternion)], sorted by time

    def add_key(self, joint_name, time, quat: Quaternion):
        self.tracks.setdefault(joint_name, []).append((time, quat))
        self.tracks[joint_name].sort(key=lambda k: k[0])

    def sample(self, joint_name, t):
        keys = self.tracks.get(joint_name)
        if not keys:
            return Quaternion.identity()
        if t <= keys[0][0]:
            return keys[0][1]
        if t >= keys[-1][0]:
            return keys[-1][1]
        for i in range(len(keys) - 1):
            t0, q0 = keys[i]
            t1, q1 = keys[i + 1]
            if t0 <= t <= t1:
                local_t = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                return Quaternion.slerp(q0, q1, local_t)
        return keys[-1][1]

    def apply(self, skeleton, t):
        tw = t % self.duration
        for joint_name in self.tracks:
            skeleton.set_pose(joint_name, self.sample(joint_name, tw))


def build_wave_clip():
    clip = KeyframeClip(duration=2.4)
    up = Quaternion.from_axis_angle(Vec3(0, 0, 1), math.radians(-100))
    wave_a = Quaternion.from_axis_angle(Vec3(0, 1, 0), math.radians(35))
    wave_b = Quaternion.from_axis_angle(Vec3(0, 1, 0), math.radians(-35))
    rest = Quaternion.identity()

    clip.add_key("upperarm_r", 0.0, rest)
    clip.add_key("upperarm_r", 0.5, up)
    clip.add_key("upperarm_r", 2.0, up)
    clip.add_key("upperarm_r", 2.4, rest)

    clip.add_key("forearm_r", 0.0, rest)
    clip.add_key("forearm_r", 0.5, wave_a)
    clip.add_key("forearm_r", 0.9, wave_b)
    clip.add_key("forearm_r", 1.3, wave_a)
    clip.add_key("forearm_r", 1.7, wave_b)
    clip.add_key("forearm_r", 2.4, rest)

    clip.add_key("head", 0.0, rest)
    clip.add_key("head", 0.5, Quaternion.from_axis_angle(Vec3(0, 1, 0), math.radians(-12)))
    clip.add_key("head", 2.4, rest)
    return clip


def procedural_walk_cycle(skeleton, t, stride_hz=1.0, hip_amp_deg=28, knee_amp_deg=45, arm_amp_deg=22):
    phi = 2.0 * math.pi * stride_hz * t

    hip_l = math.sin(phi) * math.radians(hip_amp_deg)
    hip_r = math.sin(phi + math.pi) * math.radians(hip_amp_deg)
    knee_l = max(0.0, math.sin(phi + math.pi * 0.15)) * math.radians(knee_amp_deg)
    knee_r = max(0.0, math.sin(phi + math.pi + math.pi * 0.15)) * math.radians(knee_amp_deg)

    arm_l = math.sin(phi + math.pi) * math.radians(arm_amp_deg)
    arm_r = math.sin(phi) * math.radians(arm_amp_deg)

    bob = abs(math.sin(phi)) * 0.015

    skeleton.set_pose("thigh_l", Quaternion.from_axis_angle(Vec3(1, 0, 0), hip_l))
    skeleton.set_pose("thigh_r", Quaternion.from_axis_angle(Vec3(1, 0, 0), hip_r))
    skeleton.set_pose("shin_l", Quaternion.from_axis_angle(Vec3(1, 0, 0), -knee_l))
    skeleton.set_pose("shin_r", Quaternion.from_axis_angle(Vec3(1, 0, 0), -knee_r))

    # Arms rest in a T-pose (bone offset along local +/-x). A joint's own
    # rotation only moves its CHILDREN, not the segment connecting it to its
    # own parent -- so the swing has to be driven from the clavicle (parent
    # of upperarm), not the upperarm joint itself. We also compose in a
    # fixed "bias" rotation that brings the arm down from horizontal (T-pose)
    # to vertical (A-pose) before adding the swing on top of that.
    bias_l = Quaternion.from_axis_angle(Vec3(0, 0, 1), math.radians(-85))
    bias_r = Quaternion.from_axis_angle(Vec3(0, 0, 1), math.radians(85))
    swing_l = Quaternion.from_axis_angle(Vec3(1, 0, 0), arm_l)
    swing_r = Quaternion.from_axis_angle(Vec3(1, 0, 0), arm_r)
    skeleton.set_pose("clavicle_l", swing_l * bias_l)
    skeleton.set_pose("clavicle_r", swing_r * bias_r)

    skeleton.set_pose("chest", Quaternion.from_axis_angle(Vec3(0, 1, 0), math.sin(phi) * math.radians(6)))
    return bob
