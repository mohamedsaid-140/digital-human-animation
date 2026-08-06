"""
skeleton.py
-----------
Core system representation: a hierarchical joint skeleton driven by
Forward Kinematics (FK). Each joint stores only its LOCAL transform
relative to its parent; global transforms are derived by walking the
hierarchy (the standard parent-relative joint model used by production
character rigs).
"""

from math3d import Vec3, Mat4, Quaternion


class Joint:
    def __init__(self, name, parent_index, local_translation: Vec3):
        self.name = name
        self.parent_index = parent_index
        self.local_translation = local_translation
        self.local_rotation = Quaternion.identity()


class Skeleton:
    def __init__(self):
        self.joints = []
        self.name_to_index = {}

    def add_joint(self, name, parent_name, local_translation: Vec3):
        parent_index = self.name_to_index[parent_name] if parent_name is not None else -1
        j = Joint(name, parent_index, local_translation)
        self.name_to_index[name] = len(self.joints)
        self.joints.append(j)
        return self.name_to_index[name]

    def set_pose(self, name, rotation: Quaternion):
        self.joints[self.name_to_index[name]].local_rotation = rotation

    def reset_pose(self):
        for j in self.joints:
            j.local_rotation = Quaternion.identity()

    def local_matrix(self, joint: Joint) -> Mat4:
        return Mat4.translation(joint.local_translation) * joint.local_rotation.to_mat4()

    def forward_kinematics(self):
        """G_i = G_parent(i) * L_i,  G_root = L_root."""
        globals_ = [None] * len(self.joints)
        for i, j in enumerate(self.joints):
            L = self.local_matrix(j)
            if j.parent_index == -1:
                globals_[i] = L
            else:
                globals_[i] = globals_[j.parent_index] * L
        return globals_

    def joint_positions(self):
        globals_ = self.forward_kinematics()
        pts = [Vec3(g.get(0, 3), g.get(1, 3), g.get(2, 3)) for g in globals_]
        return pts, globals_


def build_humanoid_skeleton():
    s = Skeleton()
    s.add_joint("pelvis", None, Vec3(0, 0, 0))
    s.add_joint("spine",       "pelvis", Vec3(0, 0.15, 0))
    s.add_joint("chest",       "spine",  Vec3(0, 0.20, 0))
    s.add_joint("neck",        "chest",  Vec3(0, 0.20, 0))
    s.add_joint("head",        "neck",   Vec3(0, 0.10, 0))

    s.add_joint("clavicle_l",  "chest",  Vec3(0.08, 0.15, 0))
    s.add_joint("upperarm_l",  "clavicle_l", Vec3(0.12, 0.0, 0))
    s.add_joint("forearm_l",   "upperarm_l", Vec3(0.28, 0.0, 0))
    s.add_joint("hand_l",      "forearm_l",  Vec3(0.25, 0.0, 0))

    s.add_joint("clavicle_r",  "chest",  Vec3(-0.08, 0.15, 0))
    s.add_joint("upperarm_r",  "clavicle_r", Vec3(-0.12, 0.0, 0))
    s.add_joint("forearm_r",   "upperarm_r", Vec3(-0.28, 0.0, 0))
    s.add_joint("hand_r",      "forearm_r",  Vec3(-0.25, 0.0, 0))

    s.add_joint("thigh_l",     "pelvis", Vec3(0.09, -0.05, 0))
    s.add_joint("shin_l",      "thigh_l", Vec3(0, -0.42, 0))
    s.add_joint("foot_l",      "shin_l",  Vec3(0, -0.40, 0))

    s.add_joint("thigh_r",     "pelvis", Vec3(-0.09, -0.05, 0))
    s.add_joint("shin_r",      "thigh_r", Vec3(0, -0.42, 0))
    s.add_joint("foot_r",      "shin_r",  Vec3(0, -0.40, 0))

    return s


BONES = [
    ("pelvis", "spine"), ("spine", "chest"), ("chest", "neck"), ("neck", "head"),
    ("chest", "clavicle_l"), ("clavicle_l", "upperarm_l"), ("upperarm_l", "forearm_l"), ("forearm_l", "hand_l"),
    ("chest", "clavicle_r"), ("clavicle_r", "upperarm_r"), ("upperarm_r", "forearm_r"), ("forearm_r", "hand_r"),
    ("pelvis", "thigh_l"), ("thigh_l", "shin_l"), ("shin_l", "foot_l"),
    ("pelvis", "thigh_r"), ("thigh_r", "shin_r"), ("shin_r", "foot_r"),
]
