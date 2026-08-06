"""
Milestone 1 — Digital Humans & Character Animation
Core mathematical + representational foundation.

Everything here is built from first principles on top of raw NumPy:
- Quaternion algebra (construction, multiplication, normalization, slerp, to-matrix)
- Homogeneous 4x4 transforms
- A joint-hierarchy Skeleton with Forward Kinematics (FK)
- A triangulated tube mesh generator (procedural, no external mesh assets)
- Linear Blend Skinning (LBS) binding the mesh to the skeleton

No animation/graphics libraries are used for the math itself — only NumPy for
array storage and matplotlib for the final visualization (the "output mechanism").
"""

import numpy as np

# ----------------------------------------------------------------------------
# 1. QUATERNION ALGEBRA
# ----------------------------------------------------------------------------
# A quaternion q = w + xi + yj + zk is stored as a 4-vector [w, x, y, z].
# Unit quaternions represent 3D rotations without gimbal lock and compose
# cheaply via the Hamilton product.

def quat_identity():
    return np.array([1.0, 0.0, 0.0, 0.0])


def quat_from_axis_angle(axis, angle_rad):
    """q = [cos(theta/2), sin(theta/2) * axis]"""
    axis = axis / np.linalg.norm(axis)
    half = angle_rad / 2.0
    w = np.cos(half)
    xyz = np.sin(half) * axis
    return np.array([w, *xyz])


def quat_normalize(q):
    return q / np.linalg.norm(q)


def quat_multiply(q1, q2):
    """Hamilton product q1 * q2 (apply q2 first, then q1)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_to_matrix(q):
    """Convert unit quaternion to a 3x3 rotation matrix.

    R = I + 2w[v]_x + 2[v]_x^2   (Rodrigues form in quaternion terms)
    """
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [2*(x*z - w*y),         2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def quat_slerp(q0, q1, t):
    """Spherical linear interpolation between two unit quaternions."""
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    dot = np.dot(q0, q1)
    if dot < 0.0:  # take the shorter arc
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    theta0 = np.arccos(dot)
    if theta0 < 1e-6:
        return quat_normalize(q0 + t * (q1 - q0))
    sin_theta0 = np.sin(theta0)
    a = np.sin((1 - t) * theta0) / sin_theta0
    b = np.sin(t * theta0) / sin_theta0
    return a * q0 + b * q1


# ----------------------------------------------------------------------------
# 2. HOMOGENEOUS TRANSFORMS
# ----------------------------------------------------------------------------

def make_transform(translation, quat):
    """Build a 4x4 homogeneous transform T = [R t; 0 1]."""
    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(quat)
    T[:3, 3] = translation
    return T


def transform_point(T, p):
    ph = np.array([p[0], p[1], p[2], 1.0])
    return (T @ ph)[:3]


# ----------------------------------------------------------------------------
# 3. SKELETON REPRESENTATION + FORWARD KINEMATICS
# ----------------------------------------------------------------------------

class Joint:
    def __init__(self, name, parent_index, bind_translation):
        self.name = name
        self.parent_index = parent_index          # -1 for root
        self.bind_translation = np.array(bind_translation, dtype=float)  # offset from parent, bind pose
        self.local_rotation = quat_identity()      # pose rotation, animatable


class Skeleton:
    """A joint hierarchy (tree, stored flat with parent indices)."""

    def __init__(self):
        self.joints = []
        self.name_to_index = {}

    def add_joint(self, name, parent_name, bind_translation):
        parent_index = self.name_to_index[parent_name] if parent_name is not None else -1
        j = Joint(name, parent_index, bind_translation)
        self.joints.append(j)
        self.name_to_index[name] = len(self.joints) - 1
        return len(self.joints) - 1

    def set_local_rotation(self, name, quat):
        self.joints[self.name_to_index[name]].local_rotation = quat

    def forward_kinematics(self):
        """
        Core FK recurrence:
            W_root  = L_root
            W_j     = W_parent(j) @ L_j
        where L_j is the joint's local transform (bind_translation, local_rotation).
        Returns a list of 4x4 world transforms, one per joint, in hierarchy order.
        """
        world = [None] * len(self.joints)
        for i, j in enumerate(self.joints):
            L = make_transform(j.bind_translation, j.local_rotation)
            if j.parent_index == -1:
                world[i] = L
            else:
                world[i] = world[j.parent_index] @ L
        return world

    def bind_pose_world(self):
        """World transforms with all local rotations at identity (T-pose)."""
        saved = [j.local_rotation.copy() for j in self.joints]
        for j in self.joints:
            j.local_rotation = quat_identity()
        world = self.forward_kinematics()
        for j, q in zip(self.joints, saved):
            j.local_rotation = q
        return world


# ----------------------------------------------------------------------------
# 4. PROCEDURAL MESH: TUBES ALONG EACH BONE
# ----------------------------------------------------------------------------

def make_tube_mesh(p0, p1, radius=0.05, segments=8):
    """
    Build a cylindrical tube from point p0 to p1.
    Returns (vertices Nx3, t_values Nx1) where t_values in [0,1] is the
    parametric position of each vertex along the bone (0 = p0, 1 = p1) —
    this is what skinning weights will be derived from.
    """
    p0, p1 = np.array(p0), np.array(p1)
    axis = p1 - p0
    length = np.linalg.norm(axis)
    if length < 1e-8:
        axis_dir = np.array([0, 0, 1.0])
    else:
        axis_dir = axis / length

    # build an orthonormal frame around axis_dir
    tmp = np.array([1.0, 0, 0]) if abs(axis_dir[0]) < 0.9 else np.array([0, 1.0, 0])
    side1 = np.cross(axis_dir, tmp)
    side1 /= np.linalg.norm(side1)
    side2 = np.cross(axis_dir, side1)

    verts = []
    tvals = []
    rings = 6  # rings along the length
    for ring in range(rings + 1):
        t = ring / rings
        center = p0 + axis * t
        for s in range(segments):
            theta = 2 * np.pi * s / segments
            offset = radius * (np.cos(theta) * side1 + np.sin(theta) * side2)
            verts.append(center + offset)
            tvals.append(t)
    return np.array(verts), np.array(tvals)


# ----------------------------------------------------------------------------
# 5. LINEAR BLEND SKINNING (LBS)
# ----------------------------------------------------------------------------
#
# For a vertex v bound with weights w_j to joints j:
#
#     v_deformed = sum_j  w_j * ( W_j @ inv(B_j) ) @ v_bind
#
# where B_j is the joint's world transform in the BIND pose and W_j is its
# world transform in the CURRENT pose. (W_j @ inv(B_j)) is the "skinning
# matrix" — it maps a point from bind-pose world space into joint j's local
# frame and then back out into the posed world space.

class SkinnedMesh:
    def __init__(self, skeleton: Skeleton):
        self.skeleton = skeleton
        self.vertices_bind = []      # world-space bind positions, Nx3
        self.weights = []            # list of dict{joint_index: weight}
        self.faces = []              # optional, for wireframe/triangles

    def add_bone_tube(self, parent_joint_name, child_joint_name, bind_world,
                       radius=0.05, segments=8):
        """
        Generate a tube mesh for the bone between two joints and bind its
        vertices with a linear weight blend: t=0 -> 100% parent joint,
        t=1 -> 100% child joint. This is the smooth-blend LBS case that
        avoids the classic rigid 'candy-wrapper' joint collapse.
        """
        s = self.skeleton
        pi = s.name_to_index[parent_joint_name]
        ci = s.name_to_index[child_joint_name]
        p0 = bind_world[pi][:3, 3]
        p1 = bind_world[ci][:3, 3]

        verts, tvals = make_tube_mesh(p0, p1, radius=radius, segments=segments)
        base_index = len(self.vertices_bind)

        for v, t in zip(verts, tvals):
            self.vertices_bind.append(v)
            self.weights.append({pi: 1.0 - t, ci: t})

        # simple ring-to-ring quad (as two triangles) connectivity for wireframe rendering
        rings = 6
        for r in range(rings):
            for s_ in range(segments):
                a = base_index + r * segments + s_
                b = base_index + r * segments + (s_ + 1) % segments
                c = base_index + (r + 1) * segments + s_
                d = base_index + (r + 1) * segments + (s_ + 1) % segments
                self.faces.append((a, b, d))
                self.faces.append((a, d, c))

        return base_index

    def bind(self, bind_world):
        """Store inverse bind matrices per joint for later skinning."""
        self.inv_bind = [np.linalg.inv(W) for W in bind_world]
        self.vertices_bind = np.array(self.vertices_bind)

    def deform(self, posed_world):
        """
        Apply the LBS equation above for every vertex.
        Returns an array of deformed (posed) vertex positions, Nx3.
        """
        out = np.zeros_like(self.vertices_bind)
        for vi, v in enumerate(self.vertices_bind):
            v_h = np.array([v[0], v[1], v[2], 1.0])
            acc = np.zeros(4)
            for j, w in self.weights[vi].items():
                skin_matrix = posed_world[j] @ self.inv_bind[j]
                acc += w * (skin_matrix @ v_h)
            out[vi] = acc[:3]
        return out
