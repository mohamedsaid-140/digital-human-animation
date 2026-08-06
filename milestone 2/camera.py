"""
Milestone 2 — Space, Transformation & Camera
Extends the Milestone 1 skeletal/skinning system with a complete,
from-first-principles 3D transformation pipeline:

    Model space --(world transform, M1)--> World space
    World space --(view matrix)--------> Camera (eye) space
    Camera space --(projection matrix)--> Clip space
    Clip space  --(perspective divide)--> Normalized Device Coords (NDC)
    NDC         --(viewport transform)--> Screen / pixel space

No rendering library (no OpenGL/three.js/mplot3d 3D projection) is used —
every stage below is an explicit matrix built from scratch, exactly mirroring
the fixed pipeline stages of a real GPU (vertex shader MVP transform ->
clipping -> perspective divide -> viewport / rasterizer).
"""

import numpy as np


# ----------------------------------------------------------------------------
# 1. VIEW MATRIX  (world space -> camera/eye space)
# ----------------------------------------------------------------------------

def look_at(eye, target, world_up=(0, 1, 0)):
    """
    Build a right-handed view matrix from an eye position, a look-at target,
    and a world 'up' hint.

    Derivation: construct an orthonormal camera basis (right, up, forward)
    directly from the two input directions, then express world points in
    that basis with the eye translated to the origin:

        forward f = normalize(target - eye)      (camera looks down -f in RH convention)
        right   s = normalize(f x world_up)
        up      u = s x f                         (re-orthogonalized against f)

    The view matrix is the inverse of the camera's world transform. Because
    the camera's world transform is a pure rotation R (columns s, u, -f) plus
    a translation (eye), its inverse is cheap to compute in closed form:
    inverse of an orthonormal rotation is its transpose, and the inverse of
    a translation is its negation — there is no need for a general 4x4
    matrix inverse (which would be numerically riskier and slower).
    """
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    world_up = np.asarray(world_up, dtype=float)

    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, world_up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)

    # Rotation part is R^T (transpose = inverse for an orthonormal basis)
    R = np.eye(4)
    R[0, :3] = s
    R[1, :3] = u
    R[2, :3] = -f

    T = np.eye(4)
    T[:3, 3] = -eye

    return R @ T  # rotate first conceptually, but matrix form is R * T applied to points


# ----------------------------------------------------------------------------
# 2. PROJECTION MATRICES  (camera space -> clip space)
# ----------------------------------------------------------------------------

def perspective(fovy_rad, aspect, near, far):
    """
    Standard right-handed perspective projection (OpenGL-style clip space,
    z in [-1, 1] after the divide).

    Derivation sketch: similar triangles map a point at camera-space depth
    -z_e to the image plane at distance 'near' via focal scale f = cot(fovy/2).
    The third row is chosen so that z_camera = -near -> z_ndc = -1 and
    z_camera = -far -> z_ndc = +1, while packing 1/z_camera behavior into w
    (row 4) so the GPU's perspective divide produces correct nonlinear depth.
    """
    f = 1.0 / np.tan(fovy_rad / 2.0)
    M = np.zeros((4, 4))
    M[0, 0] = f / aspect
    M[1, 1] = f
    M[2, 2] = (far + near) / (near - far)
    M[2, 3] = (2 * far * near) / (near - far)
    M[3, 2] = -1.0
    return M


def orthographic(left, right, bottom, top, near, far):
    """Parallel projection — no perspective divide distortion, used for comparison."""
    M = np.eye(4)
    M[0, 0] = 2.0 / (right - left)
    M[1, 1] = 2.0 / (top - bottom)
    M[2, 2] = -2.0 / (far - near)
    M[0, 3] = -(right + left) / (right - left)
    M[1, 3] = -(top + bottom) / (top - bottom)
    M[2, 3] = -(far + near) / (far - near)
    return M


# ----------------------------------------------------------------------------
# 3. FULL PIPELINE: MODEL -> WORLD -> VIEW -> CLIP -> NDC -> SCREEN
# ----------------------------------------------------------------------------

W_EPSILON = 1e-6  # numerical guard against division by (near-)zero w


class Camera:
    def __init__(self, eye, target, up=(0, 1, 0), fovy_deg=45.0,
                 aspect=4 / 3, near=0.1, far=10.0, projection="perspective"):
        self.eye = np.asarray(eye, dtype=float)
        self.target = np.asarray(target, dtype=float)
        self.up = up
        self.fovy = np.radians(fovy_deg)
        self.aspect = aspect
        self.near = near
        self.far = far
        self.projection_kind = projection

    def view_matrix(self):
        return look_at(self.eye, self.target, self.up)

    def proj_matrix(self):
        if self.projection_kind == "perspective":
            return perspective(self.fovy, self.aspect, self.near, self.far)
        else:
            e = 1.0  # half-extent, tuned per scene
            return orthographic(-e, e, -e, e, self.near, self.far)

    def view_proj_matrix(self):
        return self.proj_matrix() @ self.view_matrix()

    def project_point(self, world_point):
        """
        World-space point -> (ndc_xyz, w_clip).
        The perspective divide (x,y,z)/w is the one genuinely nonlinear step
        in the whole pipeline, and the one place a numerical guard is required:
        if the point is exactly at the camera's eye plane w can approach 0 and
        the divide blows up. Points with w <= near-plane epsilon are expected
        to have already been clipped in a production pipeline (Section 4);
        here we clamp defensively so a single bad vertex cannot NaN the frame.
        """
        MVP = self.view_proj_matrix()
        clip = MVP @ np.array([*world_point, 1.0])
        w = clip[3]
        if abs(w) < W_EPSILON:
            w = W_EPSILON if w >= 0 else -W_EPSILON
        ndc = clip[:3] / w
        return ndc, clip[3]

    def project_points(self, world_points):
        """Vectorized projection of an (N,3) array of points -> (N,3) NDC, (N,) w."""
        MVP = self.view_proj_matrix()
        pts_h = np.hstack([world_points, np.ones((len(world_points), 1))])
        clip = (MVP @ pts_h.T).T          # (N,4)
        w = clip[:, 3].copy()
        near_mask = np.abs(w) < W_EPSILON
        w[near_mask] = np.where(w[near_mask] >= 0, W_EPSILON, -W_EPSILON)
        ndc = clip[:, :3] / w[:, None]
        return ndc, clip[:, 3]


def viewport_transform(ndc_xy, width, height):
    """NDC [-1,1]x[-1,1] -> pixel coordinates, with y flipped (image row 0 = top)."""
    x = (ndc_xy[..., 0] * 0.5 + 0.5) * width
    y = (1.0 - (ndc_xy[..., 1] * 0.5 + 0.5)) * height
    return np.stack([x, y], axis=-1)


# ----------------------------------------------------------------------------
# 4. NEAR-PLANE CLIPPING (system correctness, not just a numerical patch)
# ----------------------------------------------------------------------------

def clip_segment_to_near_plane(p0_clip, p1_clip, near=1e-4):
    """
    A segment in clip space is kept only where w > near (i.e. in front of the
    camera's near plane). If one endpoint is behind, it is intersected with
    the w = near plane via linear interpolation so the perspective divide is
    never asked to handle a negative or near-zero w for either endpoint.

    Returns (a, b) clip-space points, or None if the whole segment is behind.
    """
    w0, w1 = p0_clip[3], p1_clip[3]
    in0, in1 = w0 > near, w1 > near
    if not in0 and not in1:
        return None
    if in0 and in1:
        return p0_clip, p1_clip
    # interpolate to the boundary w = near
    t = (near - w0) / (w1 - w0)
    intersection = p0_clip + t * (p1_clip - p0_clip)
    if in0:
        return p0_clip, intersection
    else:
        return intersection, p1_clip
