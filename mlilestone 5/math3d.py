"""
math3d.py
---------
Custom mathematical foundation for the Digital Human system.

Everything here is implemented from first principles using plain Python
floats/lists -- no numpy, no external linear-algebra library -- so the
representation of geometry and the computation on it are fully explicit
and auditable. This is the Milestone 1 mathematical core, reused
unchanged by the Milestone 2 camera/projection pipeline and the
Milestone 5 animation/dynamics layer.
"""

import math


# ---------------------------------------------------------------------------
# Vec3
# ---------------------------------------------------------------------------
class Vec3:
    __slots__ = ("x", "y", "z")

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def __add__(self, o):
        return Vec3(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o):
        return Vec3(self.x - o.x, self.y - o.y, self.z - o.z)

    def __mul__(self, s):
        return Vec3(self.x * s, self.y * s, self.z * s)

    __rmul__ = __mul__

    def __neg__(self):
        return Vec3(-self.x, -self.y, -self.z)

    def dot(self, o):
        return self.x * o.x + self.y * o.y + self.z * o.z

    def cross(self, o):
        return Vec3(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )

    def length(self):
        return math.sqrt(self.dot(self))

    def normalize(self):
        l = self.length()
        if l < 1e-12:
            return Vec3(0.0, 0.0, 0.0)
        return Vec3(self.x / l, self.y / l, self.z / l)

    def to_tuple(self):
        return (self.x, self.y, self.z)

    def lerp(self, o, t):
        return self * (1.0 - t) + o * t

    def __repr__(self):
        return f"Vec3({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"


# ---------------------------------------------------------------------------
# Mat4 -- stored row-major as a flat list of 16 floats, m[row*4 + col]
# Convention: column vectors, v' = M * v  (matches the derivation doc)
# ---------------------------------------------------------------------------
class Mat4:
    __slots__ = ("m",)

    def __init__(self, m=None):
        self.m = m if m is not None else [0.0] * 16

    @staticmethod
    def identity():
        m = [0.0] * 16
        for i in range(4):
            m[i * 4 + i] = 1.0
        return Mat4(m)

    def get(self, r, c):
        return self.m[r * 4 + c]

    def set(self, r, c, v):
        self.m[r * 4 + c] = v

    def __mul__(self, other):
        if isinstance(other, Mat4):
            a, b = self.m, other.m
            out = [0.0] * 16
            for r in range(4):
                for c in range(4):
                    s = 0.0
                    for k in range(4):
                        s += a[r * 4 + k] * b[k * 4 + c]
                    out[r * 4 + c] = s
            return Mat4(out)
        elif isinstance(other, Vec4):
            m = self.m
            x, y, z, w = other.x, other.y, other.z, other.w
            return Vec4(
                m[0] * x + m[1] * y + m[2] * z + m[3] * w,
                m[4] * x + m[5] * y + m[6] * z + m[7] * w,
                m[8] * x + m[9] * y + m[10] * z + m[11] * w,
                m[12] * x + m[13] * y + m[14] * z + m[15] * w,
            )
        raise TypeError(f"Cannot multiply Mat4 by {type(other)}")

    def transform_point(self, v: Vec3) -> Vec3:
        r = self * Vec4(v.x, v.y, v.z, 1.0)
        if abs(r.w) > 1e-12 and abs(r.w - 1.0) > 1e-9:
            return Vec3(r.x / r.w, r.y / r.w, r.z / r.w)
        return Vec3(r.x, r.y, r.z)

    def transform_vector(self, v: Vec3) -> Vec3:
        r = self * Vec4(v.x, v.y, v.z, 0.0)
        return Vec3(r.x, r.y, r.z)

    def transpose(self):
        m = self.m
        out = [0.0] * 16
        for r in range(4):
            for c in range(4):
                out[c * 4 + r] = m[r * 4 + c]
        return Mat4(out)

    def inverse_affine(self):
        """Closed-form inverse for an affine transform M = [R t; 0 1]
        via cofactor expansion of the 3x3 block (see derivation.md)."""
        m = self.m
        t = Vec3(m[3], m[7], m[11])
        a, b, c = m[0], m[1], m[2]
        d, e, f = m[4], m[5], m[6]
        g, h, i = m[8], m[9], m[10]
        A = e * i - f * h
        B = -(d * i - f * g)
        C = d * h - e * g
        det = a * A + b * B + c * C
        if abs(det) < 1e-12:
            raise ValueError("Matrix not invertible (det ~ 0)")
        invdet = 1.0 / det
        D = -(b * i - c * h)
        E = a * i - c * g
        F = -(a * h - b * g)
        G = b * f - c * e
        H = -(a * f - c * d)
        I = a * e - b * d
        Rinv = [
            [A * invdet, D * invdet, G * invdet],
            [B * invdet, E * invdet, H * invdet],
            [C * invdet, F * invdet, I * invdet],
        ]
        tix = -(Rinv[0][0] * t.x + Rinv[0][1] * t.y + Rinv[0][2] * t.z)
        tiy = -(Rinv[1][0] * t.x + Rinv[1][1] * t.y + Rinv[1][2] * t.z)
        tiz = -(Rinv[2][0] * t.x + Rinv[2][1] * t.y + Rinv[2][2] * t.z)
        out = [
            Rinv[0][0], Rinv[0][1], Rinv[0][2], tix,
            Rinv[1][0], Rinv[1][1], Rinv[1][2], tiy,
            Rinv[2][0], Rinv[2][1], Rinv[2][2], tiz,
            0.0, 0.0, 0.0, 1.0,
        ]
        return Mat4(out)

    @staticmethod
    def translation(t: Vec3):
        m = Mat4.identity()
        m.set(0, 3, t.x)
        m.set(1, 3, t.y)
        m.set(2, 3, t.z)
        return m

    @staticmethod
    def scale(s):
        sx, sy, sz = (s, s, s) if isinstance(s, (int, float)) else (s.x, s.y, s.z)
        m = Mat4.identity()
        m.set(0, 0, sx)
        m.set(1, 1, sy)
        m.set(2, 2, sz)
        return m

    @staticmethod
    def rotation_axis_angle(axis: Vec3, angle_rad: float):
        a = axis.normalize()
        x, y, z = a.x, a.y, a.z
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        t = 1.0 - c
        m = [
            t * x * x + c,     t * x * y - s * z, t * x * z + s * y, 0.0,
            t * x * y + s * z, t * y * y + c,     t * y * z - s * x, 0.0,
            t * x * z - s * y, t * y * z + s * x, t * z * z + c,     0.0,
            0.0, 0.0, 0.0, 1.0,
        ]
        return Mat4(m)

    @staticmethod
    def look_at(eye: Vec3, target: Vec3, up: Vec3):
        f = (target - eye).normalize()
        s = f.cross(up).normalize()
        u = s.cross(f)
        m = [
            s.x,  s.y,  s.z,  -s.dot(eye),
            u.x,  u.y,  u.z,  -u.dot(eye),
            -f.x, -f.y, -f.z,  f.dot(eye),
            0.0,  0.0,  0.0,  1.0,
        ]
        return Mat4(m)

    @staticmethod
    def perspective(fovy_rad, aspect, near, far):
        f = 1.0 / math.tan(fovy_rad / 2.0)
        nf = 1.0 / (near - far)
        m = [
            f / aspect, 0.0, 0.0, 0.0,
            0.0, f, 0.0, 0.0,
            0.0, 0.0, (far + near) * nf, 2.0 * far * near * nf,
            0.0, 0.0, -1.0, 0.0,
        ]
        return Mat4(m)

    @staticmethod
    def orthographic(left, right, bottom, top, near, far):
        m = Mat4.identity()
        m.set(0, 0, 2.0 / (right - left))
        m.set(1, 1, 2.0 / (top - bottom))
        m.set(2, 2, -2.0 / (far - near))
        m.set(0, 3, -(right + left) / (right - left))
        m.set(1, 3, -(top + bottom) / (top - bottom))
        m.set(2, 3, -(far + near) / (far - near))
        return m

    def __repr__(self):
        rows = [self.m[i * 4:(i + 1) * 4] for i in range(4)]
        return "Mat4(\n" + "\n".join(str(r) for r in rows) + "\n)"


class Vec4:
    __slots__ = ("x", "y", "z", "w")

    def __init__(self, x, y, z, w):
        self.x, self.y, self.z, self.w = x, y, z, w

    def __repr__(self):
        return f"Vec4({self.x:.4f}, {self.y:.4f}, {self.z:.4f}, {self.w:.4f})"


# ---------------------------------------------------------------------------
# Quaternion
# ---------------------------------------------------------------------------
class Quaternion:
    __slots__ = ("w", "x", "y", "z")

    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        self.w, self.x, self.y, self.z = w, x, y, z

    @staticmethod
    def identity():
        return Quaternion(1.0, 0.0, 0.0, 0.0)

    @staticmethod
    def from_axis_angle(axis: Vec3, angle_rad: float):
        a = axis.normalize()
        half = angle_rad * 0.5
        s = math.sin(half)
        return Quaternion(math.cos(half), a.x * s, a.y * s, a.z * s)

    def norm(self):
        return math.sqrt(self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z)

    def normalize(self):
        n = self.norm()
        if n < 1e-12:
            return Quaternion.identity()
        return Quaternion(self.w / n, self.x / n, self.y / n, self.z / n)

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def __mul__(self, o):
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = o.w, o.x, o.y, o.z
        return Quaternion(
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )

    def rotate(self, v: Vec3) -> Vec3:
        qv = Quaternion(0.0, v.x, v.y, v.z)
        r = self * qv * self.conjugate()
        return Vec3(r.x, r.y, r.z)

    def to_mat4(self):
        q = self.normalize()
        w, x, y, z = q.w, q.x, q.y, q.z
        m = [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),     0.0,
            2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),     0.0,
            2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y), 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]
        return Mat4(m)

    @staticmethod
    def slerp(q0, q1, t):
        q0 = q0.normalize()
        q1 = q1.normalize()
        dot = q0.w * q1.w + q0.x * q1.x + q0.y * q1.y + q0.z * q1.z
        if dot < 0.0:
            q1 = Quaternion(-q1.w, -q1.x, -q1.y, -q1.z)
            dot = -dot
        if dot > 0.9995:
            w = q0.w + t * (q1.w - q0.w)
            x = q0.x + t * (q1.x - q0.x)
            y = q0.y + t * (q1.y - q0.y)
            z = q0.z + t * (q1.z - q0.z)
            return Quaternion(w, x, y, z).normalize()
        theta_0 = math.acos(max(-1.0, min(1.0, dot)))
        theta = theta_0 * t
        sin_theta_0 = math.sin(theta_0)
        s0 = math.cos(theta) - dot * math.sin(theta) / sin_theta_0
        s1 = math.sin(theta) / sin_theta_0
        return Quaternion(
            s0 * q0.w + s1 * q1.w,
            s0 * q0.x + s1 * q1.x,
            s0 * q0.y + s1 * q1.y,
            s0 * q0.z + s1 * q1.z,
        )

    def __repr__(self):
        return f"Quat(w={self.w:.4f}, x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f})"
