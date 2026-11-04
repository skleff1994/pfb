"""Normal pressure fields from contact mechanics, and a standard population of them.

Each constructor returns a `Measure` built with `from_pressure`, rescaled to the total force fN.
The density is evaluated on the patch grid only, so the part of a contact that lies outside the
pad is clipped: placing a contact across the pad edge models a partial contact.

    hertz         sphere on the pad, p ~ sqrt(1 - r^2/a^2)
    xydas_kao     soft-finger family p ~ (1 - (r/a)^k)^(1/k) (Xydas-Kao 1999); k = 2 is Hertz, k -> inf a uniform disk
    flat_punch    rigid flat punch, p ~ (1 - r^2/a^2)^(-1/2): load concentrated on the rim
    ramp          linear pressure ramp over a disk, a simple model of a tilted flat contact
    line_contact  cylinder on the pad (2D Hertz strip), p ~ sqrt(1 - d^2/b^2)
    asperities    sum of Hertz contacts with given forces

`population(patch)` draws a fixed set of these fields (7 families) for the numerical studies.
All lengths are in the units of the patch.
"""

from dataclasses import dataclass

import numpy as np

from .measure import Measure, from_pressure


def _radius2(x, y, center):
    """Squared distance from (x, y) (arrays) to center."""
    return (x - center[0]) ** 2 + (y - center[1]) ** 2


def hertz(patch, center, radius, fN=1.0):
    """Hertz contact of radius a: p ~ sqrt(1 - r^2/a^2) for r < a."""
    return from_pressure(patch, lambda x, y: np.sqrt(np.clip(1 - _radius2(x, y, center) / radius**2, 0, None)), fN)


def xydas_kao(patch, center, radius, k, fN=1.0):
    """Xydas-Kao soft-finger pressure p ~ (1 - (r/a)^k)^(1/k) for r < a (k = 2: Hertz, k -> inf: uniform)."""
    def rho(x, y):
        """Unnormalized pressure at (x, y)."""
        s = np.sqrt(_radius2(x, y, center)) / radius
        return np.where(s < 1, np.clip(1 - s**k, 0, None) ** (1 / k), 0.0)
    return from_pressure(patch, rho, fN)


def flat_punch(patch, center, radius, fN=1.0):
    """Rigid flat punch: p ~ (1 - r^2/a^2)^(-1/2) for r < a, concentrated on the rim.

    The integrable singularity at r = a is capped at about one grid spacing from the rim,
    i.e. at 1 - r^2/a^2 = 2h/a, where h is the grid spacing.
    """
    s_min = 2 * max(patch.h) / radius

    def rho(x, y):
        """Unnormalized pressure at (x, y), capped near the rim."""
        s = 1 - _radius2(x, y, center) / radius**2
        return np.where(s > 0, 1 / np.sqrt(np.maximum(s, s_min)), 0.0)
    return from_pressure(patch, rho, fN)


def ramp(patch, center, radius, tilt, angle, fN=1.0):
    """Linear ramp over a disk: p ~ max(0, 1 + tilt <x - c, u> / a) for r < a, with u = (cos angle, sin angle).

    A simple model of a slightly tilted flat contact (not the exact elastic solution). The pressure
    grows along u; for tilt > 1 part of the disk lifts off.
    """
    u = np.array([np.cos(angle), np.sin(angle)])

    def rho(x, y):
        """Unnormalized pressure at (x, y)."""
        lin = 1 + tilt * ((x - center[0]) * u[0] + (y - center[1]) * u[1]) / radius
        return np.where(_radius2(x, y, center) < radius**2, np.clip(lin, 0, None), 0.0)
    return from_pressure(patch, rho, fN)


def line_contact(patch, point, angle, half_width, fN=1.0):
    """Cylinder on the pad: p ~ sqrt(1 - d^2/b^2), d the distance to the line through `point` along `angle`."""
    n = np.array([-np.sin(angle), np.cos(angle)])  # normal to the line

    def rho(x, y):
        """Unnormalized pressure at (x, y)."""
        d = (x - point[0]) * n[0] + (y - point[1]) * n[1]
        return np.sqrt(np.clip(1 - d**2 / half_width**2, 0, None))
    return from_pressure(patch, rho, fN)


def asperities(patch, centers, radii, forces):
    """Sum of Hertz contacts: forces[j] carried by a Hertz contact of radius radii[j] at centers[j]."""
    w = sum(hertz(patch, c, a, f).w for c, a, f in zip(centers, radii, forces))
    return Measure(patch, w)


# --- population -------------------------------------------------------------------------------

@dataclass
class Field:
    """A named pressure field: `family` is one of FAMILIES, `label` describes the parameters."""
    family: str
    label: str
    measure: Measure


FAMILIES = ("hertz", "xydas_kao", "flat_punch", "ramp", "partial", "line", "asperities")


def population(patch, n_per_family=3, seed=0, fN=1.0):
    """n_per_family random fields of each family in FAMILIES, deterministic for a given seed.

    Radii are drawn as fractions of the shorter pad side L. All contacts lie inside the pad,
    except the "partial" family (a Hertz contact or a ramp whose centre is closer to a pad edge
    than its radius, so that the edge clips it). Returns a list of `Field`.
    """
    rng = np.random.default_rng(seed)
    L = min(patch.Lx, patch.Ly)
    size = np.array([patch.Lx, patch.Ly])

    def inside(a):
        """Centre such that the disk of radius a lies inside the pad."""
        return a + rng.uniform(0, 1, 2) * (size - 2 * a)

    def fmt(c):
        """Point as '(x, y)' with two decimals, for the field labels."""
        return f"({c[0]:.2f}, {c[1]:.2f})"

    fields = []
    for i in range(n_per_family):
        a = rng.uniform(0.2, 0.4) * L
        c = inside(a)
        fields.append(Field("hertz", f"Hertz a={a:.2f} at {fmt(c)}", hertz(patch, c, a, fN)))

        a, k = rng.uniform(0.2, 0.4) * L, (3, 4, 8)[i % 3]
        c = inside(a)
        fields.append(Field("xydas_kao", f"Xydas-Kao k={k} a={a:.2f} at {fmt(c)}", xydas_kao(patch, c, a, k, fN)))

        a = rng.uniform(0.2, 0.4) * L
        c = inside(a)
        fields.append(Field("flat_punch", f"flat punch a={a:.2f} at {fmt(c)}", flat_punch(patch, c, a, fN)))

        a, t, th = rng.uniform(0.25, 0.45) * L, rng.uniform(0.5, 1.5), rng.uniform(0, 2 * np.pi)
        c = inside(a)
        fields.append(Field("ramp", f"ramp a={a:.2f} tilt={t:.2f} at {fmt(c)}", ramp(patch, c, a, t, th, fN)))

        # Partial contact: centre at distance in [0.1 a, 0.6 a] from a random pad edge.
        a = rng.uniform(0.3, 0.45) * L
        c = inside(a)
        axis, side = rng.integers(2), rng.integers(2)
        d = rng.uniform(0.1, 0.6) * a
        c[axis] = size[axis] - d if side else d
        if i % 2 == 0:
            nu, kind = hertz(patch, c, a, fN), "Hertz"
        else:
            nu, kind = ramp(patch, c, a, rng.uniform(0.5, 1.5), rng.uniform(0, 2 * np.pi), fN), "ramp"
        fields.append(Field("partial", f"partial {kind} a={a:.2f} at {fmt(c)}", nu))

        p, th, b = rng.uniform(0.3, 0.7, 2) * size, rng.uniform(0, np.pi), rng.uniform(0.08, 0.2) * L
        fields.append(Field("line", f"line b={b:.2f} angle={np.degrees(th):.0f} deg", line_contact(patch, p, th, b, fN)))

        radii = rng.uniform(0.08, 0.18, 2) * L
        centers = [inside(r) for r in radii]
        share = rng.uniform(0.3, 0.7)
        fields.append(Field("asperities", f"2 asperities, split {share:.2f}/{1 - share:.2f}",
                            asperities(patch, centers, radii, [share * fN, (1 - share) * fN])))
    return fields
