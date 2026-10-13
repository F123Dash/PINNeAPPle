"""Gravitational lensing demos on the real sky: a Schwarzschild black hole placed in front of the stars you are
looking at, rendered with the geodesic ray tracer of :mod:`pinneapple_physics.blackhole.raytrace`, plus the closed
forms (weak-field deflection, Einstein radius, shadow diameter) checked against the Event Horizon Telescope.
"""
from __future__ import annotations

import math

import numpy as np

from . import frames
from .stars import blackbody_rgb, load_bright_stars, teff_from_bv

G = 6.67430e-11
C = 2.99792458e8
M_SUN = 1.98840987e30
PC = 3.0856775814913673e16
MUAS = 1e-6 / 3600 * math.pi / 180

# published EHT ring diameters (Event Horizon Telescope Collaboration 2019 ApJL 875 L1; 2022 ApJL 930 L12) and the
# mass / distance used to predict them (Gravity Collab. 2019 for Sgr A*; EHT 2019 VI for M87*)
EHT = {
    "M87*": {"mass_msun": 6.5e9, "distance_pc": 16.8e6, "ring_uas": 42.0, "ring_err_uas": 3.0},
    "Sgr A*": {"mass_msun": 4.0e6, "distance_pc": 8.15e3, "ring_uas": 51.8, "ring_err_uas": 2.3},
}


def deflection_angle_rad(mass_kg, impact_parameter_m):
    """Weak-field light deflection 4GM / (c^2 b)."""
    return 4 * G * mass_kg / (C**2 * np.asarray(impact_parameter_m, dtype=float))


def einstein_radius_rad(mass_kg, d_lens_m, d_source_m):
    d_ls = d_source_m - d_lens_m
    return np.sqrt(4 * G * mass_kg / C**2 * d_ls / (d_lens_m * d_source_m))


def shadow_diameter_rad(mass_kg, distance_m):
    """Angular diameter of a Schwarzschild shadow, 2 sqrt(27) GM / (c^2 D)."""
    return 2 * math.sqrt(27) * G * mass_kg / (C**2 * distance_m)


def eht_comparison() -> list[dict]:
    rows = []
    for name, d in EHT.items():
        pred = shadow_diameter_rad(d["mass_msun"] * M_SUN, d["distance_pc"] * PC) / MUAS
        rows.append({"object": name, "predicted_uas": pred, "observed_uas": d["ring_uas"],
                     "observed_err_uas": d["ring_err_uas"],
                     "sigma": abs(pred - d["ring_uas"]) / d["ring_err_uas"]})
    return rows


def sun_deflection_arcsec(elongation_deg):
    """Deflection of starlight by the Sun at an angular distance from its centre (Eddington 1919: 1.75" at the
    limb), using b = 1 AU sin(elongation)."""
    b = 1.495978707e11 * np.sin(np.radians(elongation_deg))
    return np.degrees(deflection_angle_rad(M_SUN, b)) * 3600


def catalog_sky(width: int = 4096, height: int = 2048, mag_limit: float = 6.5, milky_way: float = 0.12) -> np.ndarray:
    """Equirectangular RGB image (x = RA / 360 * width, Dec up) of the bright-star catalogue over the Milky Way
    glow: stars drawn as Gaussian splats with black-body colours and flux ~ 10^(-0.4 V)."""
    import os

    from PIL import Image
    cat = load_bright_stars()
    ok = cat.v <= mag_limit
    ra, dec, v = cat.ra[ok], cat.dec[ok], cat.v[ok]
    bv = np.nan_to_num(cat.bv[ok], nan=0.6)
    mw = Image.open(os.path.join(os.path.dirname(__file__), "data", "milkyway.jpg")).convert("L").resize((width, height))
    img = (np.asarray(mw, np.float32) / 255.0 * milky_way)[..., None] * np.array([0.78, 0.82, 1.0], np.float32)
    x = ra / 360 * width
    y = (90 - dec) / 180 * (height - 1)
    flux = 10 ** (-0.4 * (v - 1.5))
    col = blackbody_rgb(teff_from_bv(bv))
    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            xi = np.floor(x).astype(int) + dx
            yi = np.floor(y).astype(int) + dy
            w = np.exp(-((xi + 0.5 - x) ** 2 + (yi + 0.5 - y) ** 2) / 1.2)
            np.add.at(img, (np.clip(yi, 0, height - 1), xi % width), (flux * w)[:, None] * col)
    return np.clip(img, 0, 1)


def _sample(sky, ra, dec):
    """Bilinear lookup in an equirectangular image."""
    H, W = sky.shape[:2]
    x = (ra / 360.0 * W - 0.5) % W
    y = np.clip((90.0 - dec) / 180.0 * (H - 1), 0, H - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    x1 = (x0 + 1) % W
    return ((1 - fx) * (1 - fy) * sky[y0, x0] + fx * (1 - fy) * sky[y0, x1] + (1 - fx) * fy * sky[y0 + 1, x0]
            + fx * fy * sky[y0 + 1, x1])


_TRACES: dict = {}


def _trace(W, H, r_cam, fov_deg):
    """Geodesics depend only on the camera, not on where it points: trace once per (size, r_cam, fov)."""
    from ..blackhole.raytrace import Camera
    key = (W, H, round(r_cam, 3), round(fov_deg, 3))
    if key not in _TRACES:
        if len(_TRACES) > 8:
            _TRACES.pop(next(iter(_TRACES)))
        cam = Camera(width=W, height=H, r_cam=r_cam, inclination_deg=90.0, fov_deg=fov_deg)
        grid = {"r_faces": np.geomspace(2.05, 2.1, 3), "theta_faces": np.linspace(0, math.pi, 3)}
        _TRACES[key] = (cam, cam.trace(grid, r_max=2.1))
    return _TRACES[key]


def lensed_sky(ra_deg: float, dec_deg: float, fov_deg: float = 30.0, width: int = 480, height: int = 270,
               r_cam: float = 30.0, sky: np.ndarray | None = None, supersample: int = 2) -> dict:
    """Render the real sky around (ra, dec) as seen through a Schwarzschild black hole at that position.

    The camera sits at ``r_cam`` GM/c^2 from the hole; the geodesics come from ``Camera.trace`` of the black-hole
    module (pure NumPy RK4 on u'' = 3u^2 - u). Returns the image, the shadow mask and the shadow's angular radius."""
    W, H = width * supersample, height * supersample
    cam, paths = _trace(W, H, float(r_cam), float(fov_deg))
    _, fwd, right, up = cam.basis()
    target = frames.from_radec(ra_deg, dec_deg)
    north = np.array([0.0, 0.0, 1.0]) - target * target[2]
    north = north / np.linalg.norm(north) if np.linalg.norm(north) > 1e-9 else np.array([1.0, 0, 0])
    east = np.cross(north, target)
    # seen from inside the celestial sphere east is to the left, so camera right -> -east
    B = np.stack([fwd, right, up])
    S = np.stack([target, -east, north])
    M = S.T @ B
    d = paths.escape_dir @ M.T
    d[paths.captured] = target                       # placeholder; captured rays are painted black below
    ra, dec, _ = frames.radec(d)
    sky = catalog_sky() if sky is None else sky
    img = _sample(sky, ra, dec).reshape(H, W, 3)
    cap = paths.captured.reshape(H, W)
    img[cap] = 0.0
    if supersample > 1:
        img = img.reshape(height, supersample, width, supersample, 3).mean((1, 3))
        cap = cap.reshape(height, supersample, width, supersample).mean((1, 3)) > 0.5
    alpha = math.degrees(math.asin(min(1.0, math.sqrt(27) * math.sqrt(1 - 2 / r_cam) / r_cam)))
    return {"image": img, "captured": cap, "shadow_radius_deg": alpha, "fov_deg": fov_deg}


# ------------------------------------------------------------------------------------------- accretion disc view
def thin_disc(r_in: float = 6.0, r_out: float = 32.0, aspect: float = 0.03, n: int = 128):
    """A geometrically thin, optically thin emitting disc on a spherical (r, theta) grid (G = M = c = 1):
    inner edge at the innermost stable circular orbit (6 GM/c^2), emissivity profile with the zero-torque factor
    (1 - sqrt(r_in / r)) of the Shakura-Sunyaev / Novikov-Thorne disc, temperature ~ r^-3/4, Keplerian rotation in
    the Paczynski-Wiita potential (v_phi = sqrt(r) / (r - 2), relativistic beaming follows), Gaussian vertical
    profile of half-thickness ``aspect`` * r."""
    rf = np.geomspace(3.0, 120.0, n + 1)
    thf = np.linspace(0.0, math.pi, n + 1)
    r = np.sqrt(rf[:-1] * rf[1:])[:, None]
    th = (0.5 * (thf[:-1] + thf[1:]))[None, :]
    z = np.cos(th)
    zero_torque = np.clip(1 - np.sqrt(r_in / r), 0, None) * (r >= r_in) * (r <= r_out)
    taper = np.exp(-((r / r_out) ** 6))
    rho = zero_torque * taper * r ** -1.5 * np.exp(-0.5 * (z / aspect) ** 2) + 1e-12
    T = (r / r_in) ** -0.75 * np.ones_like(th)
    vphi = np.minimum(np.sqrt(r) / (r - 2.0), 0.75) * np.ones_like(th)
    vel = np.stack([np.zeros_like(rho), np.zeros_like(rho), vphi])
    return {"r_faces": rf, "theta_faces": thf}, rho, T, vel


_DISC_CACHE: dict = {}


def accretion_view(ra_deg: float, dec_deg: float, width: int = 400, height: int = 225, inclination_deg: float = 80.0,
                   r_cam: float = 80.0, fov_deg: float = 34.0, doppler: bool = True, sky: np.ndarray | None = None):
    """A Schwarzschild black hole with a thin accretion disc placed in front of the real sky around (ra, dec):
    null geodesics through the disc (``pinneapple_physics.blackhole.raytrace``), Doppler beaming and gravitational
    redshift, the lensed background from the bright-star catalogue and the Milky Way, then bloom.

    The disc is optically thin and static, so its image depends only on the camera: it is traced and rendered once
    (~1 min) and cached; each new direction only re-samples the lensed sky along the stored escape directions."""
    from ..blackhole.raytrace import Camera, bloom, render
    key = (width, height, round(inclination_deg, 2), round(r_cam, 2), round(fov_deg, 2), bool(doppler))
    if key not in _DISC_CACHE:
        if len(_DISC_CACHE) > 4:
            _DISC_CACHE.pop(next(iter(_DISC_CACHE)))
        grid, rho, T, vel = thin_disc()
        cam = Camera(width=width, height=height, r_cam=r_cam, inclination_deg=inclination_deg, fov_deg=fov_deg,
                     ds_in=0.8, ds_out=6.0)
        paths = cam.trace(grid, r_max=40.0)
        disc = render(paths, rho, T, vel, doppler=doppler, sky=None, t_color=3800, gamma=0.75, exposure=2.6,
                      emissivity_power=1)
        _DISC_CACHE[key] = (cam, paths, disc)
    cam, paths, disc = _DISC_CACHE[key]
    _, fwd, right, up = cam.basis()
    target = frames.from_radec(ra_deg, dec_deg)
    north = np.array([0.0, 0.0, 1.0]) - target * target[2]
    north = north / np.linalg.norm(north) if np.linalg.norm(north) > 1e-9 else np.array([1.0, 0.0, 0.0])
    east = np.cross(north, target)
    M = np.stack([target, -east, north]).T @ np.stack([fwd, right, up])          # BH frame -> ICRS
    d = paths.escape_dir @ M.T
    d[paths.captured] = target
    ra, dec, _ = frames.radec(d)
    sky = catalog_sky(2048, 1024, milky_way=0.10) if sky is None else sky
    layer = _sample(sky, ra, dec).reshape(height, width, 3) * 0.55
    layer[paths.captured.reshape(height, width)] = 0.0
    img = 1 - (1 - np.clip(disc, 0, 1)) * (1 - np.clip(layer, 0, 1))             # screen blend
    img = bloom(img, 0.6, 5)
    alpha = math.degrees(math.asin(min(1.0, math.sqrt(27) * math.sqrt(1 - 2 / r_cam) / r_cam)))
    return {"image": img, "shadow_radius_deg": alpha, "fov_deg": fov_deg, "inclination_deg": inclination_deg}
