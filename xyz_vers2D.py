# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 15:39:52 2026

@author: bara.fall
"""

# igor_xyz_to_iq_no_gui.py
# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
xyz_to_3d_cld_and_iq_no_gui_full.py
CLD 3D depuis XYZ (Igor) + Modélisation I(q) Matlab-like
+ Export volume binaire 3D + affichage 3D
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from skimage import io as skio  # pour exporter stack TIFF


# =============================================================================
# PARAMÈTRES (à modifier)
# =============================================================================
XYZ_FILE = r"C:\Users\bara.fall\Downloads\positions_example.csv"   # csv: x,y,z  / txt: x y z
FMT = "csv"          # "csv" ou "txt"
SKIPROWS = 1         # 0 si pas d'entête

OUT_DIR = Path(r"C:\Users\bara.fall\Downloads\sortie_xyz_cld")
CASE_NAME = "Igor_mass_fractal"

# Unités : si XYZ est en Å, laisse SCALE=1 et q en Å^-1
SCALE = 1.0

# Paramètre Igor : Primary Rg (même unité que XYZ avant SCALE)
PRIMARY_RG = 10.0

# Voxelisation (résolution 3D) : plus petit = plus précis mais plus lourd
VOXEL_SIZE = 0.3     # (même unité que XYZ*SCALE) ex: 1 Å, 0.5 nm, ...

# Marge autour du nuage de points (en unités)
PADDING = 5.0

# CLD (3D)
N_LINES = 20000      # attention: 3D coûte plus cher
PX_MIN = 1
BINWIDTH_PX = 1
XMAX_R = 150
PLOT_LOGY = True
SEED = 0

# Pas d'échantillonnage le long des droites (en voxel)
# 1.0 = un point par voxel; 0.5 = plus fin (plus cher)
LINE_STEP_VOX = 1.0

# Modélisation I(q) Matlab-like
K0 = 1

# Exports volume binaire 3D
EXPORT_VOLUME_NPY = True
EXPORT_VOLUME_TIFF = True
EXPORT_SLICES = True
EXPORT_MIP = True

# Affichage 3D interactif (peut être lourd) :
SHOW_3D = False
SHOW_3D_MAX_VOXELS = 2_000_000  # sous-échantillonnage auto si volume trop gros


# =============================================================================
# Lecture XYZ
# =============================================================================
def read_xyz(path: str, fmt: str = "csv", skiprows: int = 0) -> np.ndarray:
    delimiter = "," if fmt.lower() == "csv" else None
    data = np.loadtxt(path, delimiter=delimiter, skiprows=int(skiprows))
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 3:
        raise ValueError("Fichier invalide: il faut au moins 3 colonnes (x,y,z).")
    return data[:, :3].astype(float)


# =============================================================================
# Igor Primary Rg -> rayon de sphère
# Rg = sqrt(3/5) R  =>  R = sqrt(5/3) Rg
# =============================================================================
def sphere_radius_from_rg(Rg_primary: float) -> float:
    return float(np.sqrt(5.0 / 3.0) * float(Rg_primary))


# =============================================================================
# Voxelisation 3D: sphères autour des centres
# =============================================================================
def voxelize_spheres(X: np.ndarray, R: float, voxel: float, padding: float):
    """
    Construit un volume binaire 3D:
      True  = phase blanche (matière = sphères)
      False = phase noire (vide)
    Retour:
      vol (bool), origin (x0,y0,z0), voxel
    """
    X = np.asarray(X, float)
    voxel = float(voxel)
    R = float(R)
    pad = float(padding)

    mn = X.min(axis=0) - (R + pad)
    mx = X.max(axis=0) + (R + pad)

    dims = np.ceil((mx - mn) / voxel).astype(int) + 1  # (nx, ny, nz)
    nx, ny, nz = dims.tolist()

    vol = np.zeros((nx, ny, nz), dtype=bool)

    r_vox = int(np.ceil(R / voxel))

    gx = np.arange(-r_vox, r_vox + 1)
    gy = np.arange(-r_vox, r_vox + 1)
    gz = np.arange(-r_vox, r_vox + 1)
    XX, YY, ZZ = np.meshgrid(gx, gy, gz, indexing="ij")
    mask_sphere = (XX**2 + YY**2 + ZZ**2) * (voxel**2) <= (R**2)

    offsets = np.column_stack([XX[mask_sphere], YY[mask_sphere], ZZ[mask_sphere]])

    for p in X:
        idx = np.round((p - mn) / voxel).astype(int)

        coords = offsets + idx[None, :]
        m = (
            (coords[:, 0] >= 0) & (coords[:, 0] < nx) &
            (coords[:, 1] >= 0) & (coords[:, 1] < ny) &
            (coords[:, 2] >= 0) & (coords[:, 2] < nz)
        )
        c = coords[m]
        vol[c[:, 0], c[:, 1], c[:, 2]] = True

    return vol, mn, voxel


# =============================================================================
# Export volume 3D + slices + MIP
# =============================================================================
def save_volume_npy(out_path: Path, vol_bool: np.ndarray):
    np.save(out_path, vol_bool.astype(np.uint8))

def save_volume_tiff_stack(out_path: Path, vol_bool: np.ndarray):
    """
    Sauvegarde un stack TIFF Z-slices ouvrable dans Fiji/ImageJ.
    Fiji attend souvent (Z,Y,X), on transpose.
    """
    stack = (vol_bool.astype(np.uint8) * 255)
    stack_zyx = np.transpose(stack, (2, 1, 0))  # (nz, ny, nx)
    skio.imsave(str(out_path), stack_zyx)

def _save_gray_png(path: Path, img2d: np.ndarray):
    plt.figure(figsize=(5, 5))
    plt.imshow(img2d.T, cmap="gray", origin="lower")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def save_middle_slices(out_dir: Path, vol_bool: np.ndarray, prefix: str):
    nx, ny, nz = vol_bool.shape
    ix, iy, iz = nx // 2, ny // 2, nz // 2

    xy = vol_bool[:, :, iz]
    xz = vol_bool[:, iy, :]
    yz = vol_bool[ix, :, :]

    _save_gray_png(out_dir / f"{prefix}_slice_XY_z{iz}.png", xy)
    _save_gray_png(out_dir / f"{prefix}_slice_XZ_y{iy}.png", xz)
    _save_gray_png(out_dir / f"{prefix}_slice_YZ_x{ix}.png", yz)

def save_mip(out_dir: Path, vol_bool: np.ndarray, prefix: str):
    mip_xy = np.max(vol_bool, axis=2)  # proj z
    mip_xz = np.max(vol_bool, axis=1)  # proj y
    mip_yz = np.max(vol_bool, axis=0)  # proj x

    _save_gray_png(out_dir / f"{prefix}_MIP_XY.png", mip_xy)
    _save_gray_png(out_dir / f"{prefix}_MIP_XZ.png", mip_xz)
    _save_gray_png(out_dir / f"{prefix}_MIP_YZ.png", mip_yz)

def show_voxel_3d(vol_bool: np.ndarray, max_voxels: int = 2_000_000):
    """
    Affichage 3D avec matplotlib.voxels.
    Sous-échantillonne automatiquement si nécessaire.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    nx, ny, nz = vol_bool.shape
    step = 1
    while (nx // step) * (ny // step) * (nz // step) > max_voxels:
        step *= 2

    v = vol_bool[::step, ::step, ::step]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(v, edgecolor=None)
    ax.set_title(f"Volume binaire 3D (downsample x{step})")
    plt.tight_layout()
    plt.show()


# =============================================================================
# Tirage de droites isotropes 3D + intersection avec boîte
# =============================================================================
def random_unit_vector(rng):
    u = rng.uniform(0.0, 1.0)
    v = rng.uniform(0.0, 1.0)
    theta = 2 * np.pi * u
    z = 2 * v - 1
    r = np.sqrt(max(0.0, 1.0 - z*z))
    return np.array([r*np.cos(theta), r*np.sin(theta), z], dtype=float)

def intersect_line_aabb(p0, d, bounds_min, bounds_max):
    tmin = -np.inf
    tmax = np.inf
    for i in range(3):
        if abs(d[i]) < 1e-12:
            if p0[i] < bounds_min[i] or p0[i] > bounds_max[i]:
                return None
        else:
            t1 = (bounds_min[i] - p0[i]) / d[i]
            t2 = (bounds_max[i] - p0[i]) / d[i]
            ta, tb = (t1, t2) if t1 <= t2 else (t2, t1)
            tmin = max(tmin, ta)
            tmax = min(tmax, tb)
            if tmin > tmax:
                return None
    return tmin, tmax

def chords_1d(arr_bool, px_min):
    a = np.asarray(arr_bool).astype(bool).astype(np.int8)
    da = np.diff(np.r_[0, a, 0])
    starts = np.where(da == 1)[0]
    ends = np.where(da == -1)[0]
    if starts.size == 0 or ends.size == 0:
        return np.array([], dtype=int)
    n = min(starts.size, ends.size)
    L = ends[:n] - starts[:n]
    return L[L >= px_min]

def sample_profile_along_line(vol, pA, pB, step_vox=1.0):
    v = pB - pA
    L = float(np.linalg.norm(v))
    if L <= 1e-12:
        return np.array([], dtype=bool)

    step = float(step_vox)
    n = max(2, int(np.floor(L / step)) + 1)
    ts = np.linspace(0.0, 1.0, n)
    pts = pA[None, :] + ts[:, None] * v[None, :]

    idx = np.rint(pts).astype(int)
    nx, ny, nz = vol.shape
    m = (
        (idx[:, 0] >= 0) & (idx[:, 0] < nx) &
        (idx[:, 1] >= 0) & (idx[:, 1] < ny) &
        (idx[:, 2] >= 0) & (idx[:, 2] < nz)
    )
    idx = idx[m]
    if idx.size == 0:
        return np.array([], dtype=bool)

    return vol[idx[:, 0], idx[:, 1], idx[:, 2]]

def cld_3d_from_volume(vol, n_lines, px_min, seed=0, step_vox=1.0):
    rng = np.random.default_rng(seed)
    nx, ny, nz = vol.shape
    bounds_min = np.array([0.0, 0.0, 0.0])
    bounds_max = np.array([nx - 1.0, ny - 1.0, nz - 1.0])

    chords = []
    for _ in range(int(n_lines)):
        d = random_unit_vector(rng)
        p0 = np.array([
            rng.uniform(0, nx - 1),
            rng.uniform(0, ny - 1),
            rng.uniform(0, nz - 1)
        ], dtype=float)

        hit = intersect_line_aabb(p0, d, bounds_min, bounds_max)
        if hit is None:
            continue
        tmin, tmax = hit
        if not np.isfinite(tmin) or not np.isfinite(tmax) or tmax <= tmin:
            continue

        pA = p0 + tmin * d
        pB = p0 + tmax * d

        prof = sample_profile_along_line(vol, pA, pB, step_vox=step_vox)
        if prof.size < 2:
            continue

        cl = chords_1d(prof, px_min=px_min)
        if cl.size:
            chords.append(cl)

    return np.concatenate(chords).astype(float) if chords else np.array([], dtype=float)


# =============================================================================
# Histogramme Matlab-like probabilité
# =============================================================================
def histogramme_matlab_probabilite(lens, binwidth=1):
    lens = np.asarray(lens, dtype=float)
    if lens.size == 0:
        return np.array([]), np.array([])
    maxv = int(np.max(lens))
    if maxv < 1:
        return np.array([]), np.array([])
    edges = np.arange(0.5, maxv + binwidth + 0.5, binwidth)
    counts, _ = np.histogram(lens, bins=edges)
    s = counts.sum()
    f = counts / s if s > 0 else counts.astype(float)
    a = np.arange(1, len(f) + 1, dtype=float)
    if binwidth == 1:
        r = a + 1.5
    elif binwidth == 2:
        r = a * 2 - 1
    else:
        r = (a * binwidth + (a - 1) * binwidth) / 2.0
    return r, f


# =============================================================================
# Modélisation I(q) Matlab-like (la même que ton code)
# =============================================================================
def modeliser_Iq_matlab(fm, fv, SurfSpe, eps_denom=1e-12, k0=1):
    fm = np.asarray(fm, dtype=float)
    fv = np.asarray(fv, dtype=float)
    if fm.size == 0 or fv.size == 0:
        return np.array([]), np.array([])

    L = max(fm.size, fv.size)
    if fm.size < L: fm = np.pad(fm, (0, L - fm.size))
    if fv.size < L: fv = np.pad(fv, (0, L - fv.size))

    ftm = np.fft.fft(fm)
    ftv = np.fft.fft(fv)

    half = L // 2 + 1
    ftm = ftm[:half]
    ftv = ftv[:half]

    denom = (1 - ftm * ftv) + eps_denom
    parent = (1 - ftm) * (1 - ftv) / denom
    a_parent = np.real(parent)

    k0 = max(1, int(k0))
    if k0 >= half:
        return np.array([]), np.array([])

    s = np.arange(k0, half + 1, dtype=float) / (float(L) / 2)
    a_parent = a_parent[k0 - 1:]

    q = 2 * np.pi * s
    parent_bis = a_parent / (s**2)
    derv = np.diff(parent_bis) / np.diff(s)

    I = (-SurfSpe / (16 * s[:-1] * (np.pi**3))) * derv
    return q[:-1], I


# =============================================================================
# Phi et Sv (même formule que ton CLD 2D)
# =============================================================================
def phi_sv_from_mean_chords(L_noire, L_blanche):
    if not (np.isfinite(L_noire) and np.isfinite(L_blanche)) or L_noire <= 0 or L_blanche <= 0:
        return np.nan, np.nan
    phi = L_noire / (L_noire + L_blanche)
    Sv = (4 * phi * (1 - phi)) * (1 / L_blanche + 1 / L_noire)
    return phi, Sv


# =============================================================================
# Export helpers
# =============================================================================
def save_csv_xy(path: Path, x, y, header):
    data = np.column_stack([x, y])
    np.savetxt(path, data, delimiter=";", header=header, comments="", fmt="%.10g")


# =============================================================================
# MAIN
# =============================================================================
def main():
    OUT = OUT_DIR / CASE_NAME
    OUT.mkdir(parents=True, exist_ok=True)

    # 1) Lire XYZ
    X = read_xyz(XYZ_FILE, fmt=FMT, skiprows=SKIPROWS) * float(SCALE)
    print( read_xyz(XYZ_FILE, fmt=FMT, skiprows=SKIPROWS))

    # 2) Rayon primaire
    R = sphere_radius_from_rg(PRIMARY_RG * float(SCALE))
    print(X)

    # 3) Voxeliser -> volume binaire 3D
    vol_blanche, origin, voxel = voxelize_spheres(X, R=R, voxel=VOXEL_SIZE, padding=PADDING)
    vol_noire = ~vol_blanche

    # ===== Exports "binaire 3D" + aperçus =====
    if EXPORT_VOLUME_NPY:
        save_volume_npy(OUT / "vol_blanche.npy", vol_blanche)
        save_volume_npy(OUT / "vol_noire.npy", vol_noire)

    if EXPORT_VOLUME_TIFF:
        save_volume_tiff_stack(OUT / "vol_blanche_stack.tif", vol_blanche)

    if EXPORT_SLICES:
        save_middle_slices(OUT, vol_blanche, prefix="vol_blanche")
        save_middle_slices(OUT, vol_noire, prefix="vol_noire")

    if EXPORT_MIP:
        save_mip(OUT, vol_blanche, prefix="vol_blanche")
        save_mip(OUT, vol_noire, prefix="vol_noire")

    if SHOW_3D:
        show_voxel_3d(vol_blanche, max_voxels=SHOW_3D_MAX_VOXELS)

    # 4) CLD 3D (noire + blanche)
    print(len(vol_noire))
    print(len(vol_blanche))
    lens_noire = cld_3d_from_volume(vol_noire, n_lines=N_LINES, px_min=PX_MIN, seed=SEED+1, step_vox=LINE_STEP_VOX)
    lens_blanche = cld_3d_from_volume(vol_blanche, n_lines=N_LINES, px_min=PX_MIN, seed=SEED+2, step_vox=LINE_STEP_VOX)

    L_noire = float(np.mean(lens_noire)) if lens_noire.size else np.nan
    L_blanche = float(np.mean(lens_blanche)) if lens_blanche.size else np.nan
    max_noire = int(np.max(lens_noire)) if lens_noire.size else 0
    max_blanche = int(np.max(lens_blanche)) if lens_blanche.size else 0

    phi, Sv = phi_sv_from_mean_chords(L_noire, L_blanche)

    # 5) Histogrammes CLD
    Rn, fn = histogramme_matlab_probabilite(lens_noire, BINWIDTH_PX)
    Rb, fb = histogramme_matlab_probabilite(lens_blanche, BINWIDTH_PX)

    # 6) I(q) via ton modèle Matlab-like
    q, Iq = np.array([]), np.array([])
    if fb.size and fn.size and np.isfinite(Sv):
        # fm = blanche ; fv = noire
        q, Iq = modeliser_Iq_matlab(fm=fb, fv=fn, SurfSpe=Sv, k0=K0)

    # 7) Exports
    if Rn.size:
        save_csv_xy(OUT / "cld_phase_noire.csv", Rn, fn, "R_voxel;f(R)_phase_noire")
    if Rb.size:
        save_csv_xy(OUT / "cld_phase_blanche.csv", Rb, fb, "R_voxel;f(R)_phase_blanche")
    if q.size and Iq.size:
        save_csv_xy(OUT / "qI.csv", q, Iq, "q;I(q)")

    # Figure CLD
    plt.figure(figsize=(7, 5))
    if Rn.size:
        mn = fn > 0 if PLOT_LOGY else np.ones_like(fn, dtype=bool)
        plt.plot(Rn[mn], fn[mn], "-o", ms=3, lw=1, label="Phase Noire")
    if Rb.size:
        mb = fb > 0 if PLOT_LOGY else np.ones_like(fb, dtype=bool)
        plt.plot(Rb[mb], fb[mb], "-o", ms=3, lw=1, label="Phase Blanche")
    plt.xlabel("R (voxel)")
    plt.ylabel("f(R) (probabilité)")
    plt.title(f"CLD 3D (depuis XYZ) — {CASE_NAME}", fontsize=13, fontweight="bold")
    if PLOT_LOGY:
        plt.yscale("log")
    plt.xlim(PX_MIN, XMAX_R)
    plt.legend(loc="lower left")

    txt = (
        "φ (Phase Noire) = %.4f\n"
        "φ (Phase Blanche) = %.4f\n"
        "⟨L Blanche⟩ = %.2f vox (max=%d)\n"
        "⟨L Noire⟩ = %.2f vox (max=%d)\n"
        "Sv = %.3e (1/vox)"
    ) % (
        phi, 1.0 - phi,
        L_blanche, max_blanche,
        L_noire, max_noire,
        Sv
    )
    plt.text(
        0.98, 0.98, txt, transform=plt.gca().transAxes,
        ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", alpha=0.85)
    )
    plt.tight_layout()
    plt.savefig(OUT / "CLD_3D.png", dpi=200)
    plt.close()

    # I(q) export
    if q.size and Iq.size:
        plt.figure(figsize=(7, 5))
        plt.plot(q, Iq, "xr", lw=2)
        plt.xlabel("Q")
        plt.ylabel("Intensité modélisée")
        plt.title(f"I(Q) — {CASE_NAME}\n(depuis CLD 3D)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUT / "Iq_linear.png", dpi=200)
        plt.close()

        plt.figure(figsize=(7, 5))
        m = np.isfinite(q) & np.isfinite(Iq) & (q > 0) & (Iq > 0)
        plt.plot(q[m], Iq[m], "xr", lw=2)
        plt.xscale("log"); plt.yscale("log")
        plt.xlabel("Q")
        plt.ylabel("Intensité modélisée")
        plt.title(f"I(Q) log-log — {CASE_NAME}\n(depuis CLD 3D)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUT / "Iq_loglog.png", dpi=200)
        plt.close()

    # Resume
    resume = [
        "===== CLD 3D depuis XYZ + I(q) Matlab-like =====",
        f"XYZ_FILE: {XYZ_FILE}",
        f"FMT: {FMT}, SKIPROWS={SKIPROWS}",
        f"SCALE: {SCALE}",
        f"PRIMARY_RG: {PRIMARY_RG} (avant scale)",
        f"R_sphere: {R} (après scale)",
        f"VOXEL_SIZE: {VOXEL_SIZE}",
        f"PADDING: {PADDING}",
        f"VOL shape: {vol_blanche.shape}",
        f"N_LINES: {N_LINES}",
        f"LINE_STEP_VOX: {LINE_STEP_VOX}",
        f"<L_noire>= {L_noire:.6g} vox ; max_noire={max_noire}",
        f"<L_blanche>= {L_blanche:.6g} vox ; max_blanche={max_blanche}",
        f"phi_noire= {phi:.6g} ; phi_blanche={1-phi:.6g}",
        f"Sv= {Sv:.6g} (1/vox)",
        f"I(q) computed: {bool(q.size and Iq.size)}",
        f"OUT: {OUT}",
        "",
        "Exports volume:",
        f"  NPY: {EXPORT_VOLUME_NPY}",
        f"  TIFF stack: {EXPORT_VOLUME_TIFF}",
        f"  Slices: {EXPORT_SLICES}",
        f"  MIP: {EXPORT_MIP}",
        f"  SHOW_3D: {SHOW_3D}",
    ]
    (OUT / "resume.txt").write_text("\n".join(resume), encoding="utf-8")

    # Liste les fichiers png générés (utile pour vérifier)
    print("\n".join(resume))
    print("\nFichiers PNG générés :")
    for p in sorted(OUT.glob("*.png")):
        print(" -", p)

    print("\nTerminé ✅")


if __name__ == "__main__":
    main()
