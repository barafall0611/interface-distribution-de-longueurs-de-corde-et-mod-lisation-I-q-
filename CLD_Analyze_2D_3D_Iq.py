# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 13:48:45 2026

@author: bara.fall
"""

# -*- coding: utf-8 -*-
"""
CLD 2D (image) + XYZ -> CLD 3D + I(q) — GUI PyQt5 

# %%
- Onglet 1 : ton interface 2D (4 cas + export + affichage CLD/IQ)
- Onglet 2 : interface XYZ (fichier xyz/csv/txt -> voxelisation sphères -> CLD 3D -> I(q) + export csv)
- PyVista : optionnel, NON bloquant si pyvistaqt.BackgroundPlotter dispo

Lancer :
    python cld_gui_full.py
"""
#

import sys
import io
import contextlib
from pathlib import Path
import numpy as np
import tifffile as tiff


from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout, QGroupBox,
    QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QPushButton, QTextEdit,
    QLabel, QTabWidget, QProgressBar, QScrollArea
)

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import matplotlib.pyplot as plt

from skimage import io as skio
from skimage.color import rgb2gray
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.draw import line as sk_line


# =============================================================================
# STYLE
# =============================================================================
APP_QSS = """
* { font-family: "Segoe UI", "Inter", Arial; font-size: 11px; }
QMainWindow { background: #0f1116; }

QGroupBox {
  background: #151824;
  border: 1px solid #2a3146;
  border-radius: 10px;
  margin-top: 18px;
  padding: 18px 10px 10px 10px;
}

QGroupBox::title {
  subcontrol-origin: margin;
  subcontrol-position: top left;
  left: 12px;
  top: 6px;
  padding: 4px 10px;
  background: #1b2133;
  border: 1px solid #2d6cdf;
  border-radius: 8px;
  color: #ffffff;
  font-weight: 900;
  font-size: 12px;
}

QLabel { color: #d6d9e0; }

QLineEdit, QSpinBox, QDoubleSpinBox {
  background: #0f1116;
  border: 1px solid #22283a;
  border-radius: 7px;
  padding: 3px 6px;
  color: #e7e9ee;
  selection-background-color: #2d6cdf;
  min-height: 20px;
}

QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 16px; }

QTextEdit {
  background: #0f1116;
  border: 1px solid #22283a;
  border-radius: 7px;
  padding: 6px;
  color: #e7e9ee;
  font-family: Consolas, "Cascadia Mono", monospace;
  font-size: 10px;
  min-height: 70px;
}

QCheckBox { color: #d6d9e0; spacing: 8px; }
QCheckBox::indicator { width: 16px; height: 16px; }
QCheckBox::indicator:unchecked { border: 1px solid #2a3146; background: #0f1116; border-radius: 4px; }
QCheckBox::indicator:checked { border: 1px solid #2d6cdf; background: #2d6cdf; border-radius: 4px; }

QPushButton {
  background: #1d2232;
  border: 1px solid #2a3146;
  color: #e7e9ee;
  padding: 5px 10px;
  border-radius: 8px;
  font-weight: 800;
  min-height: 24px;
}

QPushButton:hover { background: #232a3d; }
QPushButton:pressed { background: #192034; }
QPushButton#PrimaryButton { background: #2d6cdf; border: 1px solid #2d6cdf; }
QPushButton#PrimaryButton:hover { background: #3a78e3; }
QPushButton:disabled { background: #141826; color: #7d859d; border: 1px solid #22283a; }

QTabWidget::pane { border: 1px solid #22283a; border-radius: 10px; background: #151824; }





QTabBar::tab {
  background: #10131c;
  border: 1px solid #22283a;
  border-bottom: none;

  /* + grand */
  padding: 10px 18px;        /* hauteur + largeur */
  min-height: 34px;          /* hauteur mini */
  min-width: 120px;          /* largeur mini */
  font-size: 12.5px;         /* texte plus grand */

  margin-right: 8px;
  border-top-left-radius: 10px;
  border-top-right-radius: 10px;

  color: #cfd3dd;
  font-weight: 800;
}

QTabBar::tab:selected {
  background: #151824;
  color: #ffffff;
  border-color: #2a3146;
}


QTabWidget::pane {
  border: 1px solid #22283a;
  border-radius: 10px;
  background: #151824;
  padding-top: 4px;
}



QTabBar::tab:selected { background: #151824; color: #ffffff; border-color: #2a3146; }

QProgressBar {
  border: 1px solid #22283a;
  border-radius: 8px;
  background: #0f1116;
  text-align: center;
  color: #e7e9ee;
  padding: 1px;
  min-height: 14px;
}

QProgressBar::chunk { background-color: #2d6cdf; border-radius: 7px; }
QScrollArea { background: transparent; border: none; }
"""


# =============================================================================
# Matplotlib Canvas (propre)
# =============================================================================
class MplCanvas(FigureCanvas):
    def __init__(self):
        fig = Figure(figsize=(7, 5), dpi=100)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)


    
# =============================================================================
# OUTILS 2D (image)
# =============================================================================
def lire_en_gris(img_path):
    img = skio.imread(img_path)
    if img.ndim == 3:
        img = rgb2gray(img)
    img = img.astype(float)
    if img.max() > 1.5:
        img /= 255.0
    return img

def binariser_image(img_gray, thr_8bit=125.7):
    img8 = (img_gray * 255).round().astype(np.uint8)
    vals = np.unique(img8)
    print("Nombre de niveaux de gris :", len(vals))
    print("valeurs (min..max):", vals[:10], "...", vals[-10:])
    return img8 > thr_8bit

def corriger_bords(binary):
    b = binary.copy()
    b[0, :] = False; b[-1, :] = False
    b[:, 0] = False; b[:, -1] = False
    return b

def supprimer_pixels_isoles(binary, min_size=5):
    b = binary.astype(bool)
    b = remove_small_objects(b, min_size=min_size)
    b = remove_small_holes(b, area_threshold=min_size)
    return b


# =============================================================================
# CLD 2D – coeur
# =============================================================================
def longueurs_cordes_1d(arr_bool, px_min):
    a = np.asarray(arr_bool).astype(bool).astype(np.int8)
    da = np.diff(np.r_[0, a, 0])
    starts = np.where(da == 1)[0]
    ends = np.where(da == -1)[0]
    n = min(len(starts), len(ends))
    if n == 0:
        return np.array([], dtype=int)
    L = ends[:n] - starts[:n]
    return L[L >= px_min]

def _couper_droite_dans_image(theta, x0, y0, w, h):
    dx = np.cos(theta); dy = np.sin(theta)
    ts = []
    if abs(dx) > 1e-12:
        t = (0 - x0) / dx; y = y0 + t * dy
        if 0 <= y <= h - 1: ts.append(t)
        t = ((w - 1) - x0) / dx; y = y0 + t * dy
        if 0 <= y <= h - 1: ts.append(t)
    if abs(dy) > 1e-12:
        t = (0 - y0) / dy; x = x0 + t * dx
        if 0 <= x <= w - 1: ts.append(t)
        t = ((h - 1) - y0) / dy; x = x0 + t * dx
        if 0 <= x <= w - 1: ts.append(t)
    if len(ts) < 2:
        return None
    t1, t2 = float(np.min(ts)), float(np.max(ts))
    xA, yA = x0 + t1 * dx, y0 + t1 * dy
    xB, yB = x0 + t2 * dx, y0 + t2 * dy
    return xA, yA, xB, yB


def tirer_droites_isotropes(binary_mask_true, n_lines=9000, px_min=1, seed=0):
    rng = np.random.default_rng(seed)
    h, w = binary_mask_true.shape
    chords = []
    for _ in range(int(n_lines)):
        theta = float(rng.uniform(0, np.pi))
        x0 = float(rng.uniform(0, w - 1))
        y0 = float(rng.uniform(0, h - 1))
        seg = _couper_droite_dans_image(theta, x0, y0, w, h)
        if seg is None:
            continue
        xA, yA, xB, yB = seg
        rr, cc = sk_line(int(round(yA)), int(round(xA)),
                         int(round(yB)), int(round(xB)))
        m = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        rr, cc = rr[m], cc[m]
        if rr.size < 2:
            continue
        profile = binary_mask_true[rr, cc]
        cl = longueurs_cordes_1d(profile, px_min=px_min)
        if cl.size:
            chords.append(cl)
    return np.concatenate(chords).astype(float) if chords else np.array([])


# =============================================================================
# Histogramme / I(q) / métriques (communes)
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

def preparer_fm_fv_matlab(fm, fv, cal_len):
    fm = np.asarray(fm, dtype=float)
    fv = np.asarray(fv, dtype=float)
    L = max(len(fm), len(fv))
    if len(fm) < L: fm = np.pad(fm, (0, L - len(fm)))
    if len(fv) < L: fv = np.pad(fv, (0, L - len(fv)))
    fm2 = fm[fm != 0]
    fv2 = fv[fv != 0]
    if cal_len is not None and np.isfinite(cal_len) and cal_len != 0:
        fm2 = fm2 / cal_len
        fv2 = fv2 / cal_len
    return fm2, fv2

def modeliser_Iq_matlab_2d(fm, fv, SurfSpe, eps_denom=1e-12, k0=1):
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
    s = np.arange(k0, half + 1, dtype=float) / float(L)
    a_parent = a_parent[k0 - 1:]
    q = 2 * np.pi * s
    q = q / 2
    parent_bis = a_parent / (s ** 2)
    derv = np.diff(parent_bis) / np.diff(s)
    I = (-SurfSpe / (16 * s[:-1] * (np.pi ** 3))) * derv
    return q[:-1], I

def surface_specific_from_L(Lp_px, Lm_px, cal_len_um_per_px=None):
    if not (np.isfinite(Lp_px) and np.isfinite(Lm_px)) or (Lp_px <= 0) or (Lm_px <= 0):
        return np.nan, np.nan, np.nan, np.nan
    phi = Lp_px / (Lp_px + Lm_px)
    surf_1_per_px = (4 * phi * (1 - phi)) * (1 / Lm_px + 1 / Lp_px)
    if cal_len_um_per_px is None or (not np.isfinite(cal_len_um_per_px)) or cal_len_um_per_px == 0:
        return phi, surf_1_per_px, np.nan, np.nan
    surf_1_per_um = surf_1_per_px / cal_len_um_per_px
    return phi, surf_1_per_px, surf_1_per_um, np.nan


# =============================================================================
# Export helpers (communs)
# =============================================================================
def save_csv_xy(path: Path, x, y, header: str):
    data = np.column_stack([x, y])
    np.savetxt(path, data, delimiter=";", header=header, comments="", fmt="%.10g")

def save_image_u8(path: Path, img_float01: np.ndarray):
    arr = np.clip(img_float01 * 255, 0, 255).astype(np.uint8)
    skio.imsave(str(path), arr)

def save_bool_png(path: Path, b: np.ndarray):
    skio.imsave(str(path), (b.astype(np.uint8) * 255))


# =============================================================================
# Figures 2D
# =============================================================================
def build_cld_figure(res, case_name, *, plot_logy, px_min, xmax_r):
    fig = Figure(figsize=(7, 5), dpi=160)
    ax = fig.add_subplot(111)

    if res["Rp"].size:
        mp = res["fp"] > 0 if plot_logy else np.ones_like(res["fp"], dtype=bool)
        ax.plot(res["Rp"][mp], res["fp"][mp], "-o", markersize=3, linewidth=1, label="Phase Noire")
    if res["Rm"].size:
        mm = res["fm"] > 0 if plot_logy else np.ones_like(res["fm"], dtype=bool)
        ax.plot(res["Rm"][mm], res["fm"][mm], "-o", markersize=3, linewidth=1, label="Phase Blanche")

    ax.set_xlabel("R (pixel)")
    ax.set_ylabel("f(R) (probabilité)")
    ax.set_title(f"CLD — {case_name}", fontsize=14, fontweight="bold")

    if plot_logy:
        ax.set_yscale("log")
    ax.set_xlim(px_min, xmax_r)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)

    txt = (
        "φ (Phase Noire) = %.4f\n"
        "φ (Phase Blanche) = %.4f\n"
        "⟨Lm⟩ = %.2f px (max=%d)\n"
        "⟨Lp⟩ = %.2f px (max=%d)\n"
        "Sv = %.3e px⁻¹"
    ) % (
        res["phi"],
        1.0 - res["phi"],
        res["Lm_px"], res["max_m"],
        res["Lp_px"], res["max_p"],
        res["SurfSpe_1_per_px"]
    )

    ax.text(
        0.98, 0.98, txt,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", alpha=0.85)
    )

    fig.tight_layout()
    return fig

def build_iq_loglog_figure(res, case_name):
    fig = Figure(figsize=(7, 5), dpi=160)
    ax = fig.add_subplot(111)

    q = np.asarray(res["q"], dtype=float)
    I = np.asarray(res["I"], dtype=float)
    mask = np.isfinite(q) & np.isfinite(I) & (q > 0) & (I > 0)
    q = q[mask]; I = I[mask]
    if q.size > 1:
        q = q[1:]; I = I[1:]

    ax.plot(q, I, marker="x", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Q")
    ax.set_ylabel("Intensité modélisée")
    ax.set_title(f"I(Q) (log-log) — {case_name}", fontsize=12)

    fig.tight_layout()
    return fig

def build_iq_linear_figure(res, case_name):
    fig = Figure(figsize=(7, 5), dpi=160)
    ax = fig.add_subplot(111)

    q = np.asarray(res["q"], dtype=float)
    I = np.asarray(res["I"], dtype=float)

    mask = np.isfinite(q) & np.isfinite(I)
    q = q[mask]
    I = I[mask]
    if q.size > 1:
        q = q[1:]
        I = I[1:]

    ax.plot(q, I, "xr", linewidth=2)
    ax.set_xlabel("Q ", fontsize=14, fontweight="bold")
    ax.set_ylabel("Intensité modélisée", fontsize=14, fontweight="bold")
    ax.set_title(f"I(Q) — {case_name}\nModélisation de l'intensité en fonction de Q", fontsize=12)

    fig.tight_layout()
    return fig


# =============================================================================
# EXECUTER CAS 2D
# =============================================================================
def executer_cas_2d(
    case_name, img_gray, binary0, *,
    do_isolated, do_border,
    N_LINES, PX_MIN, BINWIDTH_PX, CAL_LEN_UM_PER_PX, MIN_OBJECT_SIZE,
    seed_p=1, seed_m=2
):
    binary = binary0.copy()
    if do_isolated:
        binary = supprimer_pixels_isoles(binary, min_size=MIN_OBJECT_SIZE)
    if do_border:
        binary = corriger_bords(binary)

    pores_mask = ~binary
    mat_mask = binary

    lens_p = tirer_droites_isotropes(pores_mask, n_lines=N_LINES, px_min=PX_MIN, seed=seed_p)
    lens_m = tirer_droites_isotropes(mat_mask,   n_lines=N_LINES, px_min=PX_MIN, seed=seed_m)

    Lp = float(np.mean(lens_p)) if lens_p.size else np.nan
    Lm = float(np.mean(lens_m)) if lens_m.size else np.nan
    max_p = int(np.max(lens_p)) if lens_p.size else 0
    max_m = int(np.max(lens_m)) if lens_m.size else 0

    phi, surf_1px, _, _ = surface_specific_from_L(Lp_px=Lp, Lm_px=Lm, cal_len_um_per_px=CAL_LEN_UM_PER_PX)
    SurfSpe_for_I = surf_1px / CAL_LEN_UM_PER_PX if (CAL_LEN_UM_PER_PX and np.isfinite(CAL_LEN_UM_PER_PX)) else surf_1px

    print(case_name)
    print(f"  nbiteration: {N_LINES}")
    print(f"  moyenne blanc :  {Lm:.6f}     maxi : {max_m}")
    print(f"  moyenne noir:  {Lp:.6f}     maxi : {max_p}")
    print(f"  porosite :      {phi:.8f}")
    print(f"  SurfSpe (sans callen) : {surf_1px}")
    print(f"  SurfSpe (avec callen) : {SurfSpe_for_I}")

    Rp, fp = histogramme_matlab_probabilite(lens_p, BINWIDTH_PX)
    Rm, fm = histogramme_matlab_probabilite(lens_m, BINWIDTH_PX)

    fm2, fv2 = np.array([]), np.array([])
    if fp.size and fm.size:
        fm2, fv2 = preparer_fm_fv_matlab(fm=fm, fv=fp, cal_len=CAL_LEN_UM_PER_PX)

    q, I = np.array([]), np.array([])
    if fm2.size and fv2.size and np.isfinite(SurfSpe_for_I):
        q, I = modeliser_Iq_matlab_2d(fm=fm, fv=fp, SurfSpe=SurfSpe_for_I, k0=1)

    return {
        "case": case_name,
        "binary": binary,
        "binary0": binary0,
        "img_gray": img_gray,
        "Lp_px": Lp, "Lm_px": Lm,
        "max_p": max_p, "max_m": max_m,
        "phi": phi,
        "SurfSpe_1_per_px": surf_1px,
        "Rp": Rp, "fp": fp,
        "Rm": Rm, "fm": fm,
        "q": q, "I": I
    }


# =============================================================================
# XYZ -> CLD 3D + I(q) (fonctions du script XYZ)
# =============================================================================
def read_xyz(path: str, fmt: str = "csv", skiprows: int = 0) -> np.ndarray:
    delimiter = "," if fmt.lower() == "csv" else None
    data = np.loadtxt(path, delimiter=delimiter, skiprows=int(skiprows))
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 3:
        raise ValueError("Fichier invalide: il faut au moins 3 colonnes (x,y,z).")
    return data[:, :3].astype(float)

def sphere(path, join):
    print('....')

def sphere_radius_from_rg(Rg_primary: float) -> float:
    return float(np.sqrt(5.0 / 3.0) * float(Rg_primary))

def estimate_typical_spacing(X: np.ndarray, n_pairs: int = 20000, seed: int = 0) -> float:
    X = np.asarray(X, float)
    n = X.shape[0]
    if n < 2:
        return np.nan
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=int(n_pairs))
    j = rng.integers(0, n, size=int(n_pairs))
    m = (i != j)
    if not np.any(m):
        return np.nan
    d = np.linalg.norm(X[i[m]] - X[j[m]], axis=1)
    return float(np.median(d)) if d.size else np.nan

def compute_R_in_xyz_units(
    X_xyz: np.ndarray,
    Rg_angstrom: float,
    angstrom_per_xyz: float | None,
    target_fraction: float = 0.35,
    seed: int = 0
):
    R_A = sphere_radius_from_rg(Rg_angstrom)

    if angstrom_per_xyz is not None and np.isfinite(angstrom_per_xyz) and angstrom_per_xyz > 0:
        R_xyz = R_A / float(angstrom_per_xyz)
        mode = "physical_conversion"
        spacing = None
        ang_used = float(angstrom_per_xyz)
    else:
        spacing = estimate_typical_spacing(X_xyz, n_pairs=20000, seed=seed)
        tf = float(np.clip(target_fraction, 0.05, 0.8))
        if not np.isfinite(spacing) or spacing <= 0:
            R_xyz = 1.0
            ang_used = float("nan")
        else:
            R_xyz = tf * spacing
            ang_used = float(R_A / R_xyz)
        mode = "auto_visual_scale"

    info = {
        "mode": mode,
        "Rg_A": float(Rg_angstrom),
        "R_A": float(R_A),
        "R_xyz": float(R_xyz),
        "angstrom_per_xyz": ang_used,
        "spacing_med": None if spacing is None else float(spacing),
        "target_fraction": float(target_fraction),
    }
    return float(R_xyz), float(R_A), info

def show_xyz_pyvista_igor_like(
    X, R_xyz, case_name,
    max_spheres=30000, seed=0,
    sphere_res=10,
    bg=(0.82, 0.82, 0.82),
    color=(0.25, 1.0, 0.25),
):
    import pyvista as pv
    X = np.asarray(X, float)
    if X.size == 0 or X.shape[0] == 0:
        print("[PyVista] Aucun point à afficher.")
        return None

    if X.shape[0] > max_spheres:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], int(max_spheres), replace=False)
        X = X[idx]
        print(f"[PyVista] Sous-échantillonnage: {max_spheres} sphères")

    mn0 = X.min(axis=0) - R_xyz
    mx0 = X.max(axis=0) + R_xyz
    max_abs = np.max(np.abs([mn0, mx0]))
    half = float(max_abs)
    mn = np.array([-half, -half, -half])
    mx = np.array([ half,  half,  half])
    bounds = (mn[0], mx[0], mn[1], mx[1], mn[2], mx[2])
    box = pv.Box(bounds=bounds)

    cloud = pv.PolyData(X)
    sphere = pv.Sphere(radius=float(R_xyz), theta_resolution=int(sphere_res), phi_resolution=int(sphere_res))
    glyphs = cloud.glyph(scale=False, orient=False, geom=sphere)

    try:
        from pyvistaqt import BackgroundPlotter
    except Exception as e:
        print("[PyVistaQt] NON DISPONIBLE -> pas d'affichage PyVista (programme continue).")
        print("Détail:", e)
        return None

    bp = BackgroundPlotter(window_size=(950, 720))
    bp.set_background(bg)
    bp.add_mesh(box, style="wireframe", color="yellow", line_width=2)
    bp.add_mesh(
        glyphs, color=color, smooth_shading=True,
        specular=1.0, specular_power=80, diffuse=0.7, ambient=0.15
    )
    bp.add_axes(line_width=2)
    bp.show_grid(color="black")

    try:
        bp.add_text(case_name, position="upper_left", font_size=12, color="black")
    except Exception:
        pass
    try:
        bp.enable_eye_dome_lighting()
    except Exception:
        pass

    print("[PyVistaQt] BackgroundPlotter OK (non bloquant).")
    return bp

def voxelize_spheres_safe(X: np.ndarray, R: float, voxel: float, padding: float, max_voxels_total: int):
    X = np.asarray(X, float)
    voxel = float(voxel)
    R = float(R)
    pad = float(padding)

    mn = X.min(axis=0) - (R + pad)
    mx = X.max(axis=0) + (R + pad)

    while True:
        dims = np.ceil((mx - mn) / voxel).astype(int) + 1
        nx, ny, nz = dims.tolist()
        total = int(nx) * int(ny) * int(nz)
        if total <= int(max_voxels_total):
            break
        voxel *= 1.25

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

    return vol, mn, voxel, (nx, ny, nz)

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
        p0 = np.array([rng.uniform(0, nx - 1), rng.uniform(0, ny - 1), rng.uniform(0, nz - 1)], dtype=float)

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

def phi_sv_from_mean_chords(L_noire, L_blanche):
    if not (np.isfinite(L_noire) and np.isfinite(L_blanche)) or L_noire <= 0 or L_blanche <= 0:
        return np.nan, np.nan
    phi = L_noire / (L_noire + L_blanche)
    Sv = (4 * phi * (1 - phi)) * (1 / L_blanche + 1 / L_noire)
    return phi, Sv

def modeliser_Iq_matlab_3d(fm, fv, SurfSpe, eps_denom=1e-12, k0=1):
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

    # (comme ton script XYZ)
    s = np.arange(k0, half + 1, dtype=float) / (float(L) / 2)
    a_parent = a_parent[k0 - 1:]

    q = 2 * np.pi * s
    parent_bis = a_parent / (s**2)
    derv = np.diff(parent_bis) / np.diff(s)
    I = (-SurfSpe / (16 * s[:-1] * (np.pi**3))) * derv
    return q[:-1], I


# =============================================================================
# Onglet 2D (interface image complète)
# =============================================================================
class CLD2DTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.img_path = ""
        self.out_dir = ""

        # IO
        self.path_value = QLineEdit(); self.path_value.setReadOnly(True); self.path_value.setPlaceholderText("Choisir une image…")
        self.btn_image = QPushButton("📷  Image…"); self.btn_image.clicked.connect(self.choose_image)

        self.out_value = QLineEdit(); self.out_value.setReadOnly(True); self.out_value.setPlaceholderText("Choisir un dossier…")
        self.btn_out = QPushButton("📁 Sortie…"); self.btn_out.clicked.connect(self.choose_outdir)

        self.prefix = QLineEdit("resultats")

        # Params
        self.nlines = QSpinBox(); self.nlines.setRange(100, 5_000_000); self.nlines.setValue(10000)
        self.thr = QDoubleSpinBox(); self.thr.setRange(0.0, 255.0); self.thr.setDecimals(1); self.thr.setValue(125.7)
        self.pxmin = QSpinBox(); self.pxmin.setRange(1, 500); self.pxmin.setValue(1)
        self.binw = QSpinBox(); self.binw.setRange(1, 50); self.binw.setValue(1)
        self.callen = QDoubleSpinBox(); self.callen.setRange(1e-9, 1e9); self.callen.setDecimals(6); self.callen.setValue(1.0)
        self.minobj = QSpinBox(); self.minobj.setRange(1, 10000); self.minobj.setValue(5)
        self.logy = QCheckBox("Axe Y en log"); self.logy.setChecked(True)
        self.xmax = QSpinBox(); self.xmax.setRange(1, 100000); self.xmax.setValue(150)
        self.export_bin = QCheckBox("Exporter les binaires"); self.export_bin.setChecked(True)

        # Run
        self.btn_run = QPushButton("▶  Lancer (avec les 4 cas) + Export")
        self.btn_run.setObjectName("PrimaryButton")
        self.btn_run.clicked.connect(self.run)
        self.progress = QProgressBar(); self.progress.setRange(0, 100); self.progress.setValue(0)

        # Log
        self.log = QTextEdit(); self.log.setPlaceholderText("Console 2D")
        self.log.setMinimumHeight(80)

        # Tabs
        self.tabs = QTabWidget()
        self.canvas_cld = MplCanvas()
        self.canvas_iq_lin = MplCanvas()
        self.canvas_iq_log = MplCanvas()

        self.canvas_cld.ax.text(
            0.5, 0.5, "Sélectionne une image puis clique sur Lancer",
            transform=self.canvas_cld.ax.transAxes,
            ha="center", va="center", fontsize=12, alpha=0.5
        )
        self.canvas_cld.draw()

        self.tabs.addTab(self.canvas_cld, "CLD (dernier cas)")
        self.tabs.addTab(self.canvas_iq_lin, "I(Q) (lin) (dernier cas)")
        self.tabs.addTab(self.canvas_iq_log, "I(Q) log-log (dernier cas)")

        # Left scroll
        left_content = QWidget()
        left_layout = QVBoxLayout(left_content)
        left_layout.setSpacing(5)

        io_box = QGroupBox("Entrées / Sorties")
        io_grid = QGridLayout(io_box)
        io_grid.setHorizontalSpacing(5); io_grid.setVerticalSpacing(5)
        io_grid.addWidget(QLabel("Image"), 0, 0)
        io_grid.addWidget(self.path_value, 0, 1)
        io_grid.addWidget(self.btn_image, 0, 2)
        io_grid.addWidget(QLabel("Dossier"), 1, 0)
        io_grid.addWidget(self.out_value, 1, 1)
        io_grid.addWidget(self.btn_out, 1, 2)
        io_grid.addWidget(QLabel("Nom du Dossier"), 2, 0)
        io_grid.addWidget(self.prefix, 2, 1, 1, 2)

        p_box = QGroupBox("Paramètres")
        form = QFormLayout(p_box)
        form.setVerticalSpacing(4)
        form.setHorizontalSpacing(10)
        form.addRow("Nb_Chord", self.nlines)
        form.addRow("thr_8bit (seuil binaire)", self.thr)
        form.addRow("PX_MIN", self.pxmin)
        form.addRow("BINWIDTH_PX", self.binw)
        form.addRow("CAL_LEN_UM_PER_PX", self.callen)
        form.addRow("MIN_OBJECT_SIZE", self.minobj)
        form.addRow(self.logy)
        form.addRow("XMAX_R_PX", self.xmax)
        form.addRow(self.export_bin)

        run_box = QGroupBox("Lancement")
        run_layout = QVBoxLayout(run_box)
        run_layout.setSpacing(6)
        run_layout.addWidget(self.btn_run)
        run_layout.addWidget(self.progress)

        log_box = QGroupBox("Console")
        log_layout = QVBoxLayout(log_box)
        log_layout.setSpacing(6)
        log_layout.addWidget(self.log)

        left_layout.addWidget(io_box)
        left_layout.addWidget(p_box)
        left_layout.addWidget(run_box)
        left_layout.addWidget(log_box, 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left_content)
        scroll.setFixedWidth(380)

        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        root.addWidget(scroll, 0)
        root.addWidget(self.tabs, 1)

    def choose_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choisir une image", "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;Tous les fichiers (*.*)"
        )
        if path:
            self.img_path = path
            self.path_value.setText(path)

    def choose_outdir(self):
        d = QFileDialog.getExistingDirectory(self, "Choisir dossier de sortie", "")
        if d:
            self.out_dir = d
            self.out_value.setText(d)

    def set_busy(self, busy: bool):
        for w in [self.btn_run, self.btn_image, self.btn_out]:
            w.setEnabled(not busy)

    def run(self):
        if not self.img_path:
            QMessageBox.warning(self, "Image manquante", "Choisis d'abord une image.")
            return
        if not self.out_dir:
            QMessageBox.warning(self, "Sortie manquante", "Choisis un dossier de sortie.")
            return

      
        Nb_Chord = int(self.nlines.value())


        thr_8bit = float(self.thr.value())
        PX_MIN = int(self.pxmin.value())
        BINWIDTH_PX = int(self.binw.value())
        CAL_LEN_UM_PER_PX = float(self.callen.value())
        MIN_OBJECT_SIZE = int(self.minobj.value())
        PLOT_LOGY = bool(self.logy.isChecked())
        XMAX_R_PX = int(self.xmax.value())
        EXPORT_BIN = bool(self.export_bin.isChecked())
        prefix = (self.prefix.text().strip() or "resultats")

        out_dir = Path(self.out_dir) / prefix
        out_dir.mkdir(parents=True, exist_ok=True)

        buf = io.StringIO()

        try:
            self.set_busy(True)
            self.progress.setValue(0)
            self.log.clear()

            with contextlib.redirect_stdout(buf):
                img_gray = lire_en_gris(self.img_path)
                binary0 = binariser_image(img_gray, thr_8bit=thr_8bit)

                save_image_u8(out_dir / "img_gray.png", img_gray)
                save_bool_png(out_dir / "binary0.png", binary0)

                cases = [
                        ("Pas de correction",     False, False, 3, 4),
                        ("correction des bords",  False, True,  5, 6),
                        ("correction pixels isolés", True, False, 7, 8),
                        ("correction pixel+bord", True,  True,  1, 2),
                    ]

                last = None
                

                for i, (name, do_iso, do_bord, sp, sm) in enumerate(cases, start=1):
                    res = executer_cas_2d(
                        name, img_gray, binary0,
                        do_isolated=do_iso, do_border=do_bord,
                        N_LINES=Nb_Chord, PX_MIN=PX_MIN, BINWIDTH_PX=BINWIDTH_PX,
                        CAL_LEN_UM_PER_PX=CAL_LEN_UM_PER_PX, MIN_OBJECT_SIZE=MIN_OBJECT_SIZE,
                        seed_p=sp, seed_m=sm
                    )
                    last = res

                    tag = f"cas_{i:02d}_{name.replace(' ', '_').replace('+','plus')}"

                    if EXPORT_BIN:
                        save_bool_png(out_dir / f"{tag}_binary_corr.png", res["binary"])

                    if res["Rp"].size:
                        save_csv_xy(out_dir / f"{tag}_cld_pores.csv", res["Rp"], res["fp"], "R_pixel;f_R_probabilite")
                    if res["Rm"].size:
                        save_csv_xy(out_dir / f"{tag}_cld_matiere.csv", res["Rm"], res["fm"], "R_pixel;f_R_probabilite")
                    if res["q"].size and res["I"].size:
                        save_csv_xy(out_dir / f"{tag}_qI.csv", res["q"], res["I"], "q;I")

                    fig_cld = build_cld_figure(res, name, plot_logy=PLOT_LOGY, px_min=PX_MIN, xmax_r=XMAX_R_PX)
                    fig_cld.savefig(str(out_dir / f"{tag}_CLD.png"))

                    if res["q"].size and res["I"].size:
                        fig_iq_lin = build_iq_linear_figure(res, name)
                        fig_iq_lin.savefig(str(out_dir / f"{tag}_Iq_linear.png"))

                        fig_iq_log = build_iq_loglog_figure(res, name)
                        fig_iq_log.savefig(str(out_dir / f"{tag}_Iq_loglog.png"))

                    self.progress.setValue(int(i / 4 * 100))

            txt_complet = buf.getvalue()
            (out_dir / "resume_complet.txt").write_text(txt_complet, encoding="utf-8")
            self.log.setPlainText(txt_complet)

            # UI update (dernier cas) — IMPORTANT : utiliser last, pas res
            if last is not None:
                # CLD
                self.canvas_cld.ax.clear()
                if last["Rp"].size:
                    mp = last["fp"] > 0 if PLOT_LOGY else np.ones_like(last["fp"], dtype=bool)
                    self.canvas_cld.ax.plot(last["Rp"][mp], last["fp"][mp], "-o", markersize=3, linewidth=1, label="Phase Noire")
                if last["Rm"].size:
                    mm = last["fm"] > 0 if PLOT_LOGY else np.ones_like(last["fm"], dtype=bool)
                    self.canvas_cld.ax.plot(last["Rm"][mm], last["fm"][mm], "-o", markersize=3, linewidth=1, label="Phase Blanche")

                self.canvas_cld.ax.set_xlabel("R (pixel)")
                self.canvas_cld.ax.set_ylabel("f(R) (probabilité)")
                self.canvas_cld.ax.set_title(f"CLD — {last['case']}", fontsize=14, fontweight="bold")
                if PLOT_LOGY:
                    self.canvas_cld.ax.set_yscale("log")
                self.canvas_cld.ax.set_xlim(PX_MIN, XMAX_R_PX)
                self.canvas_cld.ax.legend(loc="lower left", fontsize=10, framealpha=0.9)

                txt = (
                    "φ (Phase Noire) = %.4f\n"
                    "φ (Phase Blanche) = %.4f\n"
                    "⟨L Blanche⟩ = %.2f px (max=%d)\n"
                    "⟨L Noire⟩ = %.2f px (max=%d)\n"
                    "Sv = %.3e px⁻¹"
                ) % (
                    last["phi"],
                    1.0 - last["phi"],
                    last["Lm_px"], last["max_m"],
                    last["Lp_px"], last["max_p"],
                    last["SurfSpe_1_per_px"]
                )

                self.canvas_cld.ax.text(
                    0.98, 0.98, txt,
                    transform=self.canvas_cld.ax.transAxes,
                    ha="right", va="top",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.35", alpha=0.85)
                )
                self.canvas_cld.figure.tight_layout()
                self.canvas_cld.draw()

                # I(Q) lin
                self.canvas_iq_lin.ax.clear()
                q = np.asarray(last["q"], dtype=float)
                I = np.asarray(last["I"], dtype=float)
                mask = np.isfinite(q) & np.isfinite(I)
                q2 = q[mask]
                I2 = I[mask]
                if q2.size > 1:
                    q2 = q2[1:]
                    I2 = I2[1:]
                if q2.size:
                    self.canvas_iq_lin.ax.plot(q2, I2, "xr", linewidth=2)
                self.canvas_iq_lin.ax.set_xlabel("Q", fontsize=14, fontweight="bold")
                self.canvas_iq_lin.ax.set_ylabel("Intensité modélisée", fontsize=14, fontweight="bold")
                self.canvas_iq_lin.ax.set_title(f"Modélisation de l'intensité en fonction de Q - {last['case']}",
                                                fontsize=12, fontweight="bold")
                self.canvas_iq_lin.figure.tight_layout()
                self.canvas_iq_lin.draw()

                # I(Q) log-log
                self.canvas_iq_log.ax.clear()
                q = np.asarray(last["q"], dtype=float)
                I = np.asarray(last["I"], dtype=float)
                mask = np.isfinite(q) & np.isfinite(I) & (q > 0) & (I > 0)
                q = q[mask]; I = I[mask]
                if q.size > 1:
                    q = q[1:]; I = I[1:]
                if q.size:
                    self.canvas_iq_log.ax.plot(q, I, "x-", linewidth=1.5)
                    self.canvas_iq_log.ax.set_xscale("log")
                    self.canvas_iq_log.ax.set_yscale("log")
                self.canvas_iq_log.ax.set_xlabel("Q")
                self.canvas_iq_log.ax.set_ylabel("Intensité modélisée")
                self.canvas_iq_log.ax.set_title(f"I(Q) (log-log) — {last['case']}")
                self.canvas_iq_log.figure.tight_layout()
                self.canvas_iq_log.draw()

            self.progress.setValue(100)
            self.set_busy(False)

        except Exception as e:
            self.set_busy(False)
            QMessageBox.critical(self, "Erreur", str(e))


# =============================================================================
# Onglet XYZ (interface complète)
# =============================================================================
class XYZTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.xyz_path = ""
        self.out_dir = ""

        # IO
        self.xyz_value = QLineEdit(); self.xyz_value.setReadOnly(True); self.xyz_value.setPlaceholderText("Choisir un fichier XYZ…")
        self.btn_xyz = QPushButton("📄  XYZ…"); self.btn_xyz.clicked.connect(self.choose_xyz)

        self.out_value = QLineEdit(); self.out_value.setReadOnly(True); self.out_value.setPlaceholderText("Choisir un dossier…")
        self.btn_out = QPushButton("📁 Sortie…"); self.btn_out.clicked.connect(self.choose_outdir)

        self.case_name = QLineEdit("CASE_XYZ")

        # Params principaux
        self.fmt = QLineEdit("txt")  # txt/csv
        self.skiprows = QSpinBox(); self.skiprows.setRange(0, 1_000_000); self.skiprows.setValue(1)
        self.scale = QDoubleSpinBox(); self.scale.setRange(1e-12, 1e12); self.scale.setDecimals(8); self.scale.setValue(1.0)

        self.primary_rg = QDoubleSpinBox(); self.primary_rg.setRange(1e-9, 1e9); self.primary_rg.setDecimals(6); self.primary_rg.setValue(10.0)
        self.use_ang = QCheckBox("Utiliser ANGSTROM_PER_XYZ"); self.use_ang.setChecked(True)
        self.ang_per_xyz = QDoubleSpinBox(); self.ang_per_xyz.setRange(1e-12, 1e12); self.ang_per_xyz.setDecimals(8); self.ang_per_xyz.setValue(10.0)

        self.voxel_size = QDoubleSpinBox(); self.voxel_size.setRange(1e-12, 1e12); self.voxel_size.setDecimals(12); self.voxel_size.setValue(1e-7)
        self.padding = QDoubleSpinBox(); self.padding.setRange(0.0, 1e12); self.padding.setDecimals(6); self.padding.setValue(15.0)
        self.max_voxels_total = QSpinBox(); self.max_voxels_total.setRange(1, 2_000_000_000); self.max_voxels_total.setValue(350_000_000)

        self.nb_chord = QSpinBox(); self.nb_chord.setRange(100, 5_000_000); self.nb_chord.setValue(20000)


        self.pxmin = QSpinBox(); self.pxmin.setRange(1, 500); self.pxmin.setValue(1)
        self.binw = QSpinBox(); self.binw.setRange(1, 50); self.binw.setValue(1)
        self.xmax = QSpinBox(); self.xmax.setRange(1, 100000); self.xmax.setValue(150)
        self.logy = QCheckBox("Axe Y en log"); self.logy.setChecked(True)

        self.k0 = QSpinBox(); self.k0.setRange(1, 10_000); self.k0.setValue(3)
        self.line_step = QDoubleSpinBox(); self.line_step.setRange(0.1, 100.0); self.line_step.setDecimals(3); self.line_step.setValue(1.0)

        self.show_pyvista = QCheckBox("Afficher PyVista (si pyvistaqt dispo)"); self.show_pyvista.setChecked(True)
        self.pv_max_spheres = QSpinBox(); self.pv_max_spheres.setRange(1, 2_000_000); self.pv_max_spheres.setValue(30000)
        self.pv_sphere_res = QSpinBox(); self.pv_sphere_res.setRange(3, 64); self.pv_sphere_res.setValue(10)

        # Run
        self.btn_run = QPushButton("▶  Lancer XYZ (CLD 3D + I(q))")
        self.btn_run.setObjectName("PrimaryButton")
        self.btn_run.clicked.connect(self.run_xyz)
        self.progress = QProgressBar(); self.progress.setRange(0, 100); self.progress.setValue(0)

        # Log
        self.log = QTextEdit(); self.log.setPlaceholderText("Console XYZ"); self.log.setMinimumHeight(80)

        # Figures (à droite)
        self.tabs = QTabWidget()
        self.canvas_cld = MplCanvas()
        self.canvas_iq_lin = MplCanvas()
        self.canvas_iq_log = MplCanvas()
        self.tabs.addTab(self.canvas_cld, "CLD 3D")
        self.tabs.addTab(self.canvas_iq_lin, "I(Q) lin")
        self.tabs.addTab(self.canvas_iq_log, "I(Q) log-log")

        # Left panel
        left_content = QWidget()
        left_layout = QVBoxLayout(left_content)
        left_layout.setSpacing(5)

        io_box = QGroupBox("Entrées / Sorties XYZ")
        io_grid = QGridLayout(io_box)
        io_grid.setHorizontalSpacing(5); io_grid.setVerticalSpacing(5)
        io_grid.addWidget(QLabel("XYZ"), 0, 0)
        io_grid.addWidget(self.xyz_value, 0, 1)
        io_grid.addWidget(self.btn_xyz, 0, 2)
        io_grid.addWidget(QLabel("Dossier"), 1, 0)
        io_grid.addWidget(self.out_value, 1, 1)
        io_grid.addWidget(self.btn_out, 1, 2)
        io_grid.addWidget(QLabel("Nom du cas"), 2, 0)
        io_grid.addWidget(self.case_name, 2, 1, 1, 2)

        p_box = QGroupBox("Paramètres XYZ")
        form = QFormLayout(p_box)
        form.setVerticalSpacing(4)
        form.addRow("FMT (txt/csv)", self.fmt)
        form.addRow("SKIPROWS", self.skiprows)
        form.addRow("SCALE", self.scale)
        form.addRow("PRIMARY_RG_ANGSTROM (Å)", self.primary_rg)
        form.addRow(self.use_ang)
        form.addRow("ANGSTROM_PER_XYZ", self.ang_per_xyz)
        form.addRow("VOXEL_SIZE", self.voxel_size)
        form.addRow("PADDING", self.padding)
        form.addRow("MAX_VOXELS_TOTAL", self.max_voxels_total)

        form.addRow("Nb_Chord", self.nb_chord)

        form.addRow("PX_MIN", self.pxmin)
        form.addRow("BINWIDTH_PX", self.binw)
        form.addRow("XMAX_R", self.xmax)
        form.addRow("K0", self.k0)
        form.addRow("LINE_STEP_VOX", self.line_step)
        form.addRow(self.logy)
        form.addRow(self.show_pyvista)
        form.addRow("PYVISTA_MAX_SPHERES", self.pv_max_spheres)
        form.addRow("PYVISTA_SPHERE_RES", self.pv_sphere_res)

        run_box = QGroupBox("Lancement XYZ")
        run_layout = QVBoxLayout(run_box)
        run_layout.setSpacing(6)
        run_layout.addWidget(self.btn_run)
        run_layout.addWidget(self.progress)

        log_box = QGroupBox("Console XYZ")
        log_layout = QVBoxLayout(log_box)
        log_layout.setSpacing(6)
        log_layout.addWidget(self.log)

        left_layout.addWidget(io_box)
        left_layout.addWidget(p_box)
        left_layout.addWidget(run_box)
        left_layout.addWidget(log_box, 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left_content)
        scroll.setFixedWidth(360)

        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        root.addWidget(scroll, 0)
        root.addWidget(self.tabs, 1)

    def choose_xyz(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choisir un fichier XYZ", "",
            "XYZ (*.txt *.csv *.dat);;Tous les fichiers (*.*)"
        )
        if path:
            self.xyz_path = path
            self.xyz_value.setText(path)

    def choose_outdir(self):
        d = QFileDialog.getExistingDirectory(self, "Choisir dossier de sortie", "")
        if d:
            self.out_dir = d
            self.out_value.setText(d)

    def set_busy(self, busy: bool):
        for w in [self.btn_run, self.btn_xyz, self.btn_out]:
            w.setEnabled(not busy)

    def run_xyz(self):
        if not self.xyz_path:
            QMessageBox.warning(self, "XYZ manquant", "Choisis d'abord un fichier XYZ.")
            return
        if not self.out_dir:
            QMessageBox.warning(self, "Sortie manquante", "Choisis un dossier de sortie.")
            return

        FMT = (self.fmt.text().strip().lower() or "txt")
        SKIPROWS = int(self.skiprows.value())
        SCALE = float(self.scale.value())

        PRIMARY_RG = float(self.primary_rg.value())
        ANG_PER_XYZ = float(self.ang_per_xyz.value()) if self.use_ang.isChecked() else None

        VOXEL_SIZE = float(self.voxel_size.value())
        PADDING = float(self.padding.value())
        MAX_VOX = int(self.max_voxels_total.value())

       
        Nb_Chord = int(self.nb_chord.value())

        PX_MIN = int(self.pxmin.value())
        BINWIDTH = int(self.binw.value())
        XMAX_R = int(self.xmax.value())
        PLOT_LOGY = bool(self.logy.isChecked())
        K0 = int(self.k0.value())
        STEP = float(self.line_step.value())

        CASE = (self.case_name.text().strip() or "CASE_XYZ")
        OUT = Path(self.out_dir) / CASE
        OUT.mkdir(parents=True, exist_ok=True)

        SHOW_PYVISTA = bool(self.show_pyvista.isChecked())
        PV_MAX = int(self.pv_max_spheres.value())
        PV_RES = int(self.pv_sphere_res.value())

        buf = io.StringIO()
        self.progress.setValue(0)
        self.log.clear()

        try:
            self.set_busy(True)

            with contextlib.redirect_stdout(buf):
                X = read_xyz(self.xyz_path, fmt=FMT, skiprows=SKIPROWS) * SCALE

                R_xyz, R_A, info = compute_R_in_xyz_units(
                    X_xyz=X,
                    Rg_angstrom=PRIMARY_RG,
                    angstrom_per_xyz=ANG_PER_XYZ,
                    target_fraction=0.35,
                    seed=0
                )

                print("\n===== DIAGNOSTIC =====")
                print(f"XYZ_FILE: {self.xyz_path}")
                print(f"FMT={FMT}, SKIPROWS={SKIPROWS}, SCALE={SCALE}")
                print(f"Rg primaire = {PRIMARY_RG:.6g} Å  => R_phys = {R_A:.6g} Å")
                print(f"Mode = {info['mode']} ; R_xyz = {R_xyz:.6g} (unité XYZ)")
                print(f"VOXEL_SIZE demandé = {VOXEL_SIZE}")
                print(f"PADDING utilisé = {PADDING}")
                print(f"Max voxels total = {MAX_VOX}")
                print("======================\n")

                if SHOW_PYVISTA:
                    try:
                        show_xyz_pyvista_igor_like(
                            X, R_xyz, CASE,
                            max_spheres=PV_MAX,
                            seed=0,
                            sphere_res=PV_RES
                        )
                    except Exception as e:
                        print("[PyVista] Impossible d'afficher:", e)

                self.progress.setValue(25)

                vol_blanche, origin, voxel_used, dims = voxelize_spheres_safe(
                    X, R=R_xyz, voxel=VOXEL_SIZE, padding=PADDING, max_voxels_total=MAX_VOX
                )
                vol_noire = ~vol_blanche

                print(f"[Voxel] dims={dims} ; voxel_used={voxel_used:.6g} ; total={dims[0]*dims[1]*dims[2]:,}")
                self.progress.setValue(45)

             

                lens_noire = cld_3d_from_volume(vol_noire, n_lines=Nb_Chord, px_min=PX_MIN, seed=1, step_vox=STEP)
                lens_blanche = cld_3d_from_volume(vol_blanche, n_lines=Nb_Chord, px_min=PX_MIN, seed=2, step_vox=STEP)


                L_noire = float(np.mean(lens_noire)) if lens_noire.size else np.nan
                L_blanche = float(np.mean(lens_blanche)) if lens_blanche.size else np.nan
                max_noire = int(np.max(lens_noire)) if lens_noire.size else 0
                max_blanche = int(np.max(lens_blanche)) if lens_blanche.size else 0

                phi, Sv = phi_sv_from_mean_chords(L_noire, L_blanche)

                Rn, fn = histogramme_matlab_probabilite(lens_noire, BINWIDTH)
                Rb, fb = histogramme_matlab_probabilite(lens_blanche, BINWIDTH)

                self.progress.setValue(70)

                q, Iq = np.array([]), np.array([])
                if fb.size and fn.size and np.isfinite(Sv):
                    q, Iq = modeliser_Iq_matlab_3d(fm=fb, fv=fn, SurfSpe=Sv, k0=K0)

                # ============================================================
                # EXPORTS COMPLETS (comme script original)
                # ============================================================

                # ---- Volume 3D exports
                np.save(OUT / "vol_blanche.npy", vol_blanche.astype(np.uint8))
                np.save(OUT / "vol_noire.npy", vol_noire.astype(np.uint8))

                # Stack TIFF (ZYX)
                stack = (vol_blanche.astype(np.uint8) * 255)
                stack_zyx = np.transpose(stack, (2, 1, 0))
                skio.imsave(str(OUT / "vol_blanche_stack.tif"), stack_zyx)

                # ---- Aperçu volume 3D (matplotlib voxels)
                from mpl_toolkits.mplot3d import Axes3D  # noqa
                fig3d = plt.figure(figsize=(8, 7))
                ax3d = fig3d.add_subplot(111, projection="3d")

                nx, ny, nz = vol_blanche.shape
                step = 1
                while (nx // step) * (ny // step) * (nz // step) > 2_000_000:
                    step *= 2

                v = vol_blanche[::step, ::step, ::step]
                ax3d.voxels(v, edgecolor=None)
                ax3d.set_title(f"Volume binaire 3D (downsample x{step})")

                plt.tight_layout()
                fig3d.savefig(OUT / "volume_3D.png", dpi=200)
                plt.close(fig3d)

                # ---- CLD CSV
                if Rn.size:
                    save_csv_xy(OUT / "cld_phase_noire.csv", Rn, fn, "R_voxel;f(R)_phase_noire")
                if Rb.size:
                    save_csv_xy(OUT / "cld_phase_blanche.csv", Rb, fb, "R_voxel;f(R)_phase_blanche")

                # ---- CLD_3D.png
       
                if Rn.size:
                    mn = fn > 0 if PLOT_LOGY else np.ones_like(fn, dtype=bool)
                    plt.plot(Rn[mn], fn[mn], "-o", ms=3, lw=1, label="Phase Noire")
                    
                if Rb.size:
                    mb = fb > 0 if PLOT_LOGY else np.ones_like(fb, dtype=bool)
                    plt.plot(Rb[mb], fb[mb], "-o", ms=3, lw=1, label="Phase Blanche")

                plt.xlabel("R (voxel)")
                plt.ylabel("f(R)")
                plt.xlim(PX_MIN, XMAX_R)
                plt.title(f"CLD 3D — {CASE}")
                if PLOT_LOGY:
                    plt.yscale("log")
                """plt.legend()
                plt.tight_layout()
                plt.savefig(OUT / "CLD_3D.png", dpi=200)
                plt.close()"""

                # ---- I(q)
                q_bin = np.array([])    # <--- (ajout: existe toujours)
                I_bin = np.array([])    # <--- (ajout: existe toujours)

                if q.size and Iq.size:
                    save_csv_xy(OUT / "qI.csv", q, Iq, "q;I(q)")

                    # Linear
                    fig_lin = plt.figure(figsize=(7, 5))
                    plt.plot(q, Iq, "xr", lw=2)
                    plt.xlabel("Q")
                    plt.ylabel("Intensité modélisée")
                    plt.title(f"I(Q) — {CASE}")
                    plt.tight_layout()
                    plt.savefig(OUT / "Iq_linear.png", dpi=200)
                    plt.close(fig_lin)

                    # Log-log
                    fig_log = plt.figure(figsize=(7, 5))
                    m = np.isfinite(q) & np.isfinite(Iq) & (q > 0) & (Iq > 0)
                    plt.plot(q[m], Iq[m], "xr", lw=2)
                    plt.xscale("log")
                    plt.yscale("log")
                    plt.xlabel("Q")
                    plt.ylabel("Intensité modélisée")
                    plt.title(f"I(Q) log-log — {CASE}")
                    plt.tight_layout()
                    plt.savefig(OUT / "Iq_loglog.png", dpi=200)
                    plt.close(fig_log)

                    # Log binning
                    def log_binning_qI(q, I, n_bins=60):
                        q = np.asarray(q); I = np.asarray(I)
                        m = np.isfinite(q) & np.isfinite(I) & (q > 0) & (I > 0)
                        q = q[m]; I = I[m]
                        if q.size < 2:
                            return np.array([]), np.array([])
                        edges = np.logspace(np.log10(q.min()), np.log10(q.max()), n_bins + 1)
                        idx = np.digitize(q, edges) - 1
                        qb, Ib = [], []
                        for b in range(n_bins):
                            mb = idx == b
                            if np.any(mb):
                                qb.append(np.exp(np.mean(np.log(q[mb]))))
                                Ib.append(np.median(I[mb]))
                        return np.array(qb), np.array(Ib)

                    q_bin, I_bin = log_binning_qI(q, Iq)

                    if q_bin.size:
                        fig_bin = plt.figure(figsize=(7, 5))
                        plt.plot(q, Iq, "x", ms=4, label="brut")
                        plt.plot(q_bin, I_bin, "-o", ms=4, lw=1.5, label="lissage log")
                        plt.xscale("log")
                        plt.yscale("log")
                        plt.legend()
                        plt.xlabel("Q")
                        plt.ylabel("Intensité modélisée")
                        plt.title(f"I(Q) log-log — {CASE}\n(lissage logarithmique)")
                        plt.tight_layout()
                        plt.savefig(OUT / "Iq_loglog_binned.png", dpi=200)
                        plt.close(fig_bin)

                # ---- resume.txt
                resume = [
                    "===== CLD 3D depuis XYZ + I(q) (GUI) =====",
                    f"CASE: {CASE}",
                    f"VOL shape: {vol_blanche.shape}",
                    f"N_LINES: {Nb_Chord}",
                    f"<L_noire>= {L_noire:.6g}",
                    f"<L_blanche>= {L_blanche:.6g}",
                    f"phi_noire= {phi:.6g}",
                    f"Sv= {Sv:.6g} (1/vox)",
                    f"I(q) computed: {bool(q.size and Iq.size)}",
                ]
                (OUT / "resume.txt").write_text("\n".join(resume), encoding="utf-8")

                resume = [
                    "===== CLD 3D depuis XYZ + I(q) (GUI) =====",
                    f"OUT: {OUT}",
                    f"VOL shape: {vol_blanche.shape}",
                    f"N_LINES: {Nb_Chord} ; STEP={STEP}",
                    f"<L_noire>= {L_noire:.6g} vox ; max_noire={max_noire}",
                    f"<L_blanche>= {L_blanche:.6g} vox ; max_blanche={max_blanche}",
                    f"phi_noire= {phi:.6g} ; phi_blanche={1-phi:.6g}",
                    f"Sv= {Sv:.6g} (1/vox)",
                    f"I(q) computed: {bool(q.size and Iq.size)}",
                ]
                (OUT / "resume_gui_xyz.txt").write_text("\n".join(resume), encoding="utf-8")
                print("\n".join(resume))

                # mémoriser pour affichage  (MODIF: stocker q_bin / I_bin)
                self._last_xyz = {
                    "CASE": CASE,
                    "Rn": Rn, "fn": fn,
                    "Rb": Rb, "fb": fb,
                    "phi": phi, "Sv": Sv,
                    "L_noire": L_noire, "L_blanche": L_blanche,
                    "q": q, "Iq": Iq,
                    "q_bin": q_bin,     # <--- AJOUT
                    "I_bin": I_bin,     # <--- AJOUT
                    "PX_MIN": PX_MIN, "XMAX_R": XMAX_R,
                    "PLOT_LOGY": PLOT_LOGY
                }

            txt = buf.getvalue()
            self.log.setPlainText(txt)

            last = getattr(self, "_last_xyz", None)
            if last:
                # CLD 3D
                self.canvas_cld.ax.clear()
                if last["Rn"].size:
                    mn = last["fn"] > 0 if last["PLOT_LOGY"] else np.ones_like(last["fn"], dtype=bool)
                    self.canvas_cld.ax.plot(last["Rn"][mn], last["fn"][mn], "-o", ms=3, lw=1, label="Phase Noire")
                if last["Rb"].size:
                    mb = last["fb"] > 0 if last["PLOT_LOGY"] else np.ones_like(last["fb"], dtype=bool)
                    self.canvas_cld.ax.plot(last["Rb"][mb], last["fb"][mb], "-o", ms=3, lw=1, label="Phase Blanche")

                self.canvas_cld.ax.set_xlabel("R (voxel)")
                self.canvas_cld.ax.set_ylabel("f(R) (probabilité)")
                self.canvas_cld.ax.set_title(f"CLD 3D — {last['CASE']}", fontsize=13, fontweight="bold")
                if last["PLOT_LOGY"]:
                    self.canvas_cld.ax.set_yscale("log")
                self.canvas_cld.ax.set_xlim(last["PX_MIN"], last["XMAX_R"])
                self.canvas_cld.ax.legend(loc="lower left", fontsize=10, framealpha=0.9)

                txt_box = (
                    "φ (Noire) = %.4f\n"
                    "φ (Blanche) = %.4f\n"
                    "⟨L Blanche⟩ = %.2f vox\n"
                    "⟨L Noire⟩ = %.2f vox\n"
                    "Sv = %.3e (1/vox)"
                ) % (last["phi"], 1.0 - last["phi"], last["L_blanche"], last["L_noire"], last["Sv"])

                self.canvas_cld.ax.text(
                    0.98, 0.98, txt_box,
                    transform=self.canvas_cld.ax.transAxes,
                    ha="right", va="top",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.35", alpha=0.85)
                )
                self.canvas_cld.figure.tight_layout()
                self.canvas_cld.draw()
                
              
                self.canvas_cld.figure.savefig(OUT / "CLD_3D.png", dpi=200)

                


                # I(q) lin
                self.canvas_iq_lin.ax.clear()
                q = np.asarray(last["q"], float)
                Iq = np.asarray(last["Iq"], float)
                m = np.isfinite(q) & np.isfinite(Iq)
                if np.any(m):
                    self.canvas_iq_lin.ax.plot(q[m], Iq[m], "x-", lw=1.5)
                self.canvas_iq_lin.ax.set_xlabel("Q")
                self.canvas_iq_lin.ax.set_ylabel("Intensité modélisée")
                self.canvas_iq_lin.ax.set_title(f"I(Q) — {last['CASE']}", fontsize=12)
                self.canvas_iq_lin.figure.tight_layout()
                self.canvas_iq_lin.draw()

                # I(q) log-log  (MODIF: afficher le binned si dispo)
                self.canvas_iq_log.ax.clear()

                q = np.asarray(last["q"], float)
                Iq = np.asarray(last["Iq"], float)
                q_bin = np.asarray(last.get("q_bin", np.array([])), float)
                I_bin = np.asarray(last.get("I_bin", np.array([])), float)

                if q_bin.size and I_bin.size:
                    mb = np.isfinite(q_bin) & np.isfinite(I_bin) & (q_bin > 0) & (I_bin > 0)
                    if np.any(mb):
                        self.canvas_iq_log.ax.plot(q_bin[mb], I_bin[mb], "-o", ms=4, lw=1.5, label="log-binned")

                        # brut en fond (comme dans l'image export binned)
                        m = np.isfinite(q) & np.isfinite(Iq) & (q > 0) & (Iq > 0)
                        if np.any(m):
                            self.canvas_iq_log.ax.plot(q[m], Iq[m], "x", alpha=0.25, label="brut")
                        self.canvas_iq_log.ax.legend()
                else:
                    m = np.isfinite(q) & np.isfinite(Iq) & (q > 0) & (Iq > 0)
                    if np.any(m):
                        self.canvas_iq_log.ax.plot(q[m], Iq[m], "x-", lw=1.5)

                self.canvas_iq_log.ax.set_xscale("log")
                self.canvas_iq_log.ax.set_yscale("log")
                self.canvas_iq_log.ax.set_xlabel("Q")
                self.canvas_iq_log.ax.set_ylabel("Intensité modélisée")
                self.canvas_iq_log.ax.set_title(f"I(Q) log-log — {last['CASE']}", fontsize=12)
                self.canvas_iq_log.figure.tight_layout()
                self.canvas_iq_log.draw()

            self.progress.setValue(100)
            self.set_busy(False)

        except Exception as e:
            self.set_busy(False)
            QMessageBox.critical(self, "Erreur XYZ", str(e))


# =============================================================================
# Main Window (interface complète)
# =============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CLD 2D + XYZ→CLD 3D + I(q) — PyQt5")
        #self.setMinimumSize(1280, 740)
        
        self.setMinimumSize(1280, 880)
        self.resize(1600, 920)   # taille initiale (pratique)


        self.main_tabs = QTabWidget()
        self.tab2d = CLD2DTab(self)
        self.tabxyz = XYZTab(self)

        self.main_tabs.addTab(self.tab2d, "Image 2D")
        self.main_tabs.addTab(self.tabxyz, "XYZ → CLD 3D")

        self.setCentralWidget(self.main_tabs)
        self.statusBar().showMessage("Prêt.")
    


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_QSS)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
