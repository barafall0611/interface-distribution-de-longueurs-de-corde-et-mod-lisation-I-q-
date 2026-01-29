# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 15:33:04 2026

@author: bara.fall
"""

# -*- coding: utf-8 -*-

import sys
import io
import contextlib
from pathlib import Path
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout, QGroupBox,
    QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QPushButton, QTextEdit,
    QLabel, QTabWidget, QProgressBar, QScrollArea
)

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from skimage import io as skio
from skimage.color import rgb2gray
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.draw import line as sk_line


# =============================================================================
# STYLE (titres visibles + compact)
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

QGroupBox::title { letter-spacing: 0.2px; }

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
  background: #10131c; border: 1px solid #22283a; border-bottom: none;
  padding: 7px 10px; margin-right: 6px; border-top-left-radius: 9px;
  border-top-right-radius: 9px; color: #cfd3dd; font-weight: 800;
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
# OUTILS IMAGE
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
# CLD – coeur
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
# HISTOGRAMME / I(q) / METRIQUES
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
# Export helpers
# =============================================================================
def save_csv_xy(path: Path, x: np.ndarray, y: np.ndarray, header: str):
    data = np.column_stack([x, y])
    np.savetxt(path, data, delimiter=";", header=header, comments="", fmt="%.10g")

def save_image_u8(path: Path, img_float01: np.ndarray):
    arr = np.clip(img_float01 * 255, 0, 255).astype(np.uint8)
    skio.imsave(str(path), arr)

def save_bool_png(path: Path, b: np.ndarray):
    skio.imsave(str(path), (b.astype(np.uint8) * 255))


# =============================================================================
# CLD FIGURE : STYLE FIXE
# =============================================================================
def build_cld_figure(res, case_name, *, plot_logy, px_min, xmax_r):
    fig = Figure(figsize=(7, 5), dpi=160)
    ax = fig.add_subplot(111)

    if res["Rp"].size:
        mp = res["fp"] > 0 if plot_logy else np.ones_like(res["fp"], dtype=bool)
        ax.plot(res["Rp"][mp], res["fp"][mp], "-o", markersize=3, linewidth=1, label="Pores")
    if res["Rm"].size:
        mm = res["fm"] > 0 if plot_logy else np.ones_like(res["fm"], dtype=bool)
        ax.plot(res["Rm"][mm], res["fm"][mm], "-o", markersize=3, linewidth=1, label="Matière")

    ax.set_xlabel("R (pixel)")
    ax.set_ylabel("f(R) (probabilité)")
    ax.set_title(f"CLD — {case_name}", fontsize=14, fontweight="bold")

    if plot_logy:
        ax.set_yscale("log")
    ax.set_xlim(px_min, xmax_r)

    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)

    txt = (
        "φ (porosité) = %.4f\n"
        "⟨Lm⟩ = %.2f px (max=%d)\n"
        "⟨Lp⟩ = %.2f px (max=%d)\n"
        "Sv (surface spécifique) = %.3e px⁻¹"
    ) % (
        res["phi"],
        res["Lm_px"], res["max_m"],
        res["Lp_px"], res["max_p"],
        res["SurfSpe_1_per_px"]
    )

    ax.text(
        0.98, 0.98, txt,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=10,
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
    ax.set_title(f"I(Q) (log-log) — {case_name}", fontsize=12, fontweight="bold")

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

    # Comme ton affichage : commencer après le 1er point (optionnel)
    if q.size > 1:
        q = q[1:]
        I = I[1:]

    ax.plot(q, I, "xr", linewidth=2)
    ax.set_xlabel("Q", fontsize=14, fontweight="bold")
    ax.set_ylabel("Intensité modélisée", fontsize=14, fontweight="bold")
    ax.set_title("Modélisation de l'intensité en fonction de Q", fontsize=16, fontweight="bold")

    fig.tight_layout()
    return fig

# =============================================================================
# EXECUTER CAS
# =============================================================================
def executer_cas(
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
        q, I = modeliser_Iq_matlab(fm=fm, fv=fp, SurfSpe=SurfSpe_for_I, k0=1)

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
# UI Canvas
# =============================================================================
class MplCanvas(FigureCanvas):
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(7, 5), dpi=100)
        super().__init__(self.fig)


# =============================================================================
# MAIN WINDOW
# =============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("distribution de longueurs de cordes + modelisation I(q) — PyQt5")
        self.setMinimumSize(1280, 740)

        self.img_path = ""
        self.out_dir = ""

        # IO
        self.path_value = QLineEdit(); self.path_value.setReadOnly(True); self.path_value.setPlaceholderText("Choisir une image…")
        self.btn_image = QPushButton("📷  Image…"); self.btn_image.clicked.connect(self.choose_image)

        self.out_value = QLineEdit(); self.out_value.setReadOnly(True); self.out_value.setPlaceholderText("Choisir un dossier…")
        self.btn_out = QPushButton("📁 Sortie…"); self.btn_out.clicked.connect(self.choose_outdir)

        self.prefix = QLineEdit("resultats")

        # Params
        self.nlines = QSpinBox(); self.nlines.setRange(100, 5_000_000); self.nlines.setValue(60000)
        self.thr = QDoubleSpinBox(); self.thr.setRange(0.0, 255.0); self.thr.setDecimals(1); self.thr.setValue(125.7)
        self.pxmin = QSpinBox(); self.pxmin.setRange(1, 500); self.pxmin.setValue(1)
        self.binw = QSpinBox(); self.binw.setRange(1, 50); self.binw.setValue(1)
        self.callen = QDoubleSpinBox(); self.callen.setRange(1e-9, 1e9); self.callen.setDecimals(6); self.callen.setValue(1.0)
        self.minobj = QSpinBox(); self.minobj.setRange(1, 1000); self.minobj.setValue(5)
        self.logy = QCheckBox("Axe Y en log"); self.logy.setChecked(True)
        self.xmax = QSpinBox(); self.xmax.setRange(1, 100000); self.xmax.setValue(150)
        self.export_bin = QCheckBox("Exporter les binaires"); self.export_bin.setChecked(True)

        # Run
        self.btn_run = QPushButton("▶  Lancer (avec les 4 cas) + Export")
        self.btn_run.setObjectName("PrimaryButton")
        self.btn_run.clicked.connect(self.run)
        self.progress = QProgressBar(); self.progress.setRange(0, 100); self.progress.setValue(0)

        # Log
        self.log = QTextEdit(); self.log.setPlaceholderText("Console")
        self.log.setMinimumHeight(80)

        # Tabs (SANS Images)
        self.tabs = QTabWidget()
        self.canvas_cld = MplCanvas()
        self.canvas_iq = MplCanvas()
        self.canvas_iq_lin = MplCanvas()


        # message au démarrage
        self.canvas_cld.ax.text(0.5, 0.5, "Sélectionne une image puis clique sur Lancer",
                                transform=self.canvas_cld.ax.transAxes,
                                ha="center", va="center", fontsize=12, alpha=0.5)
        self.canvas_cld.draw()

        self.tabs.addTab(self.canvas_cld, "CLD (dernier cas)")
        self.tabs.addTab(self.canvas_iq_lin, "I(Q) (lin) (dernier cas)")
        self.tabs.addTab(self.canvas_iq, "I(Q) log-log (dernier cas)")


        self.statusBar().showMessage("Prêt.")

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
        io_grid.addWidget(QLabel("Préfixe"), 2, 0)
        io_grid.addWidget(self.prefix, 2, 1, 1, 2)

        p_box = QGroupBox("Paramètres")
        form = QFormLayout(p_box)
        form.setVerticalSpacing(4)
        form.setHorizontalSpacing(10)
        form.addRow("N_LINES", self.nlines)
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
        scroll.setFixedWidth(320)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)
        right_layout.addWidget(self.tabs, 1)

        main = QWidget()
        main_layout = QHBoxLayout(main)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)
        main_layout.addWidget(scroll, 0)
        main_layout.addWidget(right, 1)
        self.setCentralWidget(main)

    def choose_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choisir une image", "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;Tous les fichiers (*.*)"
        )
        if path:
            self.img_path = path
            self.path_value.setText(path)
            self.statusBar().showMessage("Image sélectionnée.")

    def choose_outdir(self):
        d = QFileDialog.getExistingDirectory(self, "Choisir dossier de sortie", "")
        if d:
            self.out_dir = d
            self.out_value.setText(d)
            self.statusBar().showMessage("Dossier de sortie sélectionné.")

    def set_busy(self, busy: bool, msg: str = ""):
        for w in [self.btn_run, self.btn_image, self.btn_out]:
            w.setEnabled(not busy)
        if msg:
            self.statusBar().showMessage(msg)

    def run(self):
        if not self.img_path:
            QMessageBox.warning(self, "Image manquante", "Choisis d'abord une image.")
            return
        if not self.out_dir:
            QMessageBox.warning(self, "Sortie manquante", "Choisis un dossier de sortie.")
            return

        N_LINES = int(self.nlines.value())
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
            self.set_busy(True, "Exécution en cours…")
            self.progress.setValue(0)
            self.log.clear()

            with contextlib.redirect_stdout(buf):
                img_gray = lire_en_gris(self.img_path)
                binary0 = binariser_image(img_gray, thr_8bit=thr_8bit)

                save_image_u8(out_dir / "img_gray.png", img_gray)
                save_bool_png(out_dir / "binary0.png", binary0)

                cases = [
                    ("correction pixel+bord", True,  True,  1, 2),
                    ("Pas de correction",     False, False, 3, 4),
                    ("correction des bords",  False, True,  5, 6),
                    ("correction pixels isolés", True, False, 7, 8),
                ]

                last = None

                for i, (name, do_iso, do_bord, sp, sm) in enumerate(cases, start=1):
                    res = executer_cas(
                        name, img_gray, binary0,
                        do_isolated=do_iso, do_border=do_bord,
                        N_LINES=N_LINES, PX_MIN=PX_MIN, BINWIDTH_PX=BINWIDTH_PX,
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

            self.progress.setValue(100)
            self.set_busy(False, f"Terminé ✅ Export: {out_dir}")

            # UI update (dernier cas)
            if last is not None:
                # CLD UI
                self.canvas_cld.ax.clear()
                if last["Rp"].size:
                    mp = last["fp"] > 0 if PLOT_LOGY else np.ones_like(last["fp"], dtype=bool)
                    self.canvas_cld.ax.plot(last["Rp"][mp], last["fp"][mp], "-o", markersize=3, linewidth=1, label="Pores")
                if last["Rm"].size:
                    mm = last["fm"] > 0 if PLOT_LOGY else np.ones_like(last["fm"], dtype=bool)
                    self.canvas_cld.ax.plot(last["Rm"][mm], last["fm"][mm], "-o", markersize=3, linewidth=1, label="Matière")

                self.canvas_cld.ax.set_xlabel("R (pixel)")
                self.canvas_cld.ax.set_ylabel("f(R) (probabilité)")
                self.canvas_cld.ax.set_title(f"CLD — {last['case']}", fontsize=14, fontweight="bold")
                if PLOT_LOGY:
                    self.canvas_cld.ax.set_yscale("log")
                self.canvas_cld.ax.set_xlim(PX_MIN, XMAX_R_PX)
                self.canvas_cld.ax.legend(loc="lower left", fontsize=10, framealpha=0.9)

                txt = (
                    "φ (porosité) = %.4f\n"
                    "⟨Lm⟩ = %.2f px (max=%d)\n"
                    "⟨Lp⟩ = %.2f px (max=%d)\n"
                    "Sv (surface spécifique) = %.3e px⁻¹"
                ) % (
                    last["phi"],
                    last["Lm_px"], last["max_m"],
                    last["Lp_px"], last["max_p"],
                    last["SurfSpe_1_per_px"]
                )
                self.canvas_cld.ax.text(
                    0.98, 0.98, txt,
                    transform=self.canvas_cld.ax.transAxes,
                    ha="right", va="top",
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.35", alpha=0.85)
                )
                self.canvas_cld.fig.tight_layout()
                self.canvas_cld.draw()
                
                
                # I(q) UI — linéaire (équivalent à ton plt.figure + plt.plot)
                self.canvas_iq_lin.ax.clear()
                q = np.asarray(last["q"], dtype=float)
                I = np.asarray(last["I"], dtype=float)
                
                mask = np.isfinite(q) & np.isfinite(I)
                q2 = q[mask]
                I2 = I[mask]
                
                # (optionnel) enlever le premier point comme tu l'as fait pour log-log
                if q2.size > 1:
                    q2 = q2[1:]
                    I2 = I2[1:]
                
                self.canvas_iq_lin.ax.plot(q2, I2, "xr", linewidth=2)
                self.canvas_iq_lin.ax.set_xlabel("Q", fontsize=14, fontweight="bold")
                self.canvas_iq_lin.ax.set_ylabel("Intensité modélisée", fontsize=14, fontweight="bold")
                self.canvas_iq_lin.ax.set_title("Modélisation de l'intensité en fonction de Q",
                                                fontsize=16, fontweight="bold")
                self.canvas_iq_lin.fig.tight_layout()
                self.canvas_iq_lin.draw()


                # Iq UI
                self.canvas_iq.ax.clear()
                q = np.asarray(last["q"], dtype=float)
                I = np.asarray(last["I"], dtype=float)
                mask = np.isfinite(q) & np.isfinite(I) & (q > 0) & (I > 0)
                q = q[mask]; I = I[mask]
                if q.size > 1:
                    q = q[1:]; I = I[1:]
                if q.size:
                    self.canvas_iq.ax.plot(q, I, "x-", linewidth=1.5)
                    self.canvas_iq.ax.set_xscale("log")
                    self.canvas_iq.ax.set_yscale("log")
                self.canvas_iq.ax.set_xlabel("Q")
                self.canvas_iq.ax.set_ylabel("Intensité modélisée")
                self.canvas_iq.ax.set_title(f"I(Q) (log-log) — {last['case']}")
                self.canvas_iq.fig.tight_layout()
                self.canvas_iq.draw()

        except Exception as e:
            self.set_busy(False, "Erreur.")
            QMessageBox.critical(self, "Erreur", str(e))


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_QSS)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
