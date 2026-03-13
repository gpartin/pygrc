"""
Copyright (c) 2026 G. Partin. MIT License.
Author: G. Partin, Date: March 2026

LFM-RAR rotation curve model for SPARC data.

Predicts galaxy rotation curves from baryonic mass using the radial
acceleration relation derived from the Lattice Field Medium (LFM)
governing equations.

Derivation summary (6 steps):
    1. GOV-04 quasi-static limit gives Newtonian gravity.
    2. GOV-01 metric interpretation gives g_obs = g_bar * (chi_0 / chi).
    3. Chi-memory timescale gives a0 = c * H0 / (2*pi) ~ 1.04e-10 m/s^2.
    4. Newtonian limit (g_bar >> a0): g_obs -> g_bar.
    5. Deep-field limit (g_bar << a0): g_obs -> sqrt(g_bar * a0).
    6. Unique simplest interpolation: g_obs^2 = g_bar^2 + g_bar * a0.

References:
    McGaugh, Lelli & Schombert (2016), PRL 117, 201101
    Lelli, McGaugh & Schombert (2017), ApJ 836, 152
    Milgrom (1983), ApJ 270, 365
    Partin (2026), LFM-PAPER-045, Section 4.5
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typing as tp

# LFM fundamental constants
CHI_0 = 19  # lattice vacuum stiffness: 3^3 - 2^3
KAPPA = 1 / 63  # chi-energy coupling: 1/(4^3 - 1)

# Physical constants
_C = 2.998e8  # speed of light [m/s]
_H0 = 2.184e-18  # Hubble constant: 67.4 km/s/Mpc in [s^-1]
_KPC = 3.086e19  # 1 kpc [m]
_KM = 1.0e3  # 1 km [m]

# Derived acceleration scale (zero free parameters)  [Step 3]
# GOV-03 chi-memory timescale tau -> Hubble time -> a0 = c * H0 / (2*pi)
A0 = _C * _H0 / (2 * np.pi)  # ~1.04e-10 m/s^2

# Default parameters
UPS_DISK = 0.50  # stellar disk M/L [M_sun/L_sun]
UPS_BULGE = 0.70  # bulge M/L [M_sun/L_sun]


def v_baryonic(
    data: pd.DataFrame,
    ups_disk: float = UPS_DISK,
    ups_bulge: float = UPS_BULGE,
):
    """
    Quadrature sum of SPARC baryonic velocity components.

    Args:
        data: SPARC DataFrame with Vgas, Vdisk, Vbul columns
        ups_disk: stellar disk mass-to-light ratio
        ups_bulge: bulge mass-to-light ratio

    Returns:
        numpy array of baryonic velocities in km/s
    """
    return np.sqrt(
        data["Vgas"].values ** 2
        + ups_disk * data["Vdisk"].values ** 2
        + ups_bulge * data["Vbul"].values ** 2
    )


def lfm_rar_velocity(
    r_kpc: np.ndarray,
    v_bar_kms: np.ndarray,
    a0: float = A0,
):
    """
    Predict rotation velocity from baryonic velocity via LFM-RAR.

    From the product structure (Step 2):
        g_obs = g_bar * chi_0 / chi(r)
    Combined with the unique simplest interpolation (Step 6):
        (chi_0/chi)^2 = 1 + a0/g_bar
    Gives:
        g_obs = sqrt(g_bar^2 + g_bar * a0)
        v_pred = sqrt(g_obs * r)

    Args:
        r_kpc: radii in kpc
        v_bar_kms: baryonic velocity in km/s
        a0: acceleration scale in m/s^2

    Returns:
        predicted rotation velocity in km/s
    """
    r_m = r_kpc * _KPC
    v_bar_ms = v_bar_kms * _KM
    r_safe = np.maximum(r_m, 1.0)
    g_bar = v_bar_ms ** 2 / r_safe
    # Step 6: g_obs^2 = g_bar^2 + g_bar * a0  (unique simplest interpolation)
    g_obs = np.sqrt(g_bar ** 2 + g_bar * a0)
    return np.sqrt(g_obs * r_safe) / _KM


class LFM:
    """
    LFM-RAR model for galaxy rotation curves.

    Predicts observed rotation velocity from baryonic mass distribution
    using the Lattice Field Medium Radial Acceleration Relation.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        name: str = "Galaxy",
        ups_disk: float = UPS_DISK,
        ups_bulge: float = UPS_BULGE,
    ):
        """
        Args:
            data: SPARC DataFrame with Rad, Vobs, errV, Vgas, Vdisk, Vbul
            name: galaxy name
            ups_disk: stellar disk mass-to-light ratio
            ups_bulge: bulge mass-to-light ratio
        """
        self.data = data.copy()
        self.name = name
        self.ups_disk = ups_disk
        self.ups_bulge = ups_bulge
        self.r = data["Rad"].values
        self.v_obs = data["Vobs"].values
        self.err_v = data["errV"].values
        self.v_bar = v_baryonic(data, ups_disk, ups_bulge)
        self._v_pred = None

    def predict(self):
        """
        Run LFM-RAR prediction with current parameters.

        Returns:
            numpy array of predicted velocities in km/s
        """
        self._v_pred = lfm_rar_velocity(self.r, self.v_bar)
        return self._v_pred

    def calibrate(self, ups_bounds: tp.Tuple[float, float] = (0.1, 1.5)):
        """
        Fit Upsilon_disk per galaxy (a0 stays fixed).

        Args:
            ups_bounds: search bounds for Upsilon_disk

        Returns:
            optimal Upsilon_disk value
        """
        v_obs = self.v_obs
        vgas = self.data["Vgas"].values
        vdisk = self.data["Vdisk"].values
        vbul = self.data["Vbul"].values
        r = self.r

        grid = np.linspace(ups_bounds[0], ups_bounds[1], 1000)
        residuals = np.array([
            np.sum((v_obs - lfm_rar_velocity(
                r, np.sqrt(vgas ** 2 + u * vdisk ** 2
                           + self.ups_bulge * vbul ** 2)
            )) ** 2) for u in grid
        ])
        best = grid[np.argmin(residuals)]
        self.ups_disk = best
        self.v_bar = v_baryonic(self.data, best, self.ups_bulge)
        self._v_pred = lfm_rar_velocity(r, self.v_bar)
        return best

    def r_squared(self):
        """Coefficient of determination (R^2)."""
        if self._v_pred is None:
            self.predict()
        ss_res = np.sum((self.v_obs - self._v_pred) ** 2)
        ss_tot = np.sum((self.v_obs - np.mean(self.v_obs)) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    def rmse(self):
        """Root mean squared error in km/s."""
        if self._v_pred is None:
            self.predict()
        return float(np.sqrt(np.mean((self.v_obs - self._v_pred) ** 2)))

    def chi2_reduced(self):
        """Reduced chi-squared."""
        if self._v_pred is None:
            self.predict()
        err = np.maximum(self.err_v, 1.0)
        return float(
            np.sum(((self.v_obs - self._v_pred) / err) ** 2) / len(self.v_obs)
        )

    def summary(self):
        """
        Returns:
            dict with name, N, R2, RMSE, chi2r keys
        """
        if self._v_pred is None:
            self.predict()
        return {
            "name": self.name,
            "N": len(self.r),
            "R2": self.r_squared(),
            "RMSE": self.rmse(),
            "chi2r": self.chi2_reduced(),
        }

    def plot(self, ax=None):
        """
        Plot observed vs predicted rotation curve.

        Args:
            ax: matplotlib axes (created if None)

        Returns:
            matplotlib axes
        """
        if self._v_pred is None:
            self.predict()

        if ax is None:
            _, ax = plt.subplots()
        ax.errorbar(
            self.r,
            self.v_obs,
            yerr=self.err_v,
            fmt="o",
            ms=4,
            color="black",
            label="Data",
        )
        ax.plot(self.r, self.v_bar, ":", color="gray", lw=1.2, label="Baryonic")
        ax.plot(
            self.r,
            self._v_pred,
            "--",
            color="tab:blue",
            label="LFM-RAR  R2={:.3f}".format(self.r_squared()),
        )
        ax.set_xlabel("Distance (kpc)")
        ax.set_ylabel("Velocity (km/s)")
        ax.set_title(self.name)
        ax.legend()
        return ax
