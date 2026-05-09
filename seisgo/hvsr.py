"""
hvsr.py  –  Horizontal-to-Vertical Spectral Ratio (HVSR) module for SeisGo
===========================================================================

Computes HVSR of seismic ambient-noise waveforms using six methods following
Bahavar et al. (2020, SRL) and the IRIS HVSR Station Toolbox paper.

Methods
-------
M1 – Diffuse Field Assumption (DFA), Sánchez-Sesma et al. (2011)
     HVSR = sqrt( (E1 + E2) / E3 )  where Ei = mean PSD of component i
M2 – Average of spectral ratios, horizontal = arithmetic mean: (HN + HE) / 2
M3 – Average of spectral ratios, horizontal = geometric mean: sqrt(HN * HE)
M4 – Average of spectral ratios, horizontal = vector summation: sqrt(HN²+HE²)
M5 – Average of spectral ratios, horizontal = quadratic mean: M4 / sqrt(2)
M6 – Average of spectral ratios, horizontal = maximum: max(HN, HE)

The main user-facing function is :func:`compute_hvsr`.  All other functions
are helpers that can also be used independently.

References
----------
Bahavar, M., Spica, Z. J., Sánchez-Sesma, F. J., Trabant, C., Zandieh, A., 
    & Toro, G. (2020). Horizontal-to-Vertical Spectral Ratio (HVSR) IRIS Station 
    Toolbox. Seismological Research Letters, 91(6), 3539–3549. 
    https://doi.org/10.1785/0220200047
Bonnefoy-Claudet, S., Köhler, A., Cornou, C., Wathelet, M., & Bard, P. Y. (2024). 
    Guidelines for implementation of the H/V spectral ratio technique on ambient 
    vibrations measurements, processing and interpretation. SESAME European research 
    project WP12 - Deliverable D23.12. European Commission - Research General Directorate. 
    Project No. EVG1-CT-2000-00026 SESAME. Bulletin of the Seismological Society of America, 
    98(1), 288–300. https://doi.org/10.1785/0120070063
McNamara, D.E., Stephenson, W.J., Odum, J.K., Williams, R.A., and Gee, L., 2015, 
    Site response in the eastern United States: A comparison of Vs30 measurements with 
    estimates from horizontal:vertical spectral ratios, in Horton, J.W., Jr., Chapman, M.C., 
    and Green, R.A., eds., The 2011 Mineral, Virginia, Earthquake,and Its Signifi cance 
    for Seismic Hazards in Eastern North America: Geological Society of America Special 
    Paper 509, p. 67–79, doi:10.1130/2015.2509(04)
Sánchez-Sesma, F. J., Rodríguez, M., Iturrarán-Viveros, U., Luzón, F., Campillo, M., 
    Margerin, L., García-Jerez, A., Suarez, M., Santoyo, M. A., & Rodríguez-Castellanos, 
    A. (2011). A theory for microtremor H/V spectral ratio: application for a layered 
    medium. Geophysical Journal International, 186(1), 221–225. 
    https://doi.org/10.1111/j.1365-246X.2011.05064.x

Author
------
Drafted by Claude AI, QCed and edited by Xiaotian Yang to ensure accurray and consistency with the rest of SeisGo.
Contributed to SeisGo (https://github.com/xtyangpsp/SeisGo)


FUTURE ENHANCEMENTS
-------------------

"""
import warnings,os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pyasdf
from obspy import Stream
# SeisGo internal helpers – reuse to stay consistent with the rest of the package
from seisgo.utils import (
    psd as seisgo_psd,
    smooth as seisgo_smooth,
    detrend,
    demean,
    taper,
    sliding_window,
    extract_waveform,
    sta_info_from_inv
)
from seisgo.types import HVSRData

# ──────────────────────────────────────────────────────────────────────────────
# Public constants
# ──────────────────────────────────────────────────────────────────────────────

METHODS = {
    1: "DFA (diffuse field assumption)",
    2: "Average-of-ratios: arithmetic mean horizontal",
    3: "Average-of-ratios: geometric mean horizontal",
    4: "Average-of-ratios: vector summation horizontal",
    5: "Average-of-ratios: quadratic mean horizontal",
    6: "Average-of-ratios: maximum horizontal",
}

# SESAME peak-ranking frequency-dependent thresholds (Bard & SESAME 2004, Table 2)
_SESAME_FREQ_THRESHOLDS = [
    (0.2,  0.25, 0.48),
    (0.5,  0.20, 0.40),
    (1.0,  0.15, 0.30),
    (2.0,  0.10, 0.25),
    (np.inf, 0.05, 0.20),
]

# ──────────────────────────────────────────────────────────────────────────────
# Low-level PSD helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_window_psds(trace, win_len_s, step_s=None, taper_frac=0.05,
                        smooth_n=None):
    """
    Compute PSDs for overlapping windows of a single :class:`obspy.Trace`.

    Uses :func:`seisgo.utils.psd` so spectral computation is consistent with
    the rest of SeisGo.

    Parameters
    ----------
    trace : obspy.Trace
        Single-component seismic trace.
    win_len_s : float
        Window length in seconds.
    step_s : float, optional
        Step between windows in seconds.  Defaults to ``win_len_s`` (no
        overlap).
    taper_frac : float
        Fraction of each window to taper (cosine). Default 0.05.
    smooth_n : int, optional
        Number of frequency samples over which to smooth the PSDs via a
        boxcar convolution (:func:`seisgo.utils.smooth`).  ``None`` = no
        smoothing.

    Returns
    -------
    freqs : ndarray, shape (Nf,)
        Frequency axis in Hz.
    psds : ndarray, shape (Nwindows, Nf)
        Power spectral densities for every window.
    """
    fs = trace.stats.sampling_rate
    data = trace.data.copy().astype(np.float64)
    data = detrend(demean(data))

    ws = int(win_len_s * fs)
    if step_s is None:
        ss = ws
    else:
        ss = int(step_s * fs)

    # Build sliding windows using SeisGo's helper
    windows, n_windows = sliding_window(data, ws, ss)
    if n_windows == 0:
        raise ValueError(
            "Trace is shorter than the requested window length "
            f"({trace.stats.npts/fs:.1f} s < {win_len_s} s)."
        )

    psds = []
    for i in range(n_windows):
        seg = windows[i].copy()
        seg = taper(seg, fraction=taper_frac)
        f, p = seisgo_psd(seg, fs)
        psds.append(p)

    psds = np.array(psds)   # shape (n_windows, Nf)

    if smooth_n is not None and smooth_n > 1:
        psds = seisgo_smooth(psds, smooth_n, axis=1)

    return f, psds


def median_psd(freqs, psds):
    """
    Compute the median PSD across all windows.

    Parameters
    ----------
    freqs : ndarray, shape (Nf,)
    psds : ndarray, shape (Nwindows, Nf)

    Returns
    -------
    freqs : ndarray, shape (Nf,)
    median_psd : ndarray, shape (Nf,)
    """
    return freqs, np.median(psds, axis=0)


def mean_psd(freqs, psds):
    """
    Compute the mean PSD across all windows.

    Parameters
    ----------
    freqs : ndarray, shape (Nf,)
    psds : ndarray, shape (Nwindows, Nf)

    Returns
    -------
    freqs : ndarray, shape (Nf,)
    mean_psd : ndarray, shape (Nf,)
    """
    return freqs, np.mean(psds, axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Component extraction helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_components(stream):
    """
    Extract Z, N/1, E/2 traces from an :class:`obspy.Stream`.

    The function accepts standard SEED channel naming (BHZ/HHZ/…) and also
    OBS-style 1/2 component codes.

    Parameters
    ----------
    stream : obspy.Stream
        Must contain exactly one vertical and two (orthogonal) horizontal
        traces.

    Returns
    -------
    tr_z, tr_n, tr_e : obspy.Trace
        Vertical, north-ish, and east-ish component traces.

    Raises
    ------
    ValueError
        If the required components cannot be uniquely identified.
    """
    z_candidates = [tr for tr in stream
                    if tr.stats.channel[-1].upper() in ('Z',)]
    n_candidates = [tr for tr in stream
                    if tr.stats.channel[-1].upper() in ('N', '1')]
    e_candidates = [tr for tr in stream
                    if tr.stats.channel[-1].upper() in ('E', '2')]

    if len(z_candidates) != 1:
        raise ValueError(
            f"Expected exactly 1 vertical (Z) trace, found {len(z_candidates)}."
        )
    if len(n_candidates) != 1:
        raise ValueError(
            f"Expected exactly 1 N/1 trace, found {len(n_candidates)}."
        )
    if len(e_candidates) != 1:
        raise ValueError(
            f"Expected exactly 1 E/2 trace, found {len(e_candidates)}."
        )

    return z_candidates[0], n_candidates[0], e_candidates[0]


# ──────────────────────────────────────────────────────────────────────────────
# HVSR computation methods (M1 – M6)
# ──────────────────────────────────────────────────────────────────────────────

def _hvsr_m1(psd_z, psd_n, psd_e):
    """
    Method 1 – Diffuse Field Assumption (DFA).

    HVSR = sqrt( (E_N + E_E) / E_Z )

    where E_i is the component-wise *averaged* PSD (not the ratio of window
    averages). This is the formulation of Sánchez-Sesma et al. (2011) as
    applied in Bahavar et al. (2020).

    Parameters
    ----------
    psd_z, psd_n, psd_e : ndarray, shape (Nf,)
        Mean PSDs for the vertical, north, and east components.

    Returns
    -------
    hvsr : ndarray, shape (Nf,)
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        hvsr = np.sqrt((psd_n + psd_e) / psd_z)
    return hvsr


def _horizontal_mean_m2(amp_n, amp_e):
    """Arithmetic mean horizontal amplitude spectrum."""
    return (amp_n + amp_e) / 2.0


def _horizontal_mean_m3(amp_n, amp_e):
    """Geometric mean horizontal amplitude spectrum."""
    return np.sqrt(amp_n * amp_e)


def _horizontal_mean_m4(amp_n, amp_e):
    """Vector-summation horizontal amplitude spectrum."""
    return np.sqrt(amp_n**2 + amp_e**2)


def _horizontal_mean_m5(amp_n, amp_e):
    """Quadratic-mean horizontal amplitude spectrum (= M4 / sqrt(2))."""
    return _horizontal_mean_m4(amp_n, amp_e) / np.sqrt(2.0)


def _horizontal_mean_m6(amp_n, amp_e):
    """Maximum horizontal amplitude spectrum."""
    return np.maximum(amp_n, amp_e)


_H_MERGERS = {
    2: _horizontal_mean_m2,
    3: _horizontal_mean_m3,
    4: _horizontal_mean_m4,
    5: _horizontal_mean_m5,
    6: _horizontal_mean_m6,
}


def _hvsr_avg_of_ratios(psds_z, psds_n, psds_e, method):
    """
    Methods 2–6 – Average of spectral ratios.

    For each window compute H̄/HZ, then average across windows.

    Parameters
    ----------
    psds_z, psds_n, psds_e : ndarray, shape (Nwindows, Nf)
        Per-window PSD arrays for the three components.
    method : int
        2, 3, 4, 5, or 6.

    Returns
    -------
    hvsr : ndarray, shape (Nf,)
        Mean HVSR across windows.
    hvsr_std : ndarray, shape (Nf,)
        Standard deviation of the per-window HVSR.
    """
    merge = _H_MERGERS[method]

    # Convert PSD (power) to amplitude spectrum (square-root of PSD)
    amp_n = np.sqrt(np.abs(psds_n))
    amp_e = np.sqrt(np.abs(psds_e))
    amp_z = np.sqrt(np.abs(psds_z))

    h_bar = merge(amp_n, amp_e)   # shape (Nwindows, Nf)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = h_bar / amp_z        # shape (Nwindows, Nf)

    hvsr = np.nanmean(ratios, axis=0)
    hvsr_std = np.nanstd(ratios, axis=0)
    return hvsr, hvsr_std


# ──────────────────────────────────────────────────────────────────────────────
# Peak detection and SESAME-style ranking
# ──────────────────────────────────────────────────────────────────────────────

def _sesame_thresholds(f0):
    """
    Return the SESAME frequency-stability threshold εf0 and log-amplitude
    threshold logθf0 for the given peak frequency.

    Parameters
    ----------
    f0 : float  –  Peak frequency (Hz)

    Returns
    -------
    eps_f, log_theta : float, float
    """
    for flim, eps, log_theta in _SESAME_FREQ_THRESHOLDS:
        if f0 < flim:
            return eps, log_theta
    return 0.05, 0.20   # fallback (f0 ≥ 2 Hz)


def _interpolate_threshold_crossing(f1, a1, f2, a2, target):
    """Linearly interpolate the frequency where the amplitude crosses target."""
    if a2 == a1:
        return f1
    return f1 + (target - a1) * (f2 - f1) / (a2 - a1)


def _half_power_bounds(freqs, hvsr, idx, threshold):
    """Find the frequency bounds where the peak crosses the half-power level."""
    n = len(freqs)
    f_lo = np.nan
    f_hi = np.nan

    # Left side
    for i in range(idx, 0, -1):
        if hvsr[i] >= threshold and hvsr[i - 1] < threshold:
            f_lo = _interpolate_threshold_crossing(
                freqs[i - 1], hvsr[i - 1], freqs[i], hvsr[i], threshold
            )
            break
        if i == 1 and hvsr[0] >= threshold:
            f_lo = freqs[0]

    # Right side
    for i in range(idx, n - 1):
        if hvsr[i] >= threshold and hvsr[i + 1] < threshold:
            f_hi = _interpolate_threshold_crossing(
                freqs[i], hvsr[i], freqs[i + 1], hvsr[i + 1], threshold
            )
            break
        if i == n - 2 and hvsr[-1] >= threshold:
            f_hi = freqs[-1]

    return f_lo, f_hi


def _peak_metrics(freqs, hvsr, idx, baseline=1.0):
    """Compute HVSR peak metrics: width/Q, asymmetry, skewness, and energy."""
    f0 = freqs[idx]
    A0 = hvsr[idx]
    threshold = A0 / np.sqrt(2.0)
    f_lo, f_hi = _half_power_bounds(freqs, hvsr, idx, threshold)

    if np.isfinite(f_lo) and np.isfinite(f_hi) and f_hi > f_lo:
        delta_f = f_hi - f_lo
        Q = f0 / delta_f if delta_f > 0 else np.nan
        low_bw = f0 - f_lo
        high_bw = f_hi - f0
        asymmetry = low_bw / high_bw if (low_bw > 0 and high_bw > 0) else np.nan
        peak_mask = (freqs >= f_lo) & (freqs <= f_hi)
    else:
        delta_f = np.nan
        Q = np.nan
        low_bw = np.nan
        high_bw = np.nan
        asymmetry = np.nan
        peak_mask = np.zeros_like(freqs, dtype=bool)

    y = np.maximum(hvsr[peak_mask] - baseline, 0.0)
    f_peak = freqs[peak_mask]
    if y.size > 0 and np.sum(y) > 0:
        mu = np.sum(f_peak * y) / np.sum(y)
        var = np.sum(((f_peak - mu) ** 2) * y) / np.sum(y)
        sigma = np.sqrt(var)
        if sigma > 0:
            skewness = (np.sum(((f_peak - mu) ** 3) * y) / np.sum(y)) / sigma**3
        else:
            skewness = np.nan
        energy = np.trapz(y, f_peak)
    else:
        skewness = np.nan
        energy = np.nan

    return {
        'f_low': f_lo,
        'f_high': f_hi,
        'half_power_width': delta_f,
        'Q': Q,
        'bandwidth_low': low_bw,
        'bandwidth_high': high_bw,
        'asymmetry': asymmetry,
        'skewness': skewness,
        'peak_energy': energy,
        'half_power_threshold': threshold,
    }


def compute_peak_metrics(freqs, hvsr, idx, baseline=1.0):
    """Compute peak metrics for an HVSR peak index.

    Parameters
    ----------
    freqs : ndarray, shape (Nf,)
        Frequency axis in Hz.
    hvsr : ndarray, shape (Nf,)
        HVSR curve.
    idx : int
        Index of the peak in ``freqs``/``hvsr``.
    baseline : float
        Baseline to subtract before computing energy and skewness. Default 1.0.

    Returns
    -------
    metrics : dict
        Contains ``f_low``, ``f_high``, ``half_power_width``, ``Q``,
        ``bandwidth_low``, ``bandwidth_high``, ``asymmetry``, ``skewness``,
        ``peak_energy``, and ``half_power_threshold``.
    """
    return _peak_metrics(freqs, hvsr, idx, baseline)

def rank_peaks(hvsrdata, min_prominence=0.5, min_amplitude=2.0):
    """
    Detect and rank HVSR peaks following the SESAME guidelines.

    Each detected local maximum is tested against six criteria (three amplitude
    clarity and three amplitude/frequency stability checks).  The resulting
    score is an integer 0–6 (6 = highest quality).

    Parameters
    ----------
    hvsrdata : HVSRData object.
    min_prominence : float
        Minimum peak prominence passed to :func:`scipy.signal.find_peaks`.
        Default 0.5.
    min_amplitude : float
        Minimum HVSR amplitude to be considered a peak. Default 2.0 (SESAME
        recommends A0 > 2 as one ranking criterion).

    Returns
    -------
    peaks : list of dict
        Each entry contains:

        ``f0`` – Peak frequency (Hz)
        ``A0`` – Peak amplitude
        ``score`` – Integer 0–6
        ``details`` – Dict with boolean results for each of the 6 criteria
    """
    freqs = hvsrdata.freqs
    hvsr_all = hvsrdata.data
    hvsr_std_all = hvsrdata.stds

    if hvsr_std_all is None:
        hvsr_std_all = np.zeros_like(hvsr_all)

    peaks_all = dict()
    for i in range(len(hvsrdata.method)):
        hvsr = hvsr_all[i]
        hvsr_std = hvsr_std_all[i]

        method_label = f"M{i+1}"

        peak_indices, _ = find_peaks(
            hvsr,
            prominence=min_prominence,
            height=min_amplitude,
        )

        peaks = []
        for idx in peak_indices:
            f0 = freqs[idx]
            A0 = hvsr[idx]
            details = {}

            # ── Amplitude clarity ────────────────────────────────────────────────
            # C1: ∃ f⁻ ∈ [f0/4, f0] such that A0 / A(f⁻) > 2
            lo_mask = (freqs >= f0 / 4.0) & (freqs < f0)
            if lo_mask.any():
                details['c1'] = bool(np.any(A0 / hvsr[lo_mask] > 2.0))
            else:
                details['c1'] = False

            # C2: ∃ f⁺ ∈ [f0, 4*f0] such that A0 / A(f⁺) > 2
            hi_mask = (freqs > f0) & (freqs <= 4.0 * f0)
            if hi_mask.any():
                details['c2'] = bool(np.any(A0 / hvsr[hi_mask] > 2.0))
            else:
                details['c2'] = False

            # C3: A0 > 2
            details['c3'] = bool(A0 > 2.0)

            # ── Amplitude & frequency stability ──────────────────────────────────
            eps_f, log_theta = _sesame_thresholds(f0)

            # C4: peak appears within 5 % on HVSR ± σ curves
            #     i.e. f0±σ curves also peak near f0
            for sign in (+1, -1):
                curve = hvsr + sign * hvsr_std
                local_peaks, _ = find_peaks(curve, prominence=0)
                if len(local_peaks) > 0:
                    nearest_idx = local_peaks[np.argmin(np.abs(freqs[local_peaks] - f0))]
                    f_near = freqs[nearest_idx]
                    details['c4'] = bool(abs(f_near - f0) / f0 <= 0.05)
                else:
                    details['c4'] = False
                if not details.get('c4', True):
                    break   # fail if either σ-curve fails

            # C5: σf < ε(f0) × f0
            sigma_f = hvsr_std[idx]   # approximate σ at peak; proper σf needs
                                    # per-window peak tracking (simplified here)
            details['c5'] = bool(sigma_f < eps_f * f0)

            # C6: σ_log(A(f0)) < log θ(f0)
            with np.errstate(divide='ignore', invalid='ignore'):
                log_sigma = np.log10(A0 + hvsr_std[idx]) - np.log10(A0)
            details['c6'] = bool(log_sigma < log_theta)

            score = sum(details.values())
            metrics = _peak_metrics(freqs, hvsr, idx)
            peaks.append(dict(
                f0=f0,
                A0=A0,
                score=score,
                details=details,
                **metrics,
            ))

        # Sort by score descending, then amplitude descending
        peaks.sort(key=lambda p: (p['score'], p['A0']), reverse=True)

        peaks_all[method_label] = peaks
        #
    return peaks_all


# ──────────────────────────────────────────────────────────────────────────────
# VS30 estimation (McNamara et al. 2015)
# ──────────────────────────────────────────────────────────────────────────────

def estimate_vs30(hvsr_peak_freq):
    """
    Estimate VS30 from the dominant HVSR peak frequency using the empirical
    relationship derived by McNamara et al. (2015) from eastern US stations::

        VS30 = 51.90 × f_peak + 254.7   (m/s)

    Parameters
    ----------
    hvsr_peak_freq : float
        Dominant HVSR peak frequency (Hz).

    Returns
    -------
    vs30 : float
        Estimated VS30 in m/s.
    """
    return 51.90 * hvsr_peak_freq + 254.7


# ──────────────────────────────────────────────────────────────────────────────
# Main user-interface function
# ──────────────────────────────────────────────────────────────────────────────

def compute_hvsr(stream, win_len_s=100.0, step_s=None,
                 method=4,
                 freqmin=0.1, freqmax=None,
                 taper_frac=0.05, smooth_n=None,
                 remove_outliers=True, outlier_std_factor=3.0,
                 verbose=True, save = False, outdir='.', outfile = None, save_format = 'asdf',
                 sta_inv=None,force_return=False):
    """
    Compute the Horizontal-to-Vertical Spectral Ratio (HVSR) from an
    :class:`obspy.Stream` containing three-component ambient-noise waveforms.

    This is the main user-interface function.  It accepts a stream, slices it
    into windows, computes PSDs, applies the chosen averaging/merging method
    and returns a results dictionary ready for plotting or further analysis.

    Parameters
    ----------
    stream : obspy.Stream
        Must contain exactly three traces whose channel codes end in Z, N (or
        1), and E (or 2).  All traces are assumed to be synchronous, of equal
        length, and pre-processed (instrument response removed, filtered).
    win_len_s : float
        Analysis window length in seconds. Default 100 s.
    step_s : float, optional
        Step between successive windows in seconds.  ``None`` defaults to
        ``win_len_s`` (non-overlapping windows).
    method : int or list of int
        HVSR computation method(s) to apply.  Accepts a single integer 1–6 or
        a list thereof.  All requested methods are returned in the output dict.

        ===  ===================================================================
        M1   Diffuse Field Assumption (Sánchez-Sesma et al. 2011)
        M2   Average-of-ratios, horizontal = arithmetic mean
        M3   Average-of-ratios, horizontal = geometric mean
        M4   Average-of-ratios, horizontal = vector summation  *(default)*
        M5   Average-of-ratios, horizontal = quadratic mean
        M6   Average-of-ratios, horizontal = maximum
        ===  ===================================================================

    freqmin : float
        Low-frequency cut for the output HVSR (Hz). Default 0.1 Hz.
    freqmax : float, optional
        High-frequency cut for the output HVSR (Hz). ``None`` = Nyquist / 2.
    taper_frac : float
        Fraction of each window tapered with a cosine function. Default 0.05.
    smooth_n : int, optional
        Number of frequency samples to smooth each per-window PSD with a
        boxcar convolution using :func:`seisgo.utils.smooth`.  ``None`` = no
        smoothing.
    remove_outliers : bool
        If ``True``, windows whose maximum PSD (on any component) exceeds
        ``outlier_std_factor`` standard deviations above the median are
        discarded before the HVSR average is formed.  Useful for short data
        segments that may contain earthquake signals. Default ``True``.
    outlier_std_factor : float
        Multiplier used when ``remove_outliers=True``. Default 3.0.
    min_peak_amplitude : float
        Minimum HVSR amplitude to qualify as a peak (SESAME criterion C3).
        Default 2.0.
    min_peak_prominence : float
        Minimum peak prominence passed to :func:`scipy.signal.find_peaks`.
        Default 0.5.
    verbose : bool
        Print progress messages. Default ``True``.
    save : bool
        If ``True``, save the results to disk in the specified format. Default
        ``False``.
    outdir : str, optional
        Output directory for saving results.  Ignored if ``save=False``.  If
        ``save=True`` and ``outfile=None``, results are saved to the current
        directory or ``outdir`` if provided.
    outfile : str, optional
        Output file path.  If ``save=True`` and ``outfile=None``, a default
        name is generated based on the station ID.  Ignored if ``save=False``.
    save_format : str
        Format to save results. Currently only 'asdf' is supported. Default 'asdf'.
    station_inv : dict, optional
        Optional station inventory used to extract station information.
    force_return : bool
        If ``True``, return the results dict even if saving to disk.  Default ``False`` 
        (results are not returned when ``save=True`` to encourage users to
        load from disk and avoid keeping large results dicts in memory).
    
    Returns
    -------
    result : dict
        Keys:

        ``'freqs'`` – ndarray, frequency axis (Hz), already clipped to
        [freqmin, freqmax].

        ``'methods'`` – dict keyed by method integer (1–6).  Each value is a
        sub-dict with:

            ``'hvsr'``     – ndarray, mean HVSR curve
            ``'hvsr_std'`` – ndarray, standard deviation (nan for M1 unless
                             bootstrapped)
            ``'label'``    – human-readable method label
            ``'peaks'``    – list of ranked peak dicts (if ``rank=True``).
                             See :func:`rank_peaks` for field descriptions.

        ``'n_windows'`` – int, number of analysis windows retained.
        ``'stream_id'`` – str, network.station identifier.

    Examples
    --------
    >>> from obspy import read
    >>> from seisgo import hvsr
    >>> st = read("my_3comp_noise.mseed")
    >>> result = hvsr.compute_hvsr(st, win_len_s=200, method=[1, 4],
    ...                            freqmin=0.05, freqmax=20.0)
    >>> hvsr.plot_hvsr(result, methods=[1, 4])

    Notes
    -----
    * Instrument response should be removed *before* calling this function.
    * All three component traces must be trimmed to the same time window.
    * For very long datasets the DFA method (M1) is preferred because it
      processes the ratio of ensemble-averaged PSDs, making it more robust
      against non-stationary noise windows.
    * MUSTANG-derived PSDs already include internal smoothing; when providing
      pre-computed (smooth) PSDs pass ``smooth_n=None``.

    See Also
    --------
    :func:`rank_peaks`, :func:`estimate_vs30`, :func:`plot_hvsr`
    """
    if isinstance(method, int):
        methods = [method]
    else:
        methods = list(method)

    for m in methods:
        if m not in range(1, 7):
            raise ValueError(f"Method must be an integer 1–6, got {m}.")

    # save or not. build file name if save is True and outfile is None
    if save and outfile is None:
        station_id = f"{stream[0].stats.network}.{stream[0].stats.station}"
        if save_format == 'asdf':
            outfile = os.path.join(outdir, f"hvsr_{station_id}.h5") 
        else:
            raise ValueError(f"Unsupported save_format: {save_format}.  "
                             "Currently only 'asdf' is supported.")
        print(f"Results will be saved to {outfile}")
    # ── Extract components ───────────────────────────────────────────────────
    tr_z, tr_n, tr_e = _get_components(stream)
    fs = tr_z.stats.sampling_rate
    station_id = f"{tr_z.stats.network}.{tr_z.stats.station}"

    if freqmax is None:
        freqmax = fs / 2.0 * 0.9   # stay slightly below Nyquist

    if verbose:
        print(f"[compute_hvsr] Station: {station_id}")
        print(f"  Sampling rate   : {fs} Hz")
        print(f"  Window length   : {win_len_s} s")
        print(f"  Step            : {step_s if step_s else win_len_s} s")
        print(f"  Frequency range : {freqmin} – {freqmax} Hz")
        print(f"  Methods         : {methods}")

    # ── Compute per-window PSDs for all three components ─────────────────────
    freqs_z, psds_z = compute_window_psds(tr_z, win_len_s, step_s,
                                           taper_frac=taper_frac,
                                           smooth_n=smooth_n)
    freqs_n, psds_n = compute_window_psds(tr_n, win_len_s, step_s,
                                           taper_frac=taper_frac,
                                           smooth_n=smooth_n)
    freqs_e, psds_e = compute_window_psds(tr_e, win_len_s, step_s,
                                           taper_frac=taper_frac,
                                           smooth_n=smooth_n)

    # Verify all components produced the same frequency axis
    if not (np.allclose(freqs_z, freqs_n) and np.allclose(freqs_z, freqs_e)):
        raise RuntimeError("Frequency axes differ between components.  "
                           "Ensure all traces have the same sampling rate and length.")

    freqs = freqs_z
    n_windows_all = psds_z.shape[0]

    # ── Optional outlier removal ──────────────────────────────────────────────
    if remove_outliers:
        keep = _remove_outlier_windows(psds_z, psds_n, psds_e,
                                       outlier_std_factor)
        n_removed = n_windows_all - keep.sum()
        psds_z = psds_z[keep]
        psds_n = psds_n[keep]
        psds_e = psds_e[keep]
        if verbose:
            print(f"  Outlier removal : {n_removed}/{n_windows_all} windows removed")

    n_windows = psds_z.shape[0]
    if n_windows == 0:
        raise RuntimeError("All windows were removed as outliers.  "
                           "Reduce outlier_std_factor or disable remove_outliers.")

    if verbose:
        print(f"  Windows retained: {n_windows}")

    # ── Frequency mask ────────────────────────────────────────────────────────
    fmask = (freqs >= freqmin) & (freqs <= freqmax)
    freqs_out = freqs[fmask]

    # Mean PSDs used by M1 (DFA)
    _, mean_psd_z = mean_psd(freqs, psds_z)
    _, mean_psd_n = mean_psd(freqs, psds_n)
    _, mean_psd_e = mean_psd(freqs, psds_e)

    #extract station information from inventory if provided, otherwise use sac headers if available, else set to nan.
    if sta_inv is not None:
        sta,net,lon,lat,ele,loc = sta_info_from_inv(sta_inv)
    elif hasattr(tr_z.stats, 'sac'):
        sta = tr_z.stats.sac.stnm
        net = tr_z.stats.sac.net
        lon = tr_z.stats.sac.stlo
        lat = tr_z.stats.sac.stla
        ele = tr_z.stats.sac.stel
        loc = tr_z.stats.sac.loc
    else:
        sta,net,lon,lat,ele,loc = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    hvsr_out_all = []
    hvsr_std_out_all = []
    for m in methods:
        label = METHODS[m]

        if m == 1:
            hvsr_full = _hvsr_m1(mean_psd_z, mean_psd_n, mean_psd_e)
            hvsr_out = hvsr_full[fmask]
            # For M1 produce a bootstrap std from per-window DFA ratios
            per_win = np.sqrt(
                (psds_n + psds_e) / np.clip(psds_z, 1e-40, None)
            )
            hvsr_std_out = np.nanstd(per_win, axis=0)[fmask]
        else:
            hvsr_full, hvsr_std_full = _hvsr_avg_of_ratios(
                psds_z, psds_n, psds_e, m)
            hvsr_out = hvsr_full[fmask]
            hvsr_std_out = hvsr_std_full[fmask]

        # Replace inf/nan with 0 to avoid downstream issues
        hvsr_out = np.nan_to_num(hvsr_out, nan=0.0, posinf=0.0, neginf=0.0)
        hvsr_std_out = np.nan_to_num(hvsr_std_out, nan=0.0, posinf=0.0, neginf=0.0)

        # Store results for this method
        hvsr_out_all.append(hvsr_out)
        hvsr_std_out_all.append(hvsr_std_out)
    #
    hvsr_out_all = np.array(hvsr_out_all)
    hvsr_std_out_all = np.array(hvsr_std_out_all)

    #build HVSRData object (in types.py) from result dict, and then save it to file using HVSRData.save() method.
    hvsr_data = HVSRData(net=net, sta=sta, loc=loc, lon=lon, lat=lat, ele=ele, freqmin=freqmin, freqmax=freqmax,
                            freqs=freqs_out, method=methods, stds=hvsr_std_out_all, data=hvsr_out_all, n_windows=n_windows, 
                            win_len_s=win_len_s, step_s=step_s, label = label)
    # save to file or not.
    if save:
        hvsr_data.save(outfile)
        if force_return:
            return hvsr_data
    else:
        return hvsr_data

# ──────────────────────────────────────────────────────────────────────────────
# Outlier removal helper
# ──────────────────────────────────────────────────────────────────────────────

def _remove_outlier_windows(psds_z, psds_n, psds_e, std_factor=3.0):
    """
    Return a boolean mask flagging windows that are *not* outliers.

    A window is considered an outlier if the maximum PSD value across all
    three components at any frequency lies more than ``std_factor`` standard
    deviations above the median of all window maxima.

    Parameters
    ----------
    psds_z, psds_n, psds_e : ndarray, shape (Nwindows, Nf)
    std_factor : float

    Returns
    -------
    keep : ndarray of bool, shape (Nwindows,)
    """
    # Maximum PSD value in each window (take max across all components)
    win_max = np.maximum(np.max(psds_z, axis=1),
                np.maximum(np.max(psds_n, axis=1),
                           np.max(psds_e, axis=1)))
    med = np.median(win_max)
    std = np.std(win_max)
    threshold = med + std_factor * std
    return win_max <= threshold


# ──────────────────────────────────────────────────────────────────────────────
# Seasonal / time-series analysis
# ──────────────────────────────────────────────────────────────────────────────

def compute_hvsr_batch(stream_list, labels=None, force_return=False, save=False, **kwargs):
    """
    Compute HVSR for a sequence of streams (e.g. one per month) and return
    results as a list.  This is the wrapper for seasonal-variation analyses
    similar to Fig. 3 in Bahavar et al. (2020). This function process all data
    in the stream_list, which can be a list of obspy.Stream or a list of h5 files.
    If it is a list of h5 files, we will read each file one by one to avoid memory issues.

    Parameters
    ----------
    stream_list : list of obspy.Stream
        Ordered list of streams (e.g. monthly noise windows).
    labels : list of str, optional
        Human-readable label for each stream (e.g. month names).
        If ``None``, integer indices are used.
    force_return : bool
        If ``True``, return the results dict even if saving to disk.  Default ``False``.
    save : bool
        If ``True``, save the results to disk in the specified format. Default ``False``.
    **kwargs
        All keyword arguments are passed directly to :func:`compute_hvsr`.

    Returns
    -------
    results : dict of dict
         Keys are network.station identifiers.  Values are dicts keyed by label,
         each containing the result dict returned by :func:`compute_hvsr` for that stream.
         For example, results['NET.STA']['Jan 2020'] is the result dict for the stream labeled 'Jan 2020' at station NET.STA.
    labels : list of str
        Corresponding labels.
    netsta_list : list of str
        List of unique network.station identifiers found in the input streams.
    """
    if labels is None:
        labels = [str(i) for i in range(len(stream_list))]

    # check the type of stream_list. if it is a list of h5 files, then loop through each file. 
    # if it is a list of obspy stream, then loop through each stream. Do not read all data into memory at once if it is a list of h5 files, to void memory issues.
    
    results_all = dict() # organize results by stations.
    netsta_list = set() # keep track of all netsta in the stream list.
    for i, st in enumerate(stream_list):
        if isinstance(st, str):
            if st.endswith('.h5'): 
                traces = extract_waveform(st,mpi=False)
                st = Stream(traces)
            else:
                raise ValueError(f"Unsupported file type: {st}. Only .h5 files are supported currently.")
        elif isinstance(st, Stream):
            pass
        else:
            raise ValueError(f"Unsupported type: {type(st)}. Only strings (file paths) and obspy.Stream objects are supported.")
        # extract network and station from the stream for labeling
        # stream may contain multiple network and stations. we will group stream by station (3 channels) and loop through each station.

        netsta = set()
        for tr in st:
            netsta.add((tr.stats.network, tr.stats.station))
        
        # loop through each station in the stream and compute hvsr for each station separately.
        for net, sta in netsta:
            st_netsta = st.select(network=net, station=sta)
            if len(st_netsta) < 3:
                warnings.warn(f"Stream for {net}.{sta} has less than 3 traces. Skipping.")
                continue
            netsta= f"{net}.{sta}"
            netsta_list.add(netsta)
            print(f"\n[compute_hvsr_timeseries] Processing {netsta} ({i+1}/{len(stream_list)}): {labels[i]}")

            try:
                res = compute_hvsr(st_netsta, force_return=force_return, **kwargs)

                if netsta not in results_all:
                    results_all[netsta] = dict()
                results_all[netsta][labels[i]] = res
            except Exception as exc:
                warnings.warn(f"Segment {labels[i]} for {netsta} failed: {exc}")
                results_all[netsta][labels[i]] = None
    if save:
        if force_return:
            return results_all, labels, list(netsta_list)
        else:
            return None, labels, list(netsta_list)
    else:
        return results_all, labels, list(netsta_list)
# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────
def plot_hvsr(hvsrdata, method=None, show_std=True, figsize=(9, 5), ymax=None, xtype='frequency', 
             title=None, save=False, figname=None, fmt='png',peaks=None):
    """
    Plot the HVSR data.

    ======Parameters=====
    hvsrdata: HVSRData object containing the data to plot.
    method: list of methods to plot. Default is all methods in self.method.
    show_std: whether to show standard deviation as shaded area. Default is True.
    figsize: figure size tuple. Default is (9, 5).
    ymax: maximum y value for the plot. Default is None (auto).
    xtype: x-axis type, either 'frequency' or 'period'. Default is 'frequency
    title: figure title. Default is None (auto).
    save: whether to save the figure. Default is False.
    figname: figure name when save is True. Default is None (auto).
    fmt: figure format when save is True. Default is 'png'.
    peaks: list of peak frequencies to highlight. Default is None (skip annotating peaks).
    """
    if hvsrdata.freqs is None or hvsrdata.data is None:
        raise ValueError("No HVSR data to plot.")
    x = hvsrdata.freqs if xtype == 'frequency' else 1.0 / np.where(hvsrdata.freqs > 0, hvsrdata.freqs, np.nan)
    if method is None:
        method = hvsrdata.method
    elif isinstance(method, int):
        method = [method]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    linestyles = ['-', '--', '-.', ':', (0,(3,1,1,1)), (0,(5,1))]

    #find the correct method indices since the data matrix does not necessarily have the same order as self.method list.
    method_indices = []
    for m in method:
        if m in hvsrdata.method:
            method_indices.append(hvsrdata.method.index(m))
        else:
            raise ValueError(f"Method {m} not found in hvsrdata.method list.")
    method_indices = np.array(method_indices)
    fig, ax = plt.subplots(figsize=figsize)
    for i, m in enumerate(method):
        m_index = method_indices[i]
        hvsr = hvsrdata.data[m_index, :]
        hvsr_std = hvsrdata.stds[m_index,:] if hvsrdata.stds is not None else None

        color = colors[i % len(colors)]
        ls = linestyles[i % len(linestyles)]
        m_label = f"M{m}"
        ax.plot(x, hvsr, color=color, linestyle=ls, linewidth=1.5, label=m_label)
        if show_std and hvsr_std is not None:
            ax.fill_between(x, np.maximum(hvsr - hvsr_std, 0), hvsr + hvsr_std, color=color, alpha=0.15)
        if peaks is not None:
            best = peaks[m_label][0]
            pf = best['f0'] if xtype == 'frequency' else 1.0 / best['f0']
            ax.axvline(pf, color=color, linestyle=':', linewidth=0.8, alpha=0.7)
            ax.annotate(f"f₀={best['f0']:.3f} Hz\nA₀={best['A0']:.2f}\nscore={best['score']}/6",
                        xy=(pf, best['A0']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=7, color=color)
    ax.axhline(1.0, color='k', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Frequency (Hz)' if xtype == 'frequency' else 'Period (s)')
    ax.set_ylabel('HVSR')
    if ymax is not None:
        ax.set_ylim(0, ymax)
    else:
        ax.set_ylim(bottom=0)
    ttl = title if title else f"HVSR – {hvsrdata.id}"
    ttl += f"\n(n_windows={hvsrdata.n_windows})"
    ax.set_title(ttl, fontsize=10)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    if save:
        if figname is None:
            sid = hvsrdata.id.replace('.', '_')
            figname = f"{sid}_HVSR.{fmt}"
        fig.savefig(figname, dpi=150, bbox_inches='tight')
        print(f"Figure saved: {figname}")
    return fig, ax


def plot_hvsr_timeseries(results, labels=None, method=4, figsize=None,
                         ymax=None, title=None, save=False, figname=None,
                         fmt='png'):
    """
    Plot HVSR curves from a time-series of results (e.g. monthly) as a panel
    figure similar to Fig. 3 of Bahavar et al. (2020).

    Parameters
    ----------
    results : list of dict
        Output of :func:`compute_hvsr_timeseries` (``None`` entries are
        skipped).
    labels : list of str
        Corresponding labels (months, dates, etc.).
    method : int
        Which method to plot in the panel. Default 4.
    figsize : tuple, optional
        Figure size. Auto-sized when ``None``.
    ymax : float, optional
        Common y-axis maximum. Auto-scaled when ``None``.
    title : str, optional
        Overall figure title.
    save, figname, fmt : see :func:`plot_hvsr`.

    Returns
    -------
    fig : matplotlib Figure
    """
    valid = [(res, lbl) for res, lbl in zip(results, labels or range(len(results)))
             if res is not None and method in res['methods']]

    n = len(valid)
    if n == 0:
        raise ValueError("No valid results to plot.")

    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))

    if figsize is None:
        figsize = (ncols * 3.5, nrows * 3.0)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             sharex=True, sharey=True)
    axes = np.array(axes).flatten()

    for ax_i, (res, lbl) in enumerate(valid):
        ax = axes[ax_i]
        sub = res['methods'][method]
        freqs = res['freqs']
        hvsr = sub['hvsr']
        hvsr_std = sub['hvsr_std']

        ax.plot(freqs, hvsr, 'k-', linewidth=1.2)
        ax.fill_between(freqs,
                        np.maximum(hvsr - hvsr_std, 0),
                        hvsr + hvsr_std,
                        color='gray', alpha=0.3)
        ax.axhline(1.0, color='k', linewidth=0.6, linestyle='--', alpha=0.5)
        ax.set_xscale('log')
        ax.set_xlim(freqs[0], freqs[-1])
        if ymax:
            ax.set_ylim(0, ymax)
        else:
            ax.set_ylim(bottom=0)
        ax.set_title(str(lbl), fontsize=8)
        ax.grid(True, which='both', alpha=0.25)

    # Hide empty subplots
    for ax in axes[len(valid):]:
        ax.set_visible(False)

    fig.supxlabel('Frequency (Hz)', fontsize=10)
    fig.supylabel('HVSR', fontsize=10)
    if title:
        fig.suptitle(title, fontsize=11)

    fig.tight_layout()

    if save:
        if figname is None:
            figname = f"HVSR_timeseries_M{method}.{fmt}"
        fig.savefig(figname, dpi=150, bbox_inches='tight')
        print(f"Figure saved: {figname}")

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: print a ranking report (mirrors Table 3 in the paper)
# ──────────────────────────────────────────────────────────────────────────────

def print_peak_report(peaks, method=None):
    """
    Print a formatted ranking report for HVSR peaks.

    Parameters
    ----------
    peaks : list of dict
        List of peak dictionaries.
    method : list of int, optional
        Which methods to include. Defaults to all methods in ``result``.
    """
    peaks_keys = list(peaks.keys())
    peaks_method_int = [int(k[-1]) for k in peaks_keys if k.startswith('M')]
    if method is None:
        method = peaks_method_int
    elif isinstance(method, int):
        method = [method]

    print(f"\n{'='*72}")
    print(f"HVSR Peak Ranking Report")
    print(f"{'='*72}")

    for m in method:
        m_label = f"M{m}"

        m_peaks = peaks[m_label]
        print(f"\nMethod {m_label}: {METHODS[m]}")
        if not m_peaks:
            print("  No peaks detected.")
            continue

        header = (f"{'f0 (Hz)':>10}  {'A0':>6}  "
                  f"{'C1':>3} {'C2':>3} {'C3':>3} "
                  f"{'C4':>3} {'C5':>3} {'C6':>3}  {'Score':>6}")
        print("  " + header)
        print("  " + "-" * (len(header)))

        for p in m_peaks:
            d = p['details']
            row = (f"{p['f0']:>10.4f}  {p['A0']:>6.2f}  "
                   f"{'✓' if d.get('c1') else '✗':>3} "
                   f"{'✓' if d.get('c2') else '✗':>3} "
                   f"{'✓' if d.get('c3') else '✗':>3}  "
                   f"{'✓' if d.get('c4') else '✗':>3} "
                   f"{'✓' if d.get('c5') else '✗':>3} "
                   f"{'✓' if d.get('c6') else '✗':>3}  "
                   f"{p['score']:>4}/6")
            print("  " + row)

        best = m_peaks[0]
        print(f"\n  Best peak metrics: Q={best['Q']:.2f}, "
              f"half_width={best['half_power_width']:.3f} Hz, "
              f"energy={best['peak_energy']:.3f}, "
              f"asymmetry={best['asymmetry']:.3f}, "
              f"skewness={best['skewness']:.3f}")

        # VS30 estimate from best peak
        best_f0 = best['f0']
        vs30 = estimate_vs30(best_f0)
        print(f"\n  VS30 estimate (McNamara et al. 2015): {vs30:.1f} m/s "
              f"(f0 = {best_f0:.4f} Hz)")

    print(f"\n{'='*72}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Data extraction from saved files
# ──────────────────────────────────────────────────────────────────────────────
def extract_hvsrdata(filename, verbose=True, format='asdf'):
    """
    Extract HVSR data from a saved ASDF file.

    Parameters
    ----------
    filename : str
        Path to the ASDF file saved by HVSRData.to_asdf().
    verbose : bool
        Print progress messages. Default True.

    Returns
    -------
    result : dict
        Dictionary with keys 'freqs', 'methods', 'n_windows', 'stream_id',
        matching the output of :func:`compute_hvsr`.
    """
    if format != 'asdf':
        raise ValueError(f"Unsupported format: {format}. Only 'asdf' is supported currently.")
    
    ds = pyasdf.ASDFDataSet(filename, mode='r')
    freqs = ds.auxiliary_data['freqs'].data
    methods = {}
    aux_paths = ds.auxiliary_data.list()
    params = None
    for path in aux_paths:
        if path.startswith('hvsr/M'):
            m_str = path.split('/M')[1]
            m = int(m_str)
            hvsr_data = ds.auxiliary_data[path].data
            params = ds.auxiliary_data[path].parameters
            std_path = f"hvsr_std/M{m}"
            if std_path in aux_paths:
                hvsr_std = ds.auxiliary_data[std_path].data
            else:
                hvsr_std = None
            methods[m] = {
                'hvsr': hvsr_data,
                'hvsr_std': hvsr_std,
                'peaks': None,  # Peaks not saved in ASDF
                'label': METHODS.get(m, '')
            }
    n_windows = params.get('n_windows') if params else None
    stream_id = params.get('id') if params else ''
    result = {
        'freqs': freqs,
        'methods': methods,
        'n_windows': n_windows,
        'stream_id': stream_id
    }
    if verbose:
        print(f"Extracted HVSR data from {filename}")
    return result
