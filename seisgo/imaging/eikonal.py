"""
seisgo.imaging.eikonal -- Eikonal-equation ambient-noise surface-wave tomography.

Turns an ensemble of pairwise dispersion measurements (DispData objects, one per
station pair, produced by aftan()/aftan_pmf() in seisgo.dispersion) into gridded 
phase-/group-velocity maps (PhaseVelocityMap, in seisgo.types) as a function of period, 
following the ambient-noise eikonal-tomography method of Lin, Ritzwoller & Snieder (2009, GJI)
and Lin & Ritzwoller (2011, GJI).

===Method summary===
For a wave radiating outward from a virtual source at a fixed period T, the
eikonal equation says the local phase slowness equals the magnitude of the
gradient of the travel-time field: |grad(tau(r))| = 1/c(r), where tau(r) is the
travel time from the source to point r and c(r) is the local phase (or group)
velocity -- the standard ray-theoretical/plane-wavefront approximation. Given one
virtual source S and a set of receivers R around it, each station pair gives one
observed travel time tau(S->R) = dist(S,R) / c_obs(S,R,T) (c_obs from the
station-pair DispData at period T). This module:

  1. groups the input DispData ensemble by virtual source (DispData.src_net/
     src_sta/src_lon/src_lat -- see that class's docstring; populated
     automatically by aftan()/aftan_pmf() from a `corrdata` argument, or via
     explicit src_*/rcv_* keyword arguments),
  2. for each (source, period), interpolates each contributing curve's velocity
     (and SNR) onto the requested period, builds the observed travel times to
     every usable receiver, and fits a smooth travel-time surface + its local
     gradient onto a regular grid via a Gaussian-weighted local plane-wave fit
     (see _local_plane_fit_grid() below -- this is the practical, easy-to-verify
     stand-in used here for the natural-neighbor/minimum-curvature surface fit of
     Lin & Ritzwoller's original GMT-based implementation; it interpolates AND
     differentiates in the same weighted-least-squares step, which sidesteps the
     classic interpolate-then-finite-difference error amplification, at the cost
     of being a local planar rather than a globally-smooth surface),
  3. quality-controls each source's own per-period map (minimum station count,
     azimuthal-coverage gap, near-field exclusion, residual-based outlier
     rejection),
  4. stacks the per-source maps at each grid node/period into a final velocity
     estimate (the mean across contributing sources) and its uncertainty (the
     standard deviation across sources -- one of the practical strengths of the
     eikonal approach: spatially-resolved uncertainty comes essentially for free,
     rather than needing a separate resolution/checkerboard test).

===Known approximations/pitfalls (see also this module's top-level docstring
discussion)===
  - Ray-theoretical: ignores finite-frequency/diffraction sensitivity.
  - Near-field breakdown close to each source, and at the map's edges (handled by
    explicit exclusion/coverage checks, but the exact cutoffs are tunable and
    matter).
  - Off-great-circle bias from strong lateral heterogeneity shows up as
    azimuthal anisotropy in the *apparent* velocity unless averaged over enough
    independent source azimuths -- the per-node source count (n_sources on the
    returned PhaseVelocityMap) is the diagnostic to watch.
  - The `smooth_km` interpolation length scale is the single most consequential
    tuning knob: too large erases real structure, too small amplifies AFTAN
    picking noise into gradient artifacts.
  - Cycle-skipped phase-velocity picks (see aftan()'s ref_period/ref_velocity)
    will show up as large, obviously-wrong local travel-time residuals -- the
    `resid_zmax` outlier rejection in _local_plane_fit_grid() catches isolated
    cases but is not a substitute for good picks going in.

===Typical usage===
>>> from seisgo.imaging.eikonal import load_dispdata_ensemble, make_grid, eikonal_tomography
>>> curves = load_dispdata_ensemble("dispersion_proj")   # from run_dispersion_pipeline.py
>>> lon_grid, lat_grid = make_grid(-90.0, -89.0, 19.0, 19.8, spacing_km=5.0)
>>> pvmap = eikonal_tomography(curves, lon_grid, lat_grid, periods=[4, 6, 8, 10],
...                             vtype='phase', smooth_km=15.0, min_stations=4)
>>> pvmap.plot(period=6.0)
>>> pvmap.save("phase_velocity_map.h5")
"""
import warnings, json, h5py

import numpy as np
from scipy.spatial import cKDTree

from seisgo.types import DispData, PhaseVelocityMap
from seisgo.dispersion import read_dispdata

_EARTH_RADIUS_KM = 6371.0


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def _km_per_degree(lat):
    """
    km per degree of longitude and latitude at geographic latitude `lat` (deg),
    spherical-Earth approximation -- adequate at the length scales (tens to a few
    hundred km) relevant to a local/regional eikonal-tomography grid; not meant
    for continental- or global-scale grids.
    """
    lat_rad = np.deg2rad(lat)
    km_per_deg_lat = (np.pi / 180.0) * _EARTH_RADIUS_KM
    km_per_deg_lon = km_per_deg_lat * np.cos(lat_rad)
    return km_per_deg_lon, km_per_deg_lat


def _haversine_km(lon1, lat1, lon2, lat2):
    """Great-circle distance (km) between (lon1,lat1) and (lon2,lat2) (deg); any of
    the four may be arrays (broadcast together)."""
    lon1r, lat1r, lon2r, lat2r = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2r - lon1r
    dlat = lat2r - lat1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def make_grid(lon_min, lon_max, lat_min, lat_max, spacing_km):
    """
    Build a regular lon/lat grid with an approximately uniform spacing_km cell
    size (the longitude step is widened by 1/cos(lat) at the grid's central
    latitude, since a fixed degree spacing would otherwise give East-West-
    compressed cells away from the equator).

    ===RETURNS===
    lon_grid,lat_grid: 1-D coordinate vectors (deg).
    """
    lat_mid = 0.5 * (lat_min + lat_max)
    km_per_deg_lon, km_per_deg_lat = _km_per_degree(lat_mid)
    dlat = spacing_km / km_per_deg_lat
    dlon = spacing_km / km_per_deg_lon
    lon_grid = np.arange(lon_min, lon_max + 0.5 * dlon, dlon)
    lat_grid = np.arange(lat_min, lat_max + 0.5 * dlat, dlat)
    return lon_grid, lat_grid


# ---------------------------------------------------------------------------
# per-source, per-period travel-time surface fit + gradient
# ---------------------------------------------------------------------------
def _local_plane_fit_grid(lon_grid, lat_grid, lon_obs, lat_obs, tt_obs,
                           smooth_km=50.0, cutoff_sigma=3.0, min_stations=4,
                           max_azgap_deg=180.0, resid_zmax=3.0):
    """
    At every node of a regular (lon_grid,lat_grid) grid, fit a Gaussian-weighted
    local plane tau(x,y) = a + b*x + c*y to the observed travel times `tt_obs` at
    receiver locations (lon_obs,lat_obs), in local km coordinates (x,y) centered on
    that node (a single, grid-wide lon/lat->km scaling is used -- see
    _km_per_degree() -- adequate for a regional-scale grid). The fitted intercept
    `a` is the smoothed/interpolated travel time at the node; the fitted slope
    (b,c) IS the local travel-time gradient (the slowness vector), used directly
    via the eikonal equation as 1/|grad(tau)| = local phase (or group) velocity --
    interpolation and differentiation happen in the same weighted least-squares
    step, rather than interpolating onto the grid and then finite-differencing
    (which would amplify small measurement-noise wiggles through the extra
    differencing operation).

    Only receivers within `cutoff_sigma`*`smooth_km` of a node contribute to that
    node's fit (found efficiently via a KD-tree over receiver locations, rather
    than a dense distance computation against every grid node); weight falls off
    as a Gaussian with length scale `smooth_km` within that cutoff.

    ===PARAMETERS===
    lon_grid,lat_grid: 1-D grid coordinate vectors (deg), as from make_grid().
    lon_obs,lat_obs,tt_obs: receiver longitude/latitude (deg) and observed travel
            time (s) from the (single, fixed) virtual source, one entry per
            receiver used.
    smooth_km: Gaussian smoothing length scale (km). The single most consequential
            tuning parameter -- too large erases real velocity structure, too
            small amplifies picking noise into gradient artifacts. default 50.
    cutoff_sigma: receivers beyond cutoff_sigma*smooth_km of a node are excluded
            from that node's fit entirely (rather than just very lightly
            weighted), for speed. default 3.0.
    min_stations: a node is left NaN unless at least this many receivers fall
            within the cutoff radius (the local plane fit has 3 free parameters,
            so this should be at least 4-5 for a stable, slightly overdetermined
            fit). default 4.
    max_azgap_deg: a node is flagged as poor-coverage (velocity set to NaN, though
            the fitted travel time/gradient are still returned in the other
            output fields for inspection) if the largest gap between the
            receivers' azimuths (as seen from that node) exceeds this -- a wide
            gap means the "plane wave" direction is effectively unconstrained
            along part of the local neighborhood. default 180 (only rejects
            receivers all bunched into a half-plane or narrower).
    resid_zmax: after the first weighted-least-squares fit, receivers whose
            weighted residual exceeds resid_zmax standard deviations are dropped
            and the fit is redone once -- a simple, one-pass robustifying step
            against isolated bad travel-time picks (e.g. a cycle-skipped phase-
            velocity measurement) without needing a full iteratively-reweighted
            scheme. Set to None to disable. default 3.0.

    ===RETURNS===
    dict with, each shape (len(lat_grid), len(lon_grid)):
        tt: fitted (smoothed) travel time (s) at each node.
        velocity: 1/|grad(tt)| (km/s); NaN where a node has no fit, a singular fit,
                or failed the azimuthal-coverage check.
        azimuth: propagation azimuth (deg, 0=north/90=east compass bearing) of the
                local travel-time gradient, i.e. the apparent wave-propagation
                direction at that node.
        neff: sum of Gaussian weights actually used in the (possibly
                outlier-trimmed) fit -- an effective, distance-weighted station
                count, for QC/coverage diagnostics.
        azgap: the largest azimuthal gap (deg) between contributing receivers, as
                seen from that node.
    """
    lat_grid = np.asarray(lat_grid, dtype=np.float64)
    lon_grid = np.asarray(lon_grid, dtype=np.float64)
    nlat, nlon = len(lat_grid), len(lon_grid)
    lat0 = 0.5 * (lat_grid[0] + lat_grid[-1])
    km_lon, km_lat = _km_per_degree(lat0)

    lon_obs = np.asarray(lon_obs, dtype=np.float64)
    lat_obs = np.asarray(lat_obs, dtype=np.float64)
    tt_obs = np.asarray(tt_obs, dtype=np.float64)
    x_obs = lon_obs * km_lon
    y_obs = lat_obs * km_lat
    tree = cKDTree(np.column_stack([x_obs, y_obs]))

    LonG, LatG = np.meshgrid(lon_grid, lat_grid)  # shape (nlat, nlon)
    Xg = LonG * km_lon
    Yg = LatG * km_lat

    tt_grid = np.full((nlat, nlon), np.nan)
    vel = np.full((nlat, nlon), np.nan)
    azimuth = np.full((nlat, nlon), np.nan)
    neff = np.zeros((nlat, nlon))
    azgap = np.full((nlat, nlon), np.nan)

    radius = cutoff_sigma * smooth_km

    def _wls_fit(dx, dy, tt, w):
        A = np.column_stack([np.ones_like(dx), dx, dy])
        sw = np.sqrt(w)
        coef, _, rank, _ = np.linalg.lstsq(A * sw[:, None], tt * sw, rcond=None)
        if rank < 3:
            return None
        resid = tt - A @ coef
        return coef, resid

    for iy in range(nlat):
        for ix in range(nlon):
            node = (Xg[iy, ix], Yg[iy, ix])
            idx = tree.query_ball_point(node, r=radius)
            if len(idx) < min_stations:
                continue
            idx = np.asarray(idx)
            dx = x_obs[idx] - node[0]
            dy = y_obs[idx] - node[1]
            ttk = tt_obs[idx]
            r = np.hypot(dx, dy)
            w = np.exp(-0.5 * (r / smooth_km) ** 2)

            fit = _wls_fit(dx, dy, ttk, w)
            if fit is None:
                continue
            coef, resid = fit
            if resid_zmax is not None and len(ttk) > min_stations:
                wstd = np.sqrt(np.sum(w * resid ** 2) / np.sum(w))
                good = np.abs(resid) <= resid_zmax * max(wstd, 1e-9)
                if min_stations <= good.sum() < len(ttk):
                    fit2 = _wls_fit(dx[good], dy[good], ttk[good], w[good])
                    if fit2 is not None:
                        coef, resid = fit2
                        dx, dy, w = dx[good], dy[good], w[good]

            a, b, c = coef
            grad = np.hypot(b, c)
            tt_grid[iy, ix] = a
            neff[iy, ix] = np.sum(w)
            az = np.degrees(np.arctan2(dx, dy)) % 360.0  # bearing from node to each receiver
            az_sorted = np.sort(az)
            gaps = np.diff(np.concatenate([az_sorted, az_sorted[:1] + 360.0]))
            gap = float(gaps.max()) if len(gaps) else 360.0
            azgap[iy, ix] = gap
            if grad <= 0:
                continue
            azimuth[iy, ix] = np.degrees(np.arctan2(b, c)) % 360.0  # apparent propagation dir.
            if gap <= max_azgap_deg:
                vel[iy, ix] = 1.0 / grad

    return dict(tt=tt_grid, velocity=vel, azimuth=azimuth, neff=neff, azgap=azgap)


# ---------------------------------------------------------------------------
# ensemble loading + curve interpolation
# ---------------------------------------------------------------------------
def load_dispdata_ensemble(sources):
    """
    Load an ensemble of DispData objects for eikonal_tomography(), from a
    directory (recursively globbed for "*.h5", matching the layout
    run_dispersion_pipeline.py writes -- one file per station pair, at
    <dispdir>/<station>/<pair_id>.h5) or an explicit list of filenames and/or
    already-loaded DispData objects (mirroring assemble_dispersion()'s own
    `sources` convention in dispersion_dev.py). Entries that fail to load are
    skipped with a printed warning rather than raising.
    """
    import os
    import glob as _glob

    if isinstance(sources, str):
        entries = sorted(_glob.glob(os.path.join(sources, '**', '*.h5'), recursive=True))
    else:
        entries = list(sources)

    out = []
    for entry in entries:
        try:
            d = entry if isinstance(entry, DispData) else read_dispdata(entry)
        except Exception as e:
            print("load_dispdata_ensemble(): skipping entry %r (%s)" % (entry, e))
            continue
        out.append(d)
    return out


def _curve_velocity_at_period(dispdata, period, vtype='phase', snr_min=None):
    """
    Interpolate one DispData's velocity (and, if snr_min is given, its SNR) onto a
    single target period, in log-period space. Returns NaN if the target period
    falls outside this curve's own valid-pick period range, if fewer than 2 valid
    picks exist at all, or (when snr_min is given) if the SNR interpolated to that
    period doesn't clear it.
    """
    p = np.asarray(dispdata.period, dtype=np.float64)
    v = np.asarray(dispdata.phase_velocity if vtype == 'phase' else dispdata.group_velocity,
                    dtype=np.float64)
    good = np.isfinite(p) & np.isfinite(v)
    if good.sum() < 2:
        return np.nan
    p, v = p[good], v[good]
    order = np.argsort(p)
    p, v = p[order], v[order]
    if period < p[0] or period > p[-1]:
        return np.nan
    vi = float(np.interp(np.log(period), np.log(p), v))
    if snr_min is not None and dispdata.snr is not None:
        s = np.asarray(dispdata.snr, dtype=np.float64)[good][order]
        si = np.interp(np.log(period), np.log(p), s)
        if si < snr_min:
            return np.nan
    return vi


def _group_by_source(dispdata_list, verbose=False):
    """
    Group a DispData ensemble by virtual source (src_net,src_sta,src_lon,src_lat).
    Entries missing any station-pair metadata (src_*/rcv_* -- see DispData's
    docstring; not populated unless aftan()/aftan_pmf() were given a `corrdata` or
    explicit src_*/rcv_* arguments) are skipped, with a single summary warning.
    """
    groups = {}
    n_skip = 0
    for d in dispdata_list:
        if (d.src_sta is None or d.rcv_sta is None or d.src_lon is None or
                d.rcv_lon is None or d.src_lat is None or d.rcv_lat is None):
            n_skip += 1
            continue
        key = (d.src_net, d.src_sta, float(d.src_lon), float(d.src_lat))
        groups.setdefault(key, []).append(d)
    if verbose and n_skip:
        print("eikonal_tomography(): skipped %d DispData entr%s with missing station-pair "
              "metadata (see DispData's src_*/rcv_* attributes; aftan()/aftan_pmf() populate "
              "these automatically from a `corrdata` argument, or accept them as explicit "
              "keyword arguments)." % (n_skip, "y" if n_skip == 1 else "ies"))
    return groups


# ---------------------------------------------------------------------------
# main driver
# ---------------------------------------------------------------------------
def eikonal_tomography(dispdata_list, lon_grid, lat_grid, periods=None, vtype='phase',
                        snr_min=5.0, smooth_km=50.0, cutoff_sigma=3.0, min_stations=4,
                        max_azgap_deg=180.0, min_wavelengths=1.0, far_field_vel=None,
                        resid_zmax=3.0, min_sources_per_node=2, verbose=False):
    """
    Eikonal-equation tomography: assemble per-virtual-source travel-time-gradient
    phase-/group-velocity maps from an ensemble of pairwise DispData dispersion
    measurements, and stack them into a final gridded velocity map (+ per-node,
    per-period uncertainty and source-count coverage) at each requested period.
    See this module's top-of-file docstring for the full method description and
    known approximations/pitfalls.

    ===PARAMETERS===
    dispdata_list: list of DispData objects (e.g. from load_dispdata_ensemble()),
            each with station-pair metadata populated (src_*/rcv_*; see
            DispData's docstring) -- entries missing it are skipped (see
            _group_by_source()).
    lon_grid,lat_grid: 1-D target grid coordinate vectors (deg), e.g. from
            make_grid().
    periods: periods (s) to build maps at. default None: the union of every
            input curve's own period samples, thinned to at most 40 log-spaced
            values if that union is larger.
    vtype: 'phase' [default] or 'group' -- which velocity to tomographically map.
    snr_min: minimum SNR (dB, interpolated to each target period) required to use
            a station pair's measurement at that period. default 5.0.
    smooth_km,cutoff_sigma,min_stations,max_azgap_deg,resid_zmax: passed straight
            to _local_plane_fit_grid() -- see its docstring, especially
            `smooth_km`, the single most consequential tuning parameter.
    min_wavelengths,far_field_vel: near-field exclusion. A receiver is only used
            at a given period if its distance from the source spans at least
            `min_wavelengths` wavelengths (period * far_field_vel); default
            far_field_vel=None uses that pair's own observed velocity as a rough
            proxy. Independently, grid nodes closer to the source than that same
            near-field distance (using, once available, the per-source map's own
            median fitted velocity) are excluded from that source's contribution
            to the stack, since the plane-wave assumption underlying the eikonal
            equation breaks down close to the source regardless of how good any
            individual pair's own measurement is.
    min_sources_per_node: a grid node/period is only reported if at least this
            many independent virtual sources contributed a valid map value there
            -- both for basic reliability (a single source cannot detect its own
            azimuthal bias) and so the per-node standard deviation used as the
            uncertainty estimate is meaningful. default 2.
    verbose: print per-source-period progress and the metadata-skip summary.
            default False.

    ===RETURNS===
    a PhaseVelocityMap (see that class) with .velocity/.uncertainty/.n_sources,
    each shape (len(periods), len(lat_grid), len(lon_grid)).
    """
    groups = _group_by_source(dispdata_list, verbose=verbose)
    if not groups:
        raise ValueError("eikonal_tomography(): no usable station-pair measurements found "
                          "(every DispData is missing station-pair metadata -- see "
                          "src_*/rcv_* on DispData, and aftan()/aftan_pmf()'s corrdata "
                          "auto-fill or explicit src_*/rcv_* keyword arguments).")

    if periods is None:
        all_periods = np.concatenate([d.period for grp in groups.values() for d in grp])
        all_periods = np.unique(all_periods[np.isfinite(all_periods) & (all_periods > 0)])
        if len(all_periods) > 40:
            periods = np.exp(np.linspace(np.log(all_periods.min()), np.log(all_periods.max()), 40))
        else:
            periods = all_periods
    else:
        periods = np.asarray(periods, dtype=np.float64)

    lon_grid = np.asarray(lon_grid, dtype=np.float64)
    lat_grid = np.asarray(lat_grid, dtype=np.float64)
    nlat, nlon, nper = len(lat_grid), len(lon_grid), len(periods)
    LonG, LatG = np.meshgrid(lon_grid, lat_grid)

    vel_stack = np.full((nper, nlat, nlon), np.nan)
    unc_stack = np.full((nper, nlat, nlon), np.nan)
    nsrc_stack = np.zeros((nper, nlat, nlon), dtype=np.int32)

    for ip, period in enumerate(periods):
        per_source_maps = []
        for key, members in groups.items():
            src_net, src_sta, src_lon, src_lat = key
            rlon, rlat, tt = [], [], []
            for d in members:
                v = _curve_velocity_at_period(d, period, vtype=vtype, snr_min=snr_min)
                if not np.isfinite(v) or v <= 0:
                    continue
                dist = d.dist
                if dist is None or dist <= 0:
                    continue
                fv_pair = far_field_vel if far_field_vel is not None else v
                if dist < min_wavelengths * period * fv_pair:
                    continue  # near-field: too close for the plane-wave assumption
                rlon.append(d.rcv_lon)
                rlat.append(d.rcv_lat)
                tt.append(dist / v)
            if len(rlon) < min_stations:
                continue

            fit = _local_plane_fit_grid(
                lon_grid, lat_grid, np.asarray(rlon), np.asarray(rlat), np.asarray(tt),
                smooth_km=smooth_km, cutoff_sigma=cutoff_sigma, min_stations=min_stations,
                max_azgap_deg=max_azgap_deg, resid_zmax=resid_zmax)

            fv_node = far_field_vel if far_field_vel is not None else np.nanmedian(fit['velocity'])
            if not np.isfinite(fv_node):
                fv_node = 3.0  # generic crustal surface-wave fallback; only used if this
                                # source's whole map came back empty (nothing to scale against).
            src_dist_km = _haversine_km(src_lon, src_lat, LonG, LatG)
            near = src_dist_km < min_wavelengths * period * fv_node
            v_this = fit['velocity'].copy()
            v_this[near] = np.nan
            per_source_maps.append(v_this)
            if verbose:
                nvalid = int(np.sum(np.isfinite(v_this)))
                print("  period=%6.2fs source=%s.%s: %d receiver(s) used, "
                      "%d/%d grid node(s) mapped" %
                      (period, src_net, src_sta, len(rlon), nvalid, nlat * nlon))

        if not per_source_maps:
            continue
        stack = np.stack(per_source_maps, axis=0)
        count = np.sum(np.isfinite(stack), axis=0)
        # nodes with zero contributing sources (nanmean/nanstd of an all-NaN slice)
        # are expected -- not every grid node sits within every source's coverage --
        # and are masked out immediately below via `keep`; suppress the resulting
        # "Mean of empty slice"/"Degrees of freedom <= 0" RuntimeWarnings, which are
        # harmless noise here rather than a sign of anything wrong.
        with warnings.catch_warnings(), np.errstate(invalid='ignore'):
            warnings.simplefilter('ignore', category=RuntimeWarning)
            mean_v = np.nanmean(stack, axis=0)
            std_v = np.nanstd(stack, axis=0)
        keep = count >= min_sources_per_node
        mean_v = np.where(keep, mean_v, np.nan)
        std_v = np.where(keep, std_v, np.nan)
        vel_stack[ip] = mean_v
        unc_stack[ip] = std_v
        nsrc_stack[ip] = count

    params = dict(vtype=vtype, snr_min=snr_min, smooth_km=smooth_km, cutoff_sigma=cutoff_sigma,
                  min_stations=min_stations, max_azgap_deg=max_azgap_deg,
                  min_wavelengths=min_wavelengths, far_field_vel=far_field_vel,
                  resid_zmax=resid_zmax, min_sources_per_node=min_sources_per_node)
    return PhaseVelocityMap(periods, lon_grid, lat_grid, vel_stack, unc_stack, nsrc_stack,
                             vtype=vtype, method='eikonal', params=params)


# ---------------------------------------------------------------------------
# result loading -- PhaseVelocityMap itself is defined in seisgo.types (see the
# import at the top of this module), alongside DispData, following this
# codebase's convention of keeping data-container classes in types.py and the
# functions that build/read them in the module that owns the algorithm.
# ---------------------------------------------------------------------------
def read_phase_velocity_map(filename):
    """
    Load a PhaseVelocityMap previously written by PhaseVelocityMap.save().
    """
    with h5py.File(filename, 'r') as f:
        vtype = f.attrs['vtype']
        method = f.attrs['method']
        params = json.loads(f.attrs['params_json'])
        period = np.array(f['period'])
        lon_grid = np.array(f['lon_grid'])
        lat_grid = np.array(f['lat_grid'])
        velocity = np.array(f['velocity'])
        uncertainty = np.array(f['uncertainty'])
        n_sources = np.array(f['n_sources'])
    return PhaseVelocityMap(period, lon_grid, lat_grid, velocity, uncertainty, n_sources,
                             vtype=vtype, method=method, params=params)
