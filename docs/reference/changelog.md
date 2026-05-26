# Changelog

## Unreleased

- Added `obsmaster` module for observatory-grade data management
- Extended `anisotropy` module with BANX beamforming pipeline

## v0.9.x

- Added `clustering` module: k-means and SOM velocity profile clustering
- Added `simulation` module: 1-D finite-difference acoustic solver
- Added `dispersion` module: phase-shift dispersion imaging, narrowband filtering, `surf96` forward solver
- Extended `monitoring.get_dvv` with multi-window, multi-process, and whitening support
- Added `helpers` module for self-describing option lists
- Improved orientation correction in `noise.assemble_raw`

## v0.8.x

- Initial public release
- Core ambient noise pipeline: `noise`, `stacking`, `monitoring`, `utils`, `types`
- `downloaders` module for FDSN data access
- `plotting` module for visualization
- `hvsr` module for site characterization

---

For the full git history see: https://github.com/xtyangpsp/SeisGo/commits/main
