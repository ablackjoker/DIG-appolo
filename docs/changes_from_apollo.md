# Changes From Apollo To Cavity

This branch contains the cavity-case variant derived from the Apollo return-capsule case.

## Main Case Changes

- Boundary defaults are changed from Apollo inlet/outlet/wall to a closed cavity:
  - Apollo: tag 4 inlet, tag 5 outlet, tag 6 wall.
  - Cavity: tag 4 moving top wall, tag 5 stationary wall.
- The default mesh path is changed from `./mesh/3dapollo372500.cas` to `./mesh/cavity_216000_01.cas`.
- The Knudsen number in `setMa_Kn_CFL` is changed from `0.01` to `0.1`.
- Inlet particle preprocessing is disabled because the cavity has no inlet or outlet.
- Mesh/time-sampling constants are changed:
  - `NCELL = 216000`
  - `NSS = 3000`
  - `Nrepeat = 200`
  - `NSCHEME = 500`

## Physical Parameters

- The cavity case uses fixed reference number density:
  - `n_ref = 2.685e25`
  - `T_in = 273.15`
  - `Twall_ref = 1.0`
  - `p_ref = n_ref * kB * T_in`
- The moving wall parameters are set as:
  - `u_wall = 1.0`
  - `T_wall = 1.0`

## Solver Changes

- NS boundary classification no longer applies freestream inlet/open outlet fluxes for the cavity case.
- The top wall uses `Flux_NSEG13_bcWallwithVelocity`.
- The initial NS field is set to a stationary cavity field with `rho = 1.0`, zero velocity, and `T = T_wall`.
- The NS convection flux switches from Rusanov (`convectionFlux(1)`) to SLAU2 (`convectionFlux(2)`).
- The GSIS implicit coefficient changes from `coe_omega = 8.0` to `coe_omega = 5.0`.
- GSIS macro iterations per coupling step change from `1000` to `200`.
- `FourierUpdateDensity()` is disabled and replaced by `origin2Conservation()`.
- DSMC-to-NS macro lower-bound and Kn-cell filters are disabled for this cavity configuration.

## Upload Notes

- `obj/` is intentionally excluded from this branch because it contains compiled object files.
- Check that the mesh file used by `meshImportTest.cpp` exists before running. The code currently defaults to `./mesh/cavity_216000_01.cas`.
