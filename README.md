# DIG Cavity Case

This branch (`cavity-case`) contains the lid-driven cavity case derived from the Apollo return-capsule case. The main code-level changes from Apollo are summarized in `docs/changes_from_apollo.md`.

## Basic Requirements

To compile and run this code on another machine, the basic requirements are:

- Linux or Linux-based HPC environment
- MPI runtime and MPI compiler wrapper, such as Intel MPI, OpenMPI, or MPICH
- C++11-compatible compiler
- METIS library and headers
- GKlib library and headers
- Matching Fluent CAS mesh file
- Enough CPU cores and memory for the selected mesh and particle count

## Build Notes

The current `Makefile` is configured for an Intel/HPC environment:

```makefile
CC := mpiicpc
CFLAGS := -O3 -xSKYLAKE-AVX512 -qopt-zmm-usage=high
```

On a different machine, update the compiler and dependency paths in `Makefile`:

```makefile
MPI_INCLUDE :=
MPI_LIB :=
METIS_INCLUDE :=
METIS_LIB :=
GK_INCLUDE :=
GK_LIB :=
```

For a more portable first build, the compiler settings may be changed to something like:

```makefile
CC := mpic++
CFLAGS := -O3 -std=c++11
```

Then build with:

```bash
make
```

## Mesh And Case Settings

The default mesh path is set in `src/meshImportTest.cpp`:

```text
./mesh/cavity_216000_01.cas
```

The corresponding mesh file is tracked in this branch:

```text
mesh/cavity_216000_01.cas
```

The mesh cell count is hard-coded in `src/meshImport.h`:

```cpp
const int NCELL = 216000;
```

If a different cavity mesh is used, update both the mesh path and `NCELL` so they match.

## Boundary Setup

The cavity case uses closed boundaries. There is no inlet/outlet particle injection.

| Fluent tag | Role | Model | Notes |
| --- | --- | --- | --- |
| `3` | Interface | None | Partition/interface boundary |
| `4` | TopWall | Diffuse/isothermal moving wall | `u_wall = 1.0`, `T_wall = 1.0` |
| `5` | Wall | Diffuse/isothermal stationary wall | `Twall_ref = 1.0` |

The inlet preprocessing call is disabled in `src/meshImportTest.cpp`:

```cpp
// process->preprocesseffquad(istep);
```

## Basic Case Parameters

The current cavity case is configured in the source code with the following basic settings:

| Quantity | Value |
| --- | --- |
| Knudsen number | `Kn = 0.1` |
| Reference temperature | `T_ref = 273.15 K` |
| Cavity gas/reference temperature | `T_in = 273.15 K` |
| Reference number density | `n_ref = 2.685e25` |
| Wall temperature | `Twall_ref = 1.0` |
| Moving wall speed | `u_wall = 1.0` |
| Moving wall temperature | `T_wall = 1.0` |
| Mesh cell count | `NCELL = 216000` |
| Initial particles per cell | `Npinitial = 100` |
| Approximate initial particle count | `21600000` |
| Total DSMC/DIG evolution steps | `NTOTAL = 10000` |
| Exponential averaging startup steps | `NSCHEME = 500` |
| Sampling start step | `NSS = 3000` |
| DIG coupling interval | `NGSIS = 200` |

## Solver Notes

This cavity branch also changes several solver settings from the Apollo case:

- NS initial field is stationary: `rho = 1.0`, `u = v = w = 0.0`, `T = T_wall`.
- Top wall uses the moving-wall G13 boundary flux `Flux_NSEG13_bcWallwithVelocity`.
- NS convection flux uses SLAU2 through `convectionFlux(2)`.
- GSIS implicit coefficient uses `coe_omega = 5.0`.
- GSIS macro solver iterations are set to `nsProcess(200, ...)`.
- `FourierUpdateDensity()` is disabled and `origin2Conservation()` is used after DSMC-to-NS transfer.
- DSMC-to-NS macro lower-bound and Kn-cell filters are disabled for this cavity configuration.
- Output velocity scaling is set to `1.0` for cavity post-processing output.

## Run

`SubmitJob.lsf` is not required for running the code. If MPI is available and the working directory and paths are set correctly, the executable can be run directly with `mpirun`.

Run from the repository root, or update the relative mesh/output paths in the code and run script accordingly:

```bash
mpirun -np 160 ./main_mpiDSMC_ht > log.txt
```

`SubmitJob.lsf` is only an optional example for an LSF cluster. Before using it on another system, update the queue name, core count, module loads, executable path, and any mesh/output paths to match that machine:

```bash
bsub < SubmitJob.lsf
```

At least two MPI processes are required. Rank 0 reads and distributes the mesh, and the compute ranks are `size - 1`.

## Output Directories

The following directories are expected by the code and are tracked with `.gitkeep` files:

```text
cellDistribute/
nsTemp/
statisticResults/
```

DSMC output is written mainly to `statisticResults/`, and NS output is written to `nsTemp/`.

## Notes For GitHub

The `obj/` directory is intentionally ignored and not tracked in this branch because it contains compiled object files.
