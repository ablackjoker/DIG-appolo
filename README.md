# DIG-appolo

DSMC/NS parallel solver case for the Apollo mesh.

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

The default mesh path is:

```text
./mesh/3dapollo372500.cas
```

The mesh path can also be passed as the first command-line argument:

```bash
mpirun -np 160 ./main_mpiDSMC_ht ./mesh/3dapollo372500.cas
```

The current code has the mesh cell count hard-coded in `src/meshImport.h`:

```cpp
const int NCELL = 372500;
```

If a different mesh is used, make sure this value matches the mesh cell count.

Important default case parameters are also set in the source code, including:

```cpp
const int NTOTAL = 10000;
const int Npinitial = 100;
const int NSS = 5000;
const int NSCHEME = 2000;
const int ifgsis = 1;
```

The main program sets the freestream/case parameters with:

```cpp
mesh->setMa_Kn_CFL(5, 0.01, 1e2, 1e3, 5);
```

## Run

Typical MPI run:

```bash
mpirun -np 160 ./main_mpiDSMC_ht > log.txt
```

For an LSF cluster, use:

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
