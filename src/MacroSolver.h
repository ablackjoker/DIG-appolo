#pragma once
#include "meshImport.h"
#include "MessagePassing.h"
#include "MpiContext.h"
#include "BoundaryCondition.h"

class MesoSolver;

class MacroSolver
{
private:
public:
    int iSize, iRank, size, rank;
    int iNcell, Ncell, iNface, eNface, Nface, var, Nl, Nr;
    meshMessage mess;
    cell *cells = NULL; edge *edges = NULL; meshImport *mesh = NULL;
    double **Qf = NULL, **Qc = NULL, **RHS = NULL, **impf = NULL;
    double *delta_t = NULL, **ns_sigmaq = NULL, **d_sigmaq = NULL, **Grad = NULL, **Umaxmin = NULL;
    double *Flux = NULL, **qfl = NULL, **qfr = NULL;
    vector<int> bcVis;
    vector<int> symmVisx, symmVisy, symmVisz, wallVis, topVis, farVis, impVis, Pin, Pout, HFS, LFS, OutWall, leftWall, rightWall, massFlowWall, slipWall;
    BoundaryConditionTable boundaryTable;
    int *nParts = NULL, *sendCount = NULL, *recCount = NULL;
    vector<int> sendCell, recCell;
    double *sendBuf = NULL, *recBuf = NULL, *fluxTotal = NULL;
    int *NsreMeIndex = NULL, *NsreMeIndex2 = NULL;
    const MpiContext* mpi = nullptr;
    double Cx[5][5], Cy[9], Cz[9], qxyz[5][4][3], mxyz[5][4][3], wxyz[5][4][3], nxyz[5][4][3], txyz[5][4][3], rxyz[5][4][3];
    MPI_Status *status = NULL; MPI_Request *request = NULL;
    MPI_Comm myGroup;
    MPI_Comm comm;
    MessagePassing *mpass = NULL;
    double *tVector = NULL;
    bool mallocSpace( );
    void bcClassification();
    void dyparcells();
    void scatter_counts(vector<int> counts_or_null,int& my, MPI_Comm comm);
    void cell_cell_initial();
    void edge_cell_initial();
    MacroSolver();
    MacroSolver(meshImport *mesh, meshMessage mess, MessagePassing *mpass, const MpiContext& mpiCtx);
    ~MacroSolver();
    bool active() const { return mpi != nullptr ? mpi->active() : myGroup != MPI_COMM_NULL; }
    bool root() const { return mpi != nullptr ? mpi->root() : myGroup == MPI_COMM_NULL; }
    bool activeLeader() const { return active() && iRank == 0; }
    bool initialW();
    bool initialBC();
    bool noreconstrucion(bool gsisTag);
    bool reconstruction(bool gsisTag);
    bool updateBC(bool gsisTag);
    bool applyNSEG13BoundaryFluxes();
    bool noSlipIsothermalWall2(double *qf, vector<int> wallVis);
    bool pressureOutlet(double pout, vector<int> wallVis);
    bool massWall(double fin, double t, vector<int> wallVis);
    bool symmetry(vector<int> symm);
    void rhs2Zero();
    bool calcTimestep();
    bool calcSource();
    bool Rusanov(bool gsisTag);
    bool SLAU2(bool gsisTag);
    bool ROEflux(bool gsisTag);
    bool AUSMPWPlus(bool gsisTag);
    void calcCellViscous();
    bool viscousFlux(bool gsisTag);
    bool implicitSolver(bool gsisTag);
    void eigen(double *lambda);
    bool normalFlux(double *qf, double *Qc, double *enormal);
    bool nsProcess(int maxIter, double maxError, bool gsisTag, int ITER);
    bool nsProcessG13(int maxIter, double maxError, bool gsisTag, int ITER);
    bool convectionFlux(int tag, bool gsisTag);
    double convectionFunction(double *qf, double *Qc, double *enormal);
    bool qf2Qc(double *qf, double *Qc);
    bool Qc2qf(double *Qc, double *qf);
    bool conservation2Origin();
    bool origin2Conservation();
    void FourierUpdateDensity();
    double psLeft(const double ma);
    double psRight(const double ma);
    double msLeft(const double ma);
    double msRight(const double ma);
    bool slau2Function(double *fl, double *fr, double *ul, double *ur, double &m, double &pt, const double *enormal);
    bool slau2Function2(const double *f, double *ut, const double *enormal);
    void clacSlauPsi(const double *qf, double *psi);
    void ausmFunction(double *fl, double *fr, double &mlpt, double &mrnt, double &ps, const double *norm);
    double ausmF(const double &p, const double &ps, const double &m);
    void initialParts();
    void initialParallel();
    void freeParallel();
    void packQfQc();
    void unPackQfQc();
    void unPackFlux();
    void packDeltaW();
    void unPackDeltaW();
    bool leastSquareGrad();
    bool limiter(bool gsisTag);  
    bool limiter2(bool gsisTag);
    bool numpyDeepCopy2D(int m, int n, double **A, double **B); 
    double calcMaxError(double **wt, double **qf);
    double calcError(int i, double **wt, double **qf);
    bool out2dat(const char *filename);
    bool nsout2dat(int istep, double **Q,int startcell, int endcell);
    void cellDeepCopy(cell &c1, const cell &c2);
    void edgeDeepCopy(edge &e1, const edge &e2);
    void calcCellHot(int istep);
    double BarthJespersen(const double &umax, const double &umin, const double &u, const double &g);
    double Venkatakrishnan1(const double &umax, const double &umin, const double &u, const double &g, const double &area);
    double Venkatakrishnan2(const double &umax, const double &umin, const double &u, const double &g, const double &area);
    double Venkatakrishnan3(const double &umax, const double &umin, const double &u, const double &g, const double &area);
    double Venkatakrishnan5(const double &umax, const double &umin, const double &u, const double &g, const double &area);
    double Wang(const double &umax, const double &umin, const double &u, const double &g);
    double minLimter(const double &umax, const double &umin, const double &u, const double &g, const double &area);
    bool Flux_NSEG13_bcWall(vector<int> wallVis);
    bool Flux_NSEG13_bcWallWithT(double T, vector<int> wallVis);
    bool Flux_NSEG13_bcWallwithVelocity(double T, double u, vector<int> wallVis);
    bool Flux_NSEG13_bcWallNew(const double *wf, vector<int> wallVis);
    bool Flux_NSEG13_inlet(vector<int> wallVis);
    bool Flux_NSEG13_outlet(vector<int> wallVis);
    bool Flux_NSEG13_massIn(double fin, double t, vector<int> wallVis);
    bool Flux_NSEG13_pressureIn(double pin, double t, vector<int> wallVis);
    bool Flux_NSEG13_pressureOut(double pout, vector<int> wallVis); 
    bool calcFlux_IFV2(double *WI, double *sq, double *enormal, double *IFv, double *a, bool tag, int index);
    void calcG13Sigma(int l, double *gm, double *ds, double *sq);
    void CalculateTangentialVector();
    void calcMassFlow(vector<int> wallVis);
    void AUSM_Plus_Up(bool gsisTag);
    void ausmPlusUpFunction(double *fl, double *fr, double &mlpt, double &mrnt, double &ps, const double *norm);
    double maSplit1(double ma, const int &sgn);
    double maSplit2(double ma, const int &sgn);
    double maSplit4(double ma, const int &sgn, const double &beta);
    double pSplit5(double ma, const int &sgn, const double &alpha);
};
