/*
 * GSIS coupling state and DSMC/NS transfer API.
 */

#include "meshImport.h"
#include "ProcessDSMC.h"
#include "MeshparticalInitial.h"
#include "MacroSolver.h"
#include "MpiContext.h"
#include <array>
// ProcessGSIS stores state used by this module.
class ProcessGSIS
{
private:
    vector<int> nsStartByRank;
    vector<int> nsEndByRank;
// refreshNsOwnerRanges (force): updates partition ownership or load data.
    void refreshNsOwnerRanges(bool force = false);
// nsOwnerOfGlobalCell (globalCell): updates partition ownership or load data.
    int nsOwnerOfGlobalCell(int globalCell) const;
// nsLocalOfGlobalCell (globalCell): works with mesh topology or geometric intersections.
    int nsLocalOfGlobalCell(int globalCell) const;
// dsmcOwnerOfGlobalCell (globalCell): updates partition ownership or load data.
    int dsmcOwnerOfGlobalCell(int globalCell) const;
// dsmcLocalOfGlobalCell (globalCell): works with mesh topology or geometric intersections.
    int dsmcLocalOfGlobalCell(int globalCell) const;
// synchronizeBoundaryTables: applies boundary-condition behavior.
    void synchronizeBoundaryTables();
public:
// Dsmc2NsCouplingConfig stores state used by this module.
    struct Dsmc2NsCouplingConfig
    {
        bool sampleFilterEnabled = true;
        double macroMinSampleCount = 3.0;
        bool highOrderSampleFilterEnabled = false;
        double highOrderMinSampleCount = 10.0;
        bool macroLowerBoundsEnabled = true;
        bool macroUpperBoundsEnabled = false;
        double rhoMin = 1.0e-8;
        double rhoMax = 1.0e300;
        double tMin = 1.0e-6;
        double tMax = 1.0e300;
        double trMin = 1.0e-6;
        double trMax = 1.0e300;
        double maMin = 0.0;
        double maMax = 50.0;
        bool knCellUpperBoundEnabled = true;
        double knCellMax = 0.3;
        bool highOrderLowerBoundsEnabled = false;
        bool highOrderUpperBoundsEnabled = false;
        double stressRatioMin = 0.0;
        double stressRatioMax = 10.0;
        double heatRatioMin = 0.0;
        double heatRatioMax = 20.0;
        double rotHeatRatioMin = 0.0;
        double rotHeatRatioMax = 20.0;
        bool lowOrderHarmonicEnabled = true;
        int lowOrderHarmonicIterations = 5;
        double lowOrderHarmonicOmega = 0.8;
        double highOrderDampingCoeff = 0.2;
        bool logFilterSummary = true;
    };
// Dsmc2NsMacroSource stores state used by this module.
    enum Dsmc2NsMacroSource : unsigned char
    {
        MACRO_REJECTED = 0,
        MACRO_DIRECT_DSMC = 1,
        MACRO_ACCUMULATED_DSMC = 2,
        MACRO_INTERPOLATED = 3,
        MACRO_OLD_NS = 4
    };
    meshImport *mesh = NULL;
    MacroSolver *pnsSolver = NULL;
    ProcessDSMC *process = NULL;
    MeshparticalInitial *partinit = NULL;
    const MpiContext* mpi = nullptr;
    meshMessage mess; 
    int icell;
    int c_rank, c_size;
    MPI_Comm comm;
    MPI_Comm calGroup;
    vector<double> nsresult;
    vector<char> dsmc2ns_macro_accepted;
    vector<char> ns_dsmc2ns_macro_accepted;
    vector<char> ns_dsmc2ns_high_order_accepted;
    vector<unsigned char> ns_dsmc2ns_macro_source;
    vector<int> ns_loworder_reconstruct_level;
    vector<int> ns_recon_index;
    vector<int> ns_recon_cells;
    vector<array<double, 6>> ns_recon_phi;
    vector<array<double, 6>> ns_recon_phi_new;
    bool dsmc2ns_acceptance_ready = false;
    bool use_exp_weighted_dsmc2ns = true;
    Dsmc2NsCouplingConfig dsmc2nsCoupling;
// ProcessGSIS: couples DSMC data with macroscopic fields.
    ProcessGSIS();
// ProcessGSIS (mesh, process, partinit, nsprocess, mpiCtx): couples DSMC data with macroscopic fields.
    ProcessGSIS(meshImport *mesh, ProcessDSMC *process, MeshparticalInitial *partinit, MacroSolver *nsprocess, const MpiContext& mpiCtx);
// ~ProcessGSIS: releases owned buffers and MPI helper state.
    ~ProcessGSIS();
// variablesetup: prepares derived solver state.
    void variablesetup();
// regsisvariables: couples DSMC data with macroscopic fields.
    void regsisvariables();
// NS2DSMC: couples DSMC data with macroscopic fields.
    void NS2DSMC();
// DSMC2NS1: couples DSMC data with macroscopic fields.
    void DSMC2NS1();
// macro_iter_process (maxError, istep): couples DSMC data with macroscopic fields.
    void macro_iter_process(double maxError, int istep);
// DSMC2NS: couples DSMC data with macroscopic fields.
    void DSMC2NS();
// isNsMpiHaloNeighborForLowOrder (ns, cell, faceSlot, neighbor): performs one solver support operation.
    bool isNsMpiHaloNeighborForLowOrder(const MacroSolver* ns, int cell, int faceSlot, int neighbor) const;
// syncNsHaloLowOrderState: performs one solver support operation.
    void syncNsHaloLowOrderState();
// reconstructLowOrderForNS: couples DSMC data with macroscopic fields.
    void reconstructLowOrderForNS();
// dampRejectedHighOrderForNS: couples DSMC data with macroscopic fields.
    void reconstructHighOrderForNS();
    void dampRejectedHighOrderForNS();
// molecular_velocity_change: updates particles or particle-derived state.
    void molecular_velocity_change();
// replication (icell, np_modify, u_ns, T_ns, Tr_ns): performs one solver support operation.
    bool replication(int icell, int np_modify, double *u_ns, double T_ns, double Tr_ns);
// MaxwellianSample (part, u, T, Tr, icell): updates particles or particle-derived state.
    particle MaxwellianSample(particle part, double *u, double T, double Tr, int icell);
// randomlocation (part, icell): updates particles or particle-derived state.
    particle randomlocation(particle part, int icell);
// deletion (icell, np_modify): performs one solver support operation.
    bool deletion(int icell, int np_modify);
};
