#include "meshImport.h"
#include "ProcessDSMC.h"
#include "MeshparticalInitial.h"
#include "MacroSolver.h"
#include "MpiContext.h"
#include <array>
class ProcessGSIS
{
private:


    vector<int> nsStartByRank;
    vector<int> nsEndByRank;

    void refreshNsOwnerRanges(bool force = false);
    int nsOwnerOfGlobalCell(int globalCell) const;
    int nsLocalOfGlobalCell(int globalCell) const;
    int dsmcOwnerOfGlobalCell(int globalCell) const;
    int dsmcLocalOfGlobalCell(int globalCell) const;
    void synchronizeBoundaryTables();

public:
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

    ProcessGSIS();
    ProcessGSIS(meshImport *mesh, ProcessDSMC *process, MeshparticalInitial *partinit, MacroSolver *nsprocess, const MpiContext& mpiCtx);
    ~ProcessGSIS();

    void variablesetup();
    void regsisvariables();
    void NS2DSMC();
    void DSMC2NS1();
    void macro_iter_process(double maxError, int istep);
    void DSMC2NS();
    bool isNsMpiHaloNeighborForLowOrder(const MacroSolver* ns, int cell, int faceSlot, int neighbor) const;
    void syncNsHaloLowOrderState();
    void reconstructLowOrderForNS();
    void dampRejectedHighOrderForNS();
    void molecular_velocity_change();
    bool replication(int icell, int np_modify, double *u_ns, double T_ns, double Tr_ns);
    particle MaxwellianSample(particle part, double *u, double T, double Tr, int icell);
    particle randomlocation(particle part, int icell);
    bool deletion(int icell, int np_modify);
};
