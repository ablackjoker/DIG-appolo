#pragma once
# include "meshImport.h"
# include "MeshparticalInitial.h"
#include "MpiContext.h"
#include "BoundaryCondition.h"
#include "MeshPartitionTransfer3D.h"
# include <mpi.h>
#include <array>
#include <random>
# include <climits>
# include <deque>
# include <unordered_map>
# include <map>
# include <string>
# include <ostream>
# include <vector>

class ProcessDSMC
{
private:
    enum class ParticleTraceOutcome { LocalDone, SentRemote, Dropped };
    enum MacroIndex
    {
        MACRO_RHO = 0,
        MACRO_UX,
        MACRO_UY,
        MACRO_UZ,
        MACRO_T,
        MACRO_SXX,
        MACRO_SXY,
        MACRO_SYY,
        MACRO_SYZ,
        MACRO_SZZ,
        MACRO_SZX,
        MACRO_QX,
        MACRO_QY,
        MACRO_QZ,
        MACRO_TR,
        MACRO_QRX,
        MACRO_QRY,
        MACRO_QRZ,
        MACRO_WIDTH
    };
    enum StressIndex
    {
        STRESS_XX = 0,
        STRESS_XY,
        STRESS_YY,
        STRESS_YZ,
        STRESS_ZZ,
        STRESS_ZX,
        STRESS_WIDTH
    };
    enum HeatIndex
    {
        HEAT_X = 0,
        HEAT_Y,
        HEAT_Z,
        HEAT_WIDTH
    };
    enum AxisIndex
    {
        AXIS_X = 0,
        AXIS_Y,
        AXIS_Z,
        AXIS_WIDTH
    };
    enum RotHeatIndex
    {
        ROT_ENERGY = 0,
        ROT_HEAT_X,
        ROT_HEAT_Y,
        ROT_HEAT_Z,
        ROT_HEAT_WIDTH
    };
    enum FluxIndex
    {
        FLUX_RHO = 0,
        FLUX_UX,
        FLUX_UY,
        FLUX_UZ,
        FLUX_ENERGY,
        FLUX_ROT,
        FLUX_WIDTH
    };
    enum WallNormalIndex
    {
        WALL_N1X = 0,
        WALL_N1Y,
        WALL_N1Z,
        WALL_N2X,
        WALL_N2Y,
        WALL_N2Z,
        WALL_NORMAL_WIDTH
    };
    struct FaceTriCache
    {
        int face = -1;
        int triTag = -1;
        double ax = 0.0, ay = 0.0, az = 0.0;
        double e1x = 0.0, e1y = 0.0, e1z = 0.0;
        double e2x = 0.0, e2y = 0.0, e2z = 0.0;
    };
    struct FaceCrossCache
    {
        int gface = -1;
        int globalCell0 = -1;
        int globalCell1 = -1;
        int localCell0 = -1;
        int localCell1 = -1;
        int globalNextFromCell0 = -1;
        int globalNextFromCell1 = -1;
        int localNextFromCell0 = -1;
        int localNextFromCell1 = -1;
        int ownerNextFromCell0 = -1;
        int ownerNextFromCell1 = -1;
    };
    struct FacePointSet
    {
        int faceType = -1;
        int nodeIds[4] = {-1, -1, -1, -1};
        double points[4][3] = {{0.0, 0.0, 0.0},
                               {0.0, 0.0, 0.0},
                               {0.0, 0.0, 0.0},
                               {0.0, 0.0, 0.0}};
    };
    struct FaceTriangleList
    {
        int count = 0;
        int nodes[2][3] = {{-1, -1, -1}, {-1, -1, -1}};
        int tag[2] = {-1, -1};
        unsigned char splitTag = meshImport::FACE_SPLIT_INVALID;
    };
    struct IntersectionHit
    {
        int face = -2;
        int tri = -1;
        double t = 0.0;
    };
    struct MacroMoments
    {
        double rho = 0.0;
        double velocity[AXIS_WIDTH] = {0.0, 0.0, 0.0};
        double kinetic = 0.0;
        double stress[STRESS_WIDTH] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double heat[HEAT_WIDTH] = {0.0, 0.0, 0.0};
        double rot[ROT_HEAT_WIDTH] = {0.0, 0.0, 0.0, 0.0};
    };
    struct BoundaryEmitPatch
    {
        DsmcReservoirBoundaryState reservoirState = DSMC_RESERVOIR_INLET;
        int boundaryTag = -1;
        BoundaryCondition boundary;
        int edgeid = -1;
        int cellLocal = -1;
        int triTag = -1;
        int faceType = -1;
        unsigned char splitTag = meshImport::FACE_SPLIT_INVALID;
        std::array<double, 3> p0{{0.0, 0.0, 0.0}};
        std::array<double, 3> e10{{0.0, 0.0, 0.0}};
        std::array<double, 3> e20{{0.0, 0.0, 0.0}};
        std::array<double, 3> normal{{0.0, 0.0, 0.0}};
        double normalMag = 0.0;
        double area = 0.0;
        double rotation[3][3] = {{0.0, 0.0, 0.0},
                                 {0.0, 0.0, 0.0},
                                 {0.0, 0.0, 0.0}};
    };
    struct BoundaryGasState
    {
        double rhoPhysical = 0.0;
        double uxPhysical = 0.0;
        double uyPhysical = 0.0;
        double uzPhysical = 0.0;
        double tPhysical = 0.0;
        double sx = 0.0;
        double sy = 0.0;
        double sz = 0.0;
        double stp1 = 0.0;
        double stp2 = 0.0;
        double snp = 0.0;
        double fs1 = 0.0;
        double fs2 = 0.0;
        double tNormalized = 0.0;
    };
    struct BoundarySurfaceTally
    {
        double hits = 0.0;
        std::array<double, 3> momentumDelta{{0.0, 0.0, 0.0}};
        double energyTransDelta = 0.0;
        double energyRotDelta = 0.0;
    };
    struct ParticleTransportOps
    {
        static bool mapIncomingFaceCellLocal(
            const ProcessDSMC& process,
            int gface, int gcell,
            int& crossfid_local,
            int& cellid_local);
        static ParticleTraceOutcome traceParticleDtleft(
            ProcessDSMC& self, particle& part, int& cellid, int& crossfid,
            int& tag_triangle, double& dtleft, int& ifout, bool inclusive_hit);
    };
    vector<int> cellTriOffset;
    vector<FaceTriCache> cellTriCache;
    vector<FaceCrossCache> faceCrossCache;
    vector<BoundaryEmitPatch> boundaryEmitCache;
    bool boundaryEmitCacheDirty = true;
    unordered_map<int, BoundarySurfaceTally> boundarySteadyTally;
    vector<unsigned int> cacheTouchedStamp;
    vector<int> cacheTouchedCells;
    vector<double> out2dat_buffer;
    unsigned int cacheTouchedEpoch = 1u;
    void rebuildFaceTraversalCache();
    void rebuildFaceCrossCache();
    void rebuildBoundaryEmitCache();
    const DsmcReservoirBoundaryConfig& reservoirConfigForState(DsmcReservoirBoundaryState state) const;
    bool shouldInjectBoundary(DsmcReservoirBoundaryState state) const;
    BoundaryGasState computeBoundaryGasState(const BoundaryEmitPatch& patch);
    int emitParticlesFromBoundaryPatch(const BoundaryEmitPatch& patch, double& expectedCount);
    int computeBoundaryInsertCount(const BoundaryEmitPatch& patch, BoundaryGasState& gas);
    particle sampleBoundaryParticle(const BoundaryEmitPatch& patch, const BoundaryGasState& gas);
    void traceBoundaryParticle(particle& part, const BoundaryEmitPatch& patch, double dtleft);
    bool intersectCachedTri(const FaceTriCache& tri, const particle* part, double& t) const;
    bool tryCachedCellIntersection(
        particle* part, int Ncell_id, double& min_t, int acrossfid,
        int& tag_triangle, int& hit_face) const;
    bool loadFacePointsSafe(int faceid, FacePointSet& facePoints) const;
    bool decodeFaceTriangles(int faceid, const FacePointSet& facePoints,
                             FaceTriangleList& triangles) const;
    void intersectFaceTriangles(int faceid, int skipTriTag, particle* part,
                                IntersectionHit& bestHit);
    int intersectFallbackCell(particle* part, int cellLocal, int acrossFace,
                              int previousTri, double& min_t, int& tag_triangle);
    static int macroOffset(int localCell, MacroIndex index);
    static int axisOffset(int localCell, AxisIndex index);
    static int heatOffset(int localCell, HeatIndex index);
    static int stressOffset(int localCell, StressIndex index);
    static int rotHeatOffset(int localCell, RotHeatIndex index);
    void accumulateCellMoments(int localCell, const ParticleBucketSoA& bucket,
                               MacroMoments& moments);
    MacroMoments loadMacroMoments(int localCell, const vector<double>& rho,
                                  const vector<double>& velocity,
                                  const vector<double>& kinetic,
                                  const vector<double>& stress,
                                  const vector<double>& heat,
                                  const vector<double>& rot) const;
    void accumulateStepMomentsInto(int localCell, vector<double>& rho,
                                   vector<double>& velocity,
                                   vector<double>& kinetic,
                                   vector<double>& stress,
                                   vector<double>& heat,
                                   vector<double>& rot,
                                   double historyWeight);
    void updateStepWindowMoments(int localCell, int istep);
    void writeAveragedMacro(int localCell, const MacroMoments& moments,
                            double sampleCount, vector<double>& target);
    void guardInvalidMacro(vector<double>& target, int localCell);
    void writeLocalMacro(int localCell, const MacroMoments& moments);
    void writeRecordMacro(int localCell, const MacroMoments& moments);
    bool ownsBoundaryFace(int localFace) const;
    void recordBoundaryStressHeat(int localFace, const double velocityPre[3],
                                  const double velocityPost[3],
                                  double rotPre, double rotPost);
public:
    enum Dsmc2NsSparseState : unsigned char
    {
        DSMC2NS_SPARSE_NORMAL = 0,
        DSMC2NS_SPARSE_ACCUMULATING = 1,
        DSMC2NS_SPARSE_RELEASED = 2
    };
    vector<double> steady_rho, steady_T, steady_U, steady_sigma, steady_q, steady_qr, step_rho, step_T, step_U, step_sigma, step_q;
    vector<double> step_qr, stepinter_rho, stepinter_T, stepinter_U, stepinter_sigma, stepinter_q, stepinter_qr;
    vector<double> stepsum_rho, stepsum_T, stepsum_U, stepsum_sigma, stepsum_q, stepsum_qr;
    vector<double> record, final_record, local;
    vector<double> dsmc2ns_window_samples;
    vector<char> dsmc2ns_window_valid;
    vector<unsigned char> dsmc2ns_sparse_state;
    vector<int> dsmc2ns_sparse_accum_steps;
    using Flux6 = array<double, FLUX_WIDTH>;
    using WallNormalPair = array<double, WALL_NORMAL_WIDTH>;
    vector<Flux6> crossFlux;
    vector<Flux6> nsboundaryflux;
    vector<Flux6> crossFlux_statistic;
    int tol_collision_pairs;
    int tol_collision_times;
    int istep;
    MPI_Comm calGroup;
    MPI_Comm comm;
    meshMessage mess;
    vector<DsmcCell> cells;
    vector<DsmcEdge> edges;
    vector<double> localPointXY;
    vector<unsigned char> faceSplitTag;
    int rank, size, c_rank, c_size; 
    int iNcell, nface;
    int ncell, startGrids, endGrids;
    int Madata = MACRO_WIDTH;
    vector<int> rank_cell_all;
    PartitionState3D partitionState;
    vector<int> old_local_cell;
    vector<int> local_cells;vector<int> face_gids; 
    unordered_map<int,int> gid2local;unordered_map<int,int> face_gid2local;
    vector<int> faceGid2LocalDense;
    vector<int> migrationPeerRanks;
    vector<char> migrationPeerMask;
    vector<particle> recv_cache;
    vector<vector<particle>> migrate_send_particles, migrate_recv_particles;
    vector<vector<DtleftPacket>> dtleft_send_packets, dtleft_recv_packets;
    unordered_map<std::size_t, int> partition_edge_flux;
    bool enable_partition_flux_weights = true;
    bool enable_partition_time_weights = true;
    unordered_map<int, double> cell_time_weight_accum;
    vector<double> cell_time_weight_ema;
    double cell_time_weight_ema_alpha = 0.5;
    vector<int> wallVis, inletVis, outletVis, symmetry, topwall;
    vector<int> wallVisMove, inletVisMove, outletVisMove,topwallMove;
    BoundaryConditionTable boundaryTable;
    meshImport *mesh = NULL;
    MeshparticalInitial *partinit = NULL;
    const MpiContext* mpi = nullptr;
    map<int, WallNormalPair> wallMap;
    ProcessDSMC();
    ~ProcessDSMC();
    ProcessDSMC(meshImport *mesh, meshMessage mess, MeshparticalInitial *partinit,const MpiContext& mpiCtx);
    void macro_list_setup();
    void initial_chache_storage();
    void variable_deep_copy();
    bool boundaryClassification();
    void wall_normal();
    void rebuildBoundaryDerivedState();
    int ownerOfGlobalCell(int gid) const;
    int localOfGlobalCell(int gid) const;
    int localOfGlobalFace(int gid) const;
    bool isOwnedLocalCell(int local) const;
    int globalOfLocalCell(int local) const;
    ParticleBucketSoA &currParticles(int globalCell);
    const ParticleBucketSoA &currParticles(int globalCell) const;
    int particleCount(int globalCell) const;
    bool checkParticleBucketConsistency(const char* stage) const;
    void clearNextParticleBuffers();
    void clearNextParticles(int globalCell);
    void installLocalGeometry(LocalGeometry3D&& geometry);
    void rebuildFaceLookup();
    void rebuild_migration_peer_cache();
    void ensure_partition_flux_storage();
    void reset_partition_flux_counts();
    void record_partition_flux(int oldCell, int endOldCell, int startLocalCell);
    void ensure_partition_time_storage();
    void reset_partition_time_weights();
    void accumulate_cell_time_weight(int globalCell, double dt);
    void macro_list_delete();
    void processFacesQuadNormals(const vector<int>& visList);
    void advection(int istep);
    void tryTri(int faceid, int triTag, double *A,  double *B,  double *C, particle* part, double& bestT, int& bestFace, int& bestTri);
    void loadPoint(int idx, double *P);
    int check_intersection2(particle* part, int Ncell_id, double& min_t, int acrossfid,int& tag_triangle);
    int tetra_intersection(particle* part, int Ncell_id, double& min_t, int acrossfid);
    void compute_plane_equation(double* Apointer,double* Bpointer, double* Cpointer,double* a, double* b, double* c, double* d);
    bool ray_triangle_intersect(double* location, double* velocity, double* Apointer,double* Bpointer, double* Cpointer, double& t, double& u, double& v);
    void CrossboundarySwitch(particle* part, int& cellid, int& ifout, double& dtleft,int crossfid,double t,int triangle_tag);
    void normalize(double vec[3]);
    void buildRotationMatrix(double* norm, double (*R)[3]);
    void preprocesseffquad(int istep);
    void dt_ray_crosscell();
    void scalar_product(double* norm1, double* norm2, double scalar, int dim);
    void cache2chain();
    bool migrateParticlesOnce();
    bool cache2chain_dt(int localNeedFull, int globalNeedFull);
    void current_macro_zero();
    void stepinter_macro_zero(int icell);
    void statistic_macroPre();
    void statistic_macro(int istep);
    bool out2dat(int istep);
    bool outBoundaryStressHeat(int istep);
    double collision_diameter(double cr);
    double collision_diameter(double cr, double collisionDiameterPrefactor);
    void collisionDSMC();
    void collisionVelocity(ParticleBucketSoA& bucket, int idx1, int idx2, int& tol_collision_times, double consant_crm, double& crm, double collisionDiameterPrefactor);
};
