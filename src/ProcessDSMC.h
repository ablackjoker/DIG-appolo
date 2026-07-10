/*
 * Core DSMC transport, collision, statistic, and output interface.
 */

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

// ProcessDSMC stores state used by this module.
class ProcessDSMC
{
private:
    enum class ParticleTraceOutcome { LocalDone, SentRemote, Dropped };
// MacroIndex stores state used by this module.
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
// StressIndex stores state used by this module.
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
// HeatIndex stores state used by this module.
    enum HeatIndex
    {
        HEAT_X = 0,
        HEAT_Y,
        HEAT_Z,
        HEAT_WIDTH
    };
// AxisIndex stores state used by this module.
    enum AxisIndex
    {
        AXIS_X = 0,
        AXIS_Y,
        AXIS_Z,
        AXIS_WIDTH
    };
// RotHeatIndex stores state used by this module.
    enum RotHeatIndex
    {
        ROT_ENERGY = 0,
        ROT_HEAT_X,
        ROT_HEAT_Y,
        ROT_HEAT_Z,
        ROT_HEAT_WIDTH
    };
// FluxIndex stores state used by this module.
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
// WallNormalIndex stores state used by this module.
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
// FaceTriCache stores state used by this module.
    struct FaceTriCache
    {
        int face = -1;
        int triTag = -1;
        double ax = 0.0, ay = 0.0, az = 0.0;
        double e1x = 0.0, e1y = 0.0, e1z = 0.0;
        double e2x = 0.0, e2y = 0.0, e2z = 0.0;
    };
// FaceCrossCache stores state used by this module.
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
// FacePointSet stores state used by this module.
    struct FacePointSet
    {
        int faceType = -1;
        int nodeIds[4] = {-1, -1, -1, -1};
        double points[4][3] = {{0.0, 0.0, 0.0},
                               {0.0, 0.0, 0.0},
                               {0.0, 0.0, 0.0},
                               {0.0, 0.0, 0.0}};
    };
// FaceTriangleList stores state used by this module.
    struct FaceTriangleList
    {
        int count = 0;
        int nodes[2][3] = {{-1, -1, -1}, {-1, -1, -1}};
        int tag[2] = {-1, -1};
        unsigned char splitTag = meshImport::FACE_SPLIT_INVALID;
    };
// IntersectionHit stores state used by this module.
    struct IntersectionHit
    {
        int face = -2;
        int tri = -1;
        double t = 0.0;
    };
// MacroMoments stores state used by this module.
    struct MacroMoments
    {
        double rho = 0.0;
        double velocity[AXIS_WIDTH] = {0.0, 0.0, 0.0};
        double kinetic = 0.0;
        double stress[STRESS_WIDTH] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double heat[HEAT_WIDTH] = {0.0, 0.0, 0.0};
        double rot[ROT_HEAT_WIDTH] = {0.0, 0.0, 0.0, 0.0};
    };
// BoundaryEmitPatch stores state used by this module.
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
// BoundaryGasState stores state used by this module.
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
// BoundarySurfaceTally stores state used by this module.
    struct BoundarySurfaceTally
    {
        double hits = 0.0;
        std::array<double, 3> momentumDelta{{0.0, 0.0, 0.0}};
        double energyTransDelta = 0.0;
        double energyRotDelta = 0.0;
    };
// ParticleTransportOps stores state used by this module.
    struct ParticleTransportOps
    {
// mapIncomingFaceCellLocal (process, gface, gcell, crossfid_local, cellid_local): works with mesh topology or geometric intersections.
        static bool mapIncomingFaceCellLocal(
            const ProcessDSMC& process,
            int gface, int gcell,
            int& crossfid_local,
            int& cellid_local);
// traceParticleDtleft (self, part, cellid, crossfid, tag_triangle, dtleft, ifout, inclusive_hit): updates particles or particle-derived state.
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
// rebuildFaceTraversalCache: works with mesh topology or geometric intersections.
    void rebuildFaceTraversalCache();
// rebuildFaceCrossCache: works with mesh topology or geometric intersections.
    void rebuildFaceCrossCache();
// rebuildBoundaryEmitCache: applies boundary-condition behavior.
    void rebuildBoundaryEmitCache();
// reservoirConfigForState (state): performs one solver support operation.
    const DsmcReservoirBoundaryConfig& reservoirConfigForState(DsmcReservoirBoundaryState state) const;
// shouldInjectBoundary (state): applies boundary-condition behavior.
    bool shouldInjectBoundary(DsmcReservoirBoundaryState state) const;
// computeBoundaryGasState (patch): applies boundary-condition behavior.
    BoundaryGasState computeBoundaryGasState(const BoundaryEmitPatch& patch);
// emitParticlesFromBoundaryPatch (patch, expectedCount): updates particles or particle-derived state.
    int emitParticlesFromBoundaryPatch(const BoundaryEmitPatch& patch, double& expectedCount);
// computeBoundaryInsertCount (patch, gas): applies boundary-condition behavior.
    int computeBoundaryInsertCount(const BoundaryEmitPatch& patch, BoundaryGasState& gas);
// sampleBoundaryParticle (patch, gas): updates particles or particle-derived state.
    particle sampleBoundaryParticle(const BoundaryEmitPatch& patch, const BoundaryGasState& gas);
// traceBoundaryParticle (part, patch, dtleft): updates particles or particle-derived state.
    void traceBoundaryParticle(particle& part, const BoundaryEmitPatch& patch, double dtleft);
// intersectCachedTri (tri, part, t): works with mesh topology or geometric intersections.
    bool intersectCachedTri(const FaceTriCache& tri, const particle* part, double& t) const;
// tryCachedCellIntersection (part, Ncell_id, min_t, acrossfid, tag_triangle, hit_face): works with mesh topology or geometric intersections.
    bool tryCachedCellIntersection(
        particle* part, int Ncell_id, double& min_t, int acrossfid,
        int& tag_triangle, int& hit_face) const;
// loadFacePointsSafe (faceid, facePoints): updates partition ownership or load data.
    bool loadFacePointsSafe(int faceid, FacePointSet& facePoints) const;
// decodeFaceTriangles (faceid, facePoints, triangles): works with mesh topology or geometric intersections.
    bool decodeFaceTriangles(int faceid, const FacePointSet& facePoints,
                             FaceTriangleList& triangles) const;
// intersectFaceTriangles (faceid, skipTriTag, part, bestHit): works with mesh topology or geometric intersections.
    void intersectFaceTriangles(int faceid, int skipTriTag, particle* part,
                                IntersectionHit& bestHit);
// intersectFallbackCell (part, cellLocal, acrossFace, previousTri, min_t, tag_triangle): works with mesh topology or geometric intersections.
    int intersectFallbackCell(particle* part, int cellLocal, int acrossFace,
                              int previousTri, double& min_t, int& tag_triangle);
// macroOffset (localCell, index): couples DSMC data with macroscopic fields.
    static int macroOffset(int localCell, MacroIndex index);
// axisOffset (localCell, index): performs one solver support operation.
    static int axisOffset(int localCell, AxisIndex index);
// heatOffset (localCell, index): performs one solver support operation.
    static int heatOffset(int localCell, HeatIndex index);
// stressOffset (localCell, index): performs one solver support operation.
    static int stressOffset(int localCell, StressIndex index);
// rotHeatOffset (localCell, index): performs one solver support operation.
    static int rotHeatOffset(int localCell, RotHeatIndex index);
// accumulateCellMoments (localCell, bucket, moments): works with mesh topology or geometric intersections.
    void accumulateCellMoments(int localCell, const ParticleBucketSoA& bucket,
                               MacroMoments& moments);
// loadMacroMoments (localCell, rho, velocity, kinetic, stress, heat, rot): updates partition ownership or load data.
    MacroMoments loadMacroMoments(int localCell, const vector<double>& rho,
                                  const vector<double>& velocity,
                                  const vector<double>& kinetic,
                                  const vector<double>& stress,
                                  const vector<double>& heat,
                                  const vector<double>& rot) const;
// accumulateStepMomentsInto (localCell, rho, velocity, kinetic, stress, heat, rot, historyWeight): performs one solver support operation.
    void accumulateStepMomentsInto(int localCell, vector<double>& rho,
                                   vector<double>& velocity,
                                   vector<double>& kinetic,
                                   vector<double>& stress,
                                   vector<double>& heat,
                                   vector<double>& rot,
                                   double historyWeight);
// updateStepWindowMoments (localCell, istep): writes solver fields or diagnostics.
    void updateStepWindowMoments(int localCell, int istep);
// writeAveragedMacro (localCell, moments, sampleCount, target): writes solver fields or diagnostics.
    void writeAveragedMacro(int localCell, const MacroMoments& moments,
                            double sampleCount, vector<double>& target);
// guardInvalidMacro (target, localCell): couples DSMC data with macroscopic fields.
    void guardInvalidMacro(vector<double>& target, int localCell);
// writeLocalMacro (localCell, moments): writes solver fields or diagnostics.
    void writeLocalMacro(int localCell, const MacroMoments& moments);
// writeRecordMacro (localCell, moments): writes solver fields or diagnostics.
    void writeRecordMacro(int localCell, const MacroMoments& moments);
// ownsBoundaryFace (localFace): works with mesh topology or geometric intersections.
    bool ownsBoundaryFace(int localFace) const;
// recordBoundaryStressHeat (localFace, velocityPre, velocityPost, rotPre, rotPost): applies boundary-condition behavior.
    void recordBoundaryStressHeat(int localFace, const double velocityPre[3],
                                  const double velocityPost[3],
                                  double rotPre, double rotPost);
public:
// Dsmc2NsSparseState stores state used by this module.
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
    vector<int> wallVis, inletVis, outletVis, symmetry;
    vector<int> wallVisMove, inletVisMove, outletVisMove;
    BoundaryConditionTable boundaryTable;
    meshImport *mesh = NULL;
    MeshparticalInitial *partinit = NULL;
    const MpiContext* mpi = nullptr;
    map<int, WallNormalPair> wallMap;
// ProcessDSMC: performs one solver support operation.
    ProcessDSMC();
// ~ProcessDSMC: releases owned buffers and MPI helper state.
    ~ProcessDSMC();
// ProcessDSMC (mesh, mess, partinit, mpiCtx): performs one solver support operation.
    ProcessDSMC(meshImport *mesh, meshMessage mess, MeshparticalInitial *partinit,const MpiContext& mpiCtx);
// macro_list_setup: couples DSMC data with macroscopic fields.
    void macro_list_setup();
// initial_chache_storage: prepares derived solver state.
    void initial_chache_storage();
// variable_deep_copy: performs one solver support operation.
    void variable_deep_copy();
// boundaryClassification: parses mesh or configuration input.
    bool boundaryClassification();
// wall_normal: applies boundary-condition behavior.
    void wall_normal();
// rebuildBoundaryDerivedState: applies boundary-condition behavior.
    void rebuildBoundaryDerivedState();
// ownerOfGlobalCell (gid): updates partition ownership or load data.
    int ownerOfGlobalCell(int gid) const;
// localOfGlobalCell (gid): works with mesh topology or geometric intersections.
    int localOfGlobalCell(int gid) const;
// localOfGlobalFace (gid): works with mesh topology or geometric intersections.
    int localOfGlobalFace(int gid) const;
// isOwnedLocalCell (local): works with mesh topology or geometric intersections.
    bool isOwnedLocalCell(int local) const;
// globalOfLocalCell (local): works with mesh topology or geometric intersections.
    int globalOfLocalCell(int local) const;
// currParticles (globalCell): updates particles or particle-derived state.
    ParticleBucketSoA &currParticles(int globalCell);
// currParticles (globalCell): updates particles or particle-derived state.
    const ParticleBucketSoA &currParticles(int globalCell) const;
// particleCount (globalCell): updates particles or particle-derived state.
    int particleCount(int globalCell) const;
// checkParticleBucketConsistency (stage): updates particles or particle-derived state.
    bool checkParticleBucketConsistency(const char* stage) const;
// clearNextParticleBuffers: updates particles or particle-derived state.
    void clearNextParticleBuffers();
// clearNextParticles (globalCell): updates particles or particle-derived state.
    void clearNextParticles(int globalCell);
// installLocalGeometry (geometry): works with mesh topology or geometric intersections.
    void installLocalGeometry(LocalGeometry3D&& geometry);
// rebuildFaceLookup: works with mesh topology or geometric intersections.
    void rebuildFaceLookup();
// rebuild_migration_peer_cache: prepares derived solver state.
    void rebuild_migration_peer_cache();
// ensure_partition_flux_storage: updates partition ownership or load data.
    void ensure_partition_flux_storage();
// reset_partition_flux_counts: updates partition ownership or load data.
    void reset_partition_flux_counts();
// record_partition_flux (oldCell, endOldCell, startLocalCell): updates partition ownership or load data.
    void record_partition_flux(int oldCell, int endOldCell, int startLocalCell);
// ensure_partition_time_storage: updates partition ownership or load data.
    void ensure_partition_time_storage();
// reset_partition_time_weights: updates partition ownership or load data.
    void reset_partition_time_weights();
// accumulate_cell_time_weight (globalCell, dt): works with mesh topology or geometric intersections.
    void accumulate_cell_time_weight(int globalCell, double dt);
// macro_list_delete: couples DSMC data with macroscopic fields.
    void macro_list_delete();
// processFacesQuadNormals (visList): works with mesh topology or geometric intersections.
    void processFacesQuadNormals(const vector<int>& visList);
// advection (istep): updates particles or particle-derived state.
    void advection(int istep);
// tryTri (faceid, triTag, A, B, C, part, bestT, bestFace, bestTri): works with mesh topology or geometric intersections.
    void tryTri(int faceid, int triTag, double *A,  double *B,  double *C, particle* part, double& bestT, int& bestFace, int& bestTri);
// loadPoint (idx, P): updates partition ownership or load data.
    void loadPoint(int idx, double *P);
// check_intersection2 (part, Ncell_id, min_t, acrossfid, tag_triangle): works with mesh topology or geometric intersections.
    int check_intersection2(particle* part, int Ncell_id, double& min_t, int acrossfid,int& tag_triangle);
// tetra_intersection (part, Ncell_id, min_t, acrossfid): works with mesh topology or geometric intersections.
    int tetra_intersection(particle* part, int Ncell_id, double& min_t, int acrossfid);
// compute_plane_equation (Apointer, Bpointer, Cpointer, a, b, c, d): prepares derived solver state.
    void compute_plane_equation(double* Apointer,double* Bpointer, double* Cpointer,double* a, double* b, double* c, double* d);
// ray_triangle_intersect (location, velocity, Apointer, Bpointer, Cpointer, t, u, v): works with mesh topology or geometric intersections.
    bool ray_triangle_intersect(double* location, double* velocity, double* Apointer,double* Bpointer, double* Cpointer, double& t, double& u, double& v);
// CrossboundarySwitch (part, cellid, ifout, dtleft, crossfid, t, triangle_tag): applies boundary-condition behavior.
    void CrossboundarySwitch(particle* part, int& cellid, int& ifout, double& dtleft,int crossfid,double t,int triangle_tag);
// normalize (vec): performs one solver support operation.
    void normalize(double vec[3]);
// buildRotationMatrix (norm, R)): works with mesh topology or geometric intersections.
    void buildRotationMatrix(double* norm, double (*R)[3]);
// preprocesseffquad (istep): performs one solver support operation.
    void preprocesseffquad(int istep);
// dt_ray_crosscell: works with mesh topology or geometric intersections.
    void dt_ray_crosscell();
// scalar_product (norm1, norm2, scalar, dim): performs one solver support operation.
    void scalar_product(double* norm1, double* norm2, double scalar, int dim);
// cache2chain: performs one solver support operation.
    void cache2chain();
// migrateParticlesOnce: updates particles or particle-derived state.
    bool migrateParticlesOnce();
// cache2chain_dt (localNeedFull, globalNeedFull): performs one solver support operation.
    bool cache2chain_dt(int localNeedFull, int globalNeedFull);
// current_macro_zero: couples DSMC data with macroscopic fields.
    void current_macro_zero();
// stepinter_macro_zero (icell): couples DSMC data with macroscopic fields.
    void stepinter_macro_zero(int icell);
// statistic_macroPre: couples DSMC data with macroscopic fields.
    void statistic_macroPre();
// statistic_macro (istep): couples DSMC data with macroscopic fields.
    void statistic_macro(int istep);
// out2dat (istep): writes solver fields or diagnostics.
    bool out2dat(int istep);
// outBoundaryStressHeat (istep): writes solver fields or diagnostics.
    bool outBoundaryStressHeat(int istep);
// collision_diameter (cr): updates particles or particle-derived state.
    double collision_diameter(double cr);
// collision_diameter (cr, collisionDiameterPrefactor): updates particles or particle-derived state.
    double collision_diameter(double cr, double collisionDiameterPrefactor);
// collisionDSMC: updates particles or particle-derived state.
    void collisionDSMC();
// collisionVelocity (bucket, idx1, idx2, tol_collision_times, consant_crm, crm, collisionDiameterPrefactor): updates particles or particle-derived state.
    void collisionVelocity(ParticleBucketSoA& bucket, int idx1, int idx2, int& tol_collision_times, double consant_crm, double& crm, double collisionDiameterPrefactor);
};
