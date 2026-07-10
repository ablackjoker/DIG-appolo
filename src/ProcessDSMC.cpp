# include "ProcessDSMC.h"
#include <chrono>
#include <random>
# include <unordered_map>
# include <mpi.h>
# include <algorithm>
# include <deque>
#include <iomanip>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sstream>
#include <fstream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

#define HSK_EPSILON 1e-10
#define HSK_INF_SERIES 3.5

namespace
{
constexpr double kDsmc2NsMinSampleCount = 3.0;
constexpr double kMacroMinDivisor = 1.0e-12;
}

bool ProcessDSMC::ParticleTransportOps::mapIncomingFaceCellLocal(
    const ProcessDSMC& process,
    int gface, int gcell,
    int& crossfid_local,
    int& cellid_local)
{
    crossfid_local = process.localOfGlobalFace(gface);
    if (crossfid_local < 0) return false;
    cellid_local = process.localOfGlobalCell(gcell);
    return (cellid_local >= 0 && crossfid_local >= 0);
}

int ProcessDSMC::macroOffset(int localCell, MacroIndex index)
{
    return localCell * MACRO_WIDTH + index;
}

int ProcessDSMC::axisOffset(int localCell, AxisIndex index)
{
    return localCell * AXIS_WIDTH + index;
}

int ProcessDSMC::heatOffset(int localCell, HeatIndex index)
{
    return localCell * HEAT_WIDTH + index;
}

int ProcessDSMC::stressOffset(int localCell, StressIndex index)
{
    return localCell * STRESS_WIDTH + index;
}

int ProcessDSMC::rotHeatOffset(int localCell, RotHeatIndex index)
{
    return localCell * ROT_HEAT_WIDTH + index;
}

ProcessDSMC::ParticleTraceOutcome ProcessDSMC::ParticleTransportOps::traceParticleDtleft(
    ProcessDSMC& self, particle& part, int& cellid, int& crossfid,
    int& tag_triangle, double& dtleft, int& ifout, bool inclusive_hit)
{
    double t = 0.0;
    ifout = 0;
    while (dtleft > 0.0)
    {
        crossfid = self.check_intersection2(&part, cellid, t, crossfid, tag_triangle);
        if (crossfid == -2 || !std::isfinite(t) || t <= 0.0)
        {
            ifout = -1;
            dtleft = 0.0;
            return ParticleTraceOutcome::Dropped;
        }
        if (inclusive_hit ? (t >= dtleft) : (t > dtleft))
        {
            self.scalar_product(part.p_location, part.p_velocity, dtleft, DIM);
            const int gcell_final = self.globalOfLocalCell(cellid);
            const int finalRank = self.ownerOfGlobalCell(gcell_final);
            if (gcell_final >= 0 && finalRank >= 0 && finalRank < self.c_size &&
                finalRank != self.c_rank)
            {
                part.p_mesh_serial = gcell_final;
                part.p_rank_serial = finalRank;
                part.dt_left = 0.0;
                if ((int)self.migrate_send_particles.size() < self.c_size)
                    self.migrate_send_particles.resize((std::size_t)self.c_size);
                self.migrate_send_particles[(std::size_t)finalRank].push_back(part);
                ifout = 2;
                dtleft = 0.0;
                return ParticleTraceOutcome::SentRemote;
            }
            dtleft = 0.0;
            return ParticleTraceOutcome::LocalDone;
        }
        self.CrossboundarySwitch(&part, cellid, ifout, dtleft, crossfid, t, tag_triangle);
        if (ifout == -1 || cellid == -1)
        {
            dtleft = 0.0;
            return ParticleTraceOutcome::Dropped;
        }
        if (ifout == 2)
        {
            const int gcell_dst = cellid;
            int dstRank = part.p_rank_serial;
            if (dstRank < 0 || dstRank >= self.c_size)
                dstRank = self.ownerOfGlobalCell(gcell_dst);
            if (dstRank < 0 || dstRank >= self.c_size)
            {
                dtleft = 0.0;
                return ParticleTraceOutcome::Dropped;
            }
            part.p_mesh_serial = gcell_dst;
            part.p_rank_serial = dstRank;
            part.dt_left = dtleft;
            int gface_send = crossfid;
            if (crossfid >= 0 && crossfid < (int)self.face_gids.size())
                gface_send = self.face_gids[crossfid];
            DtleftPacket packet;
            packet.p = part;
            packet.gface = gface_send;
            packet.gcell = gcell_dst;
            packet.tri = tag_triangle;
            if ((int)self.dtleft_send_packets.size() < self.c_size)
                self.dtleft_send_packets.resize((std::size_t)self.c_size);
            self.dtleft_send_packets[(std::size_t)dstRank].push_back(packet);
            dtleft = 0.0;
            return ParticleTraceOutcome::SentRemote;
        }
    }
    return ParticleTraceOutcome::LocalDone;
}

ProcessDSMC :: ProcessDSMC()
{
}

ProcessDSMC :: ~ProcessDSMC()
{
    macro_list_delete();
}

ProcessDSMC :: ProcessDSMC(meshImport *mesh, meshMessage mess, MeshparticalInitial *partinit, const MpiContext& mpiCtx)
{
    this->mpi = &mpiCtx;
    this->partinit = partinit;
    this->mess = mess;
    this->ncell = this->mess.Ncell;
    this->calGroup = mpiCtx.calGroup;
    this->comm = mpiCtx.comm;
    this->mesh = mesh;
    this->rank = mpiCtx.rank;
    this->size = mpiCtx.size;
    this->c_rank = mpiCtx.c_rank;
    this->c_size = mpiCtx.c_size;
    variable_deep_copy();
    if (this->mpi->active())
    {
        macro_list_setup();
        initial_chache_storage();
        rebuildBoundaryDerivedState();
#ifdef CHECK_PARTICLE_BUCKETS
        checkParticleBucketConsistency("initial");
#endif
    }
}

void ProcessDSMC::macro_list_setup()
{
    int eNface = this->mess.eNface;
    this->steady_rho.resize(this->iNcell,0);
    this->steady_T.resize(this->iNcell,0);
    this->steady_U.resize(this->iNcell*AXIS_WIDTH,0);
    this->steady_sigma.resize(this->iNcell*STRESS_WIDTH,0);
    this->steady_q.resize(this->iNcell*HEAT_WIDTH,0);
    this->steady_qr.resize(this->iNcell*ROT_HEAT_WIDTH,0);
    this->step_rho.resize(this->iNcell,0);
    this->step_T.resize(this->iNcell,0);
    this->step_U.resize(this->iNcell*AXIS_WIDTH,0);
    this->step_sigma.resize(this->iNcell*STRESS_WIDTH,0);
    this->step_q.resize(this->iNcell*HEAT_WIDTH,0);
    this->step_qr.resize(this->iNcell*ROT_HEAT_WIDTH,0);
    this->stepinter_rho.resize(this->iNcell,0);
    this->stepinter_T.resize(this->iNcell,0);
    this->stepinter_U.resize(this->iNcell*AXIS_WIDTH,0);
    this->stepinter_sigma.resize(this->iNcell*STRESS_WIDTH,0);
    this->stepinter_q.resize(this->iNcell*HEAT_WIDTH,0);
    this->stepinter_qr.resize(this->iNcell*ROT_HEAT_WIDTH,0);
    this->stepsum_rho.resize(this->iNcell,0);
    this->stepsum_T.resize(this->iNcell,0);
    this->stepsum_U.resize(this->iNcell*AXIS_WIDTH,0);
    this->stepsum_sigma.resize(this->iNcell*STRESS_WIDTH,0);
    this->stepsum_q.resize(this->iNcell*HEAT_WIDTH,0);
    this->stepsum_qr.resize(this->iNcell*ROT_HEAT_WIDTH,0);
    this->crossFlux.assign(eNface, Flux6{});
    this->crossFlux_statistic.assign(eNface, Flux6{});
    this->nsboundaryflux.assign(eNface, Flux6{});
    this->record.resize(MACRO_WIDTH*this->iNcell);
    this->final_record.resize(MACRO_WIDTH*this->iNcell);
    this->local.resize(MACRO_WIDTH*this->iNcell);
    this->dsmc2ns_window_samples.assign((std::size_t)this->iNcell, 0.0);
    this->dsmc2ns_window_valid.assign((std::size_t)this->iNcell, 0);
    this->dsmc2ns_sparse_state.assign((std::size_t)this->iNcell, DSMC2NS_SPARSE_NORMAL);
    this->dsmc2ns_sparse_accum_steps.assign((std::size_t)this->iNcell, 0);
}

void ProcessDSMC::variable_deep_copy()
{
    this->rank_cell_all.resize(this->ncell,0);
    for(int i = 0; i < this->ncell; i ++)
    {
        this->rank_cell_all[i] = this->partinit->rank_cell_all[i];
    }
    this->partitionState.assign(this->rank_cell_all, this->c_size, this->partinit->partitionState.epoch);
    vector<int>().swap(this->partinit->rank_cell_all);
    if(!this->mpi->active()) return;
    this->cells = move(this->partinit->cells);
    this->edges = move(this->partinit->edges);
    this->localPointXY = move(this->partinit->localPointXY);
    this->faceSplitTag = move(this->partinit->faceSplitTag);
    this->boundaryEmitCacheDirty = true;
    this->local_cells.resize(this->partinit->my_ncell);
    this->face_gids.resize(this->partinit->my_nface);
    this->gid2local.clear();
    this->gid2local.reserve(this->partinit->gid2local.size());
    this->face_gid2local.reserve(face_gids.size());
    for(int i = 0; i < this->partinit->my_ncell; i ++)
    {
        this->local_cells[i] = this->partinit->local_cells[i];
    }
    for(int i = 0; i < this->partinit->my_nface; i ++)
    {
        this->face_gids[i] = this->partinit->face_gids[i];
    }
    this->face_gid2local.insert(this->partinit->face_gid2local.begin(), this->partinit->face_gid2local.end());
    this->gid2local.insert(this->partinit->gid2local.begin(), this->partinit->gid2local.end());
    this->iNcell = this->partinit->my_owned_ncell;
    this->nface = this->partinit->my_nface;
    rebuildFaceLookup();
    rebuildFaceCrossCache();
    rebuildFaceTraversalCache();
    rebuild_migration_peer_cache();
    ensure_partition_time_storage();
    vector<int>().swap(this->partinit->local_cells);
    vector<int>().swap(this->partinit->face_gids);
    unordered_map<int, int>().swap(this->partinit->gid2local);
    vector<double>().swap(this->partinit->localPointXY);
    vector<unsigned char>().swap(this->partinit->faceSplitTag);
    vector<DsmcCell>().swap(this->partinit->cells);
    vector<DsmcEdge>().swap(this->partinit->edges);
    unordered_map<int, int>().swap(this->partinit->face_gid2local);
}

void ProcessDSMC::initial_chache_storage()
{
    this->migrate_send_particles.resize(this->c_size);
    this->migrate_recv_particles.resize(this->c_size);
    this->dtleft_send_packets.resize(this->c_size);
    this->dtleft_recv_packets.resize(this->c_size);
}

bool ProcessDSMC::boundaryClassification()
{
    this->inletVis.clear();
    this->wallVis.clear();
    this->outletVis.clear();
    this->topwall.clear();
    this->inletVisMove.clear();
    this->wallVisMove.clear();
    this->outletVisMove.clear();
    this->topwallMove.clear();
    const int edgeCount = std::min(this->nface, (int)this->edges.size());
    auto boundaryRole = [this](int tag) -> int
    {
        const BoundaryCondition& bc = this->boundaryTable.byTag(tag);
        switch (bc.dsmcRole)
        {
            case BoundaryRole::Inlet:
                return 1;
            case BoundaryRole::Outlet:
                return 2;
            case BoundaryRole::Wall:
                return 3;
            case BoundaryRole::TopWall:
                return 4;
            default:
                break;
        }
        return 0;
    };
    int tag = -1;
    for (int i = 0; i < edgeCount; i++){
        tag = this->edges[i].faceTag;
        const int role = boundaryRole(tag);
        switch (role){
            case 1: this->inletVisMove.push_back(i); break;
            case 2: this->outletVisMove.push_back(i); break;
            case 3: this->wallVisMove.push_back(i); break;
            case 4: this->topwallMove.push_back(i); break;
            default: break;
        }
        if (role != 0) {
            const int g1 = this->edges[i].faceMap[4];
            const int g2 = this->edges[i].faceMap[5];
            const int gin = (g1 >= 0 && g1 < this->mess.Ncell) ? g1 :
                            ((g2 >= 0 && g2 < this->mess.Ncell) ? g2 : -1);
            if (gin < 0 || this->ownerOfGlobalCell(gin) != this->c_rank) continue;
        }
        switch (role){
            case 1:
                this->inletVis.push_back(i);
                break;
            case 2:
                this->outletVis.push_back(i);
                break;
            case 3:
                this->wallVis.push_back(i);
                break;
            case 4:
                this->topwall.push_back(i); // owned wall
                break;
            default:
                break;
        }
    }
    return true;
}

void ProcessDSMC::wall_normal()
{
    wallMap.clear();
    if (faceSplitTag.size() != edges.size())
    {
        cout << "error: faceSplitTag size mismatch, faceSplitTag="
             << faceSplitTag.size()
             << " edges=" << edges.size() << endl;
    }
    processFacesQuadNormals(wallVisMove);
    processFacesQuadNormals(inletVisMove);
    processFacesQuadNormals(outletVisMove);
    processFacesQuadNormals(topwallMove);
    boundaryEmitCacheDirty = true;
    rebuildBoundaryEmitCache();
}

void ProcessDSMC::rebuildBoundaryDerivedState()
{
    if (this->mpi != nullptr && !this->mpi->active()) return;
    boundaryClassification();
    wall_normal();
}

const DsmcReservoirBoundaryConfig& ProcessDSMC::reservoirConfigForState(DsmcReservoirBoundaryState state) const
{
    switch (state)
    {
        case DSMC_RESERVOIR_OUTLET:
            return DsmcOutletReservoir;
        case DSMC_RESERVOIR_INLET:
        default:
            return DsmcInletReservoir;
    }
}

bool ProcessDSMC::shouldInjectBoundary(DsmcReservoirBoundaryState state) const
{
    const DsmcReservoirBoundaryConfig& config = reservoirConfigForState(state);
    return config.enabled && config.injectParticles;
}

void ProcessDSMC::rebuildBoundaryEmitCache()
{
    if (!boundaryEmitCacheDirty) return;
    boundaryEmitCache.clear();
    auto loadPointSafe = [&](int idx, std::array<double, AXIS_WIDTH>& P) -> bool
    {
        if (idx < 0) return false;
        const std::size_t p = static_cast<std::size_t>(axisOffset(idx, AXIS_X));
        if (p + AXIS_Z >= this->localPointXY.size()) return false;
        P[AXIS_X] = this->localPointXY[p + AXIS_X];
        P[AXIS_Y] = this->localPointXY[p + AXIS_Y];
        P[AXIS_Z] = this->localPointXY[p + AXIS_Z];
        return true;
    };
    auto triangleArea = [](const std::array<double, 3>& A,
                           const std::array<double, 3>& B,
                           const std::array<double, 3>& C) -> double
    {
        const double ax = B[0] - A[0];
        const double ay = B[1] - A[1];
        const double az = B[2] - A[2];
        const double bx = C[0] - A[0];
        const double by = C[1] - A[1];
        const double bz = C[2] - A[2];
        const double cx = ay * bz - az * by;
        const double cy = az * bx - ax * bz;
        const double cz = ax * by - ay * bx;
        return 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
    };
    auto addPatch = [&](DsmcReservoirBoundaryState reservoirState,
                        int boundaryTag, const BoundaryCondition& boundary,
                        int edgeid, int cellLocal, int triTag, int faceType,
                        unsigned char splitTag, const std::array<double, 3> facePts[4],
                        const int triLocal[3], const double normIn[3])
    {
        BoundaryEmitPatch patch;
        patch.reservoirState = reservoirState;
        patch.boundaryTag = boundaryTag;
        patch.boundary = boundary;
        patch.edgeid = edgeid;
        patch.cellLocal = cellLocal;
        patch.triTag = triTag;
        patch.faceType = faceType;
        patch.splitTag = splitTag;
        const std::array<double, 3>& P0 = facePts[triLocal[0]];
        const std::array<double, 3>& P1 = facePts[triLocal[1]];
        const std::array<double, 3>& P2 = facePts[triLocal[2]];
        patch.p0 = P0;
        for (int d = 0; d < 3; ++d)
        {
            patch.e10[(std::size_t)d] = P1[(std::size_t)d] - P0[(std::size_t)d];
            patch.e20[(std::size_t)d] = P2[(std::size_t)d] - P0[(std::size_t)d];
            patch.normal[(std::size_t)d] = normIn[d];
        }
        patch.area = triangleArea(P0, P1, P2);
        patch.normalMag = std::sqrt(patch.normal[0] * patch.normal[0] +
                                    patch.normal[1] * patch.normal[1] +
                                    patch.normal[2] * patch.normal[2]);
        if (!(patch.area > 0.0) || !std::isfinite(patch.area) ||
            !(patch.normalMag > 0.0) || !std::isfinite(patch.normalMag))
        {
            cout << "BOUNDARY_EMIT_CACHE_BAD_PATCH"
                 << " rank=" << this->c_rank
                 << " edgeId=" << edgeid
                 << " faceType=" << faceType
                 << " splitTag=" << (int)splitTag
                 << " tagTri=" << triTag
                 << " area=" << setprecision(16) << patch.area
                 << " norm=(" << patch.normal[0] << "," << patch.normal[1] << "," << patch.normal[2] << ")"
                 << endl;
            return;
        }
        double negativeNormal[3] = {-patch.normal[0], -patch.normal[1], -patch.normal[2]};
        buildRotationMatrix(negativeNormal, patch.rotation);
        boundaryEmitCache.push_back(patch);
    };
    auto addBoundaryPatches = [&](const vector<int>& reservoirFaces,
                                  DsmcReservoirBoundaryState reservoirState)
    {
        for (int edgeid : reservoirFaces)
        {
            if (edgeid < 0 || edgeid >= (int)this->edges.size()) continue;
            const DsmcEdge& edge = this->edges[(std::size_t)edgeid];
            const BoundaryCondition& bc = this->boundaryTable.byTag(edge.faceTag);
            if (!bc.injectParticles) continue;
            const int gcell1 = edge.faceMap[4];
            const int gcell2 = edge.faceMap[5];
            int cellLocal = -1;
            if (gcell1 >= 0 && gcell1 < ncell)
                cellLocal = this->localOfGlobalCell(gcell1);
            if (cellLocal < 0 && gcell2 >= 0 && gcell2 < ncell)
                cellLocal = this->localOfGlobalCell(gcell2);
            if (cellLocal < 0) continue;
            const int faceType = edge.faceType;
            if (faceType != 3 && faceType != 4) continue;
            const int nodeCount = (faceType == 4) ? 4 : 3;
            std::array<double, 3> facePts[4];
            bool validNodes = true;
            for (int n = 0; n < nodeCount; ++n)
            {
                if (!loadPointSafe(edge.faceMap[n], facePts[n]))
                {
                    validNodes = false;
                    break;
                }
            }
            if (!validNodes) continue;
            int triByTag[2][3] = {{0, 1, 2}, {-1, -1, -1}};
            unsigned char splitTag = meshImport::FACE_SPLIT_INVALID;
            int nTris = 1;
            if (faceType == 4)
            {
                splitTag = (edgeid >= 0 && edgeid < (int)this->faceSplitTag.size())
                    ? this->faceSplitTag[(std::size_t)edgeid]
                    : meshImport::FACE_SPLIT_02;
                int tri0[3], tri1[3];
                meshImport::decode_quad_split_tag(splitTag, tri0, tri1);
                if (tri0[0] < 0 || tri1[0] < 0)
                {
                    splitTag = meshImport::FACE_SPLIT_02;
                    meshImport::decode_quad_split_tag(splitTag, tri0, tri1);
                }
                if (tri0[0] < 0 || tri1[0] < 0) continue;
                for (int n = 0; n < 3; ++n)
                {
                    triByTag[0][n] = tri0[n];
                    triByTag[1][n] = tri1[n];
                }
                nTris = 2;
            }
            for (int tagTri = 1; tagTri <= nTris; ++tagTri)
            {
                const int triSlot = (faceType == 4) ? (tagTri - 1) : 0;
                const int triLocal[3] = {
                    triByTag[triSlot][0],
                    triByTag[triSlot][1],
                    triByTag[triSlot][2]
                };
                if (triLocal[0] < 0 || triLocal[0] >= nodeCount ||
                    triLocal[1] < 0 || triLocal[1] >= nodeCount ||
                    triLocal[2] < 0 || triLocal[2] >= nodeCount)
                    continue;
                double norm[3] = {0.0, 0.0, 0.0};
                if (faceType == 4)
                {
                    auto it = this->wallMap.find(edgeid);
                    if (it == this->wallMap.end()) continue;
                    const auto& wm = it->second;
                    const int normalBase = (tagTri == 1) ? WALL_N1X : WALL_N2X;
                    norm[AXIS_X] = wm[(std::size_t)normalBase + AXIS_X];
                    norm[AXIS_Y] = wm[(std::size_t)normalBase + AXIS_Y];
                    norm[AXIS_Z] = wm[(std::size_t)normalBase + AXIS_Z];
                }
                else
                {
                    norm[0] = edge.edgeNormal[0];
                    norm[1] = edge.edgeNormal[1];
                    norm[2] = edge.edgeNormal[2];
                    const double normLen = std::sqrt(norm[0] * norm[0] + norm[1] * norm[1] + norm[2] * norm[2]);
                    if (normLen > 0.0 && std::isfinite(normLen))
                    {
                        norm[0] /= normLen;
                        norm[1] /= normLen;
                        norm[2] /= normLen;
                    }
                }
                addPatch(reservoirState, edge.faceTag, bc, edgeid, cellLocal, (faceType == 3) ? -1 : tagTri,
                         faceType, splitTag, facePts, triLocal, norm);
            }
        }
    };
    addBoundaryPatches(this->inletVis, DSMC_RESERVOIR_INLET);
    addBoundaryPatches(this->outletVis, DSMC_RESERVOIR_OUTLET);
    boundaryEmitCacheDirty = false;
}

int ProcessDSMC::ownerOfGlobalCell(int gid) const
{
    if (gid < 0 || gid >= this->ncell) return -1;
    const int stateOwner = this->partitionState.ownerOf(gid);
    if (stateOwner >= 0 && stateOwner < this->c_size)
        return stateOwner;
    if (gid < (int)this->rank_cell_all.size())
    {
        const int owner = this->rank_cell_all[(std::size_t)gid];
        if (owner >= 0 && owner < this->c_size)
            return owner;
    }
    return -1;
}

int ProcessDSMC::localOfGlobalCell(int gid) const
{
    if (gid < 0 || gid >= this->ncell) return -1;
    auto it = this->gid2local.find(gid);
    if (it == this->gid2local.end()) return -1;
    return it->second;
}

bool ProcessDSMC::isOwnedLocalCell(int local) const
{
    return local >= 0 && local < this->iNcell;
}

int ProcessDSMC::globalOfLocalCell(int local) const
{
    if (local < 0 || local >= (int)this->local_cells.size()) return -1;
    return this->local_cells[(std::size_t)local];
}

ParticleBucketSoA &ProcessDSMC::currParticles(int globalCell)
{
    const int localCell = this->localOfGlobalCell(globalCell);
    if (this->partinit == nullptr ||
        localCell < 0 ||
        localCell >= this->iNcell ||
        localCell >= (int)this->partinit->cell_particles_curr.size())
    {
        std::cerr << "Invalid currParticles access"
                  << " rank=" << this->c_rank
                  << " globalCell=" << globalCell
                  << " localCell=" << localCell
                  << " iNcell=" << this->iNcell
                  << " bucketSize=" << (this->partinit ? (int)this->partinit->cell_particles_curr.size() : -1)
                  << std::endl;
        std::abort();
    }
    return this->partinit->cell_particles_curr[(std::size_t)localCell];
}
const ParticleBucketSoA &ProcessDSMC::currParticles(int globalCell) const
{
    const int localCell = this->localOfGlobalCell(globalCell);
    if (this->partinit == nullptr ||
        localCell < 0 ||
        localCell >= this->iNcell ||
        localCell >= (int)this->partinit->cell_particles_curr.size())
    {
        std::cerr << "Invalid const currParticles access"
                  << " rank=" << this->c_rank
                  << " globalCell=" << globalCell
                  << " localCell=" << localCell
                  << " iNcell=" << this->iNcell
                  << " bucketSize=" << (this->partinit ? (int)this->partinit->cell_particles_curr.size() : -1)
                  << std::endl;
        std::abort();
    }
    return this->partinit->cell_particles_curr[(std::size_t)localCell];
}

int ProcessDSMC::particleCount(int globalCell) const
{
    if (this->partinit == nullptr)
        return 0;
    const int localCell = this->localOfGlobalCell(globalCell);
    if (localCell < 0 ||
        localCell >= this->iNcell ||
        localCell >= (int)this->partinit->cell_particles_curr.size())
        return 0;
    return (int)this->partinit->cell_particles_curr[(std::size_t)localCell].size();
}

bool ProcessDSMC::checkParticleBucketConsistency(const char* stage) const
{
    if (this->partinit == nullptr) return true;
    if ((int)this->partinit->cell_particles_curr.size() != this->iNcell)
    {
        cout << "PARTICLE_BUCKET_SIZE_MISMATCH"
             << " stage=" << stage
             << " rank=" << this->c_rank
             << " bucketSize=" << this->partinit->cell_particles_curr.size()
             << " iNcell=" << this->iNcell
             << endl;
        return false;
    }
    for (int lc = 0; lc < this->iNcell; ++lc)
    {
        const int gid = this->globalOfLocalCell(lc);
        if (gid < 0)
        {
            cout << "PARTICLE_BUCKET_BAD_GLOBAL"
                 << " stage=" << stage
                 << " rank=" << this->c_rank
                 << " lc=" << lc
                 << endl;
            return false;
        }
        const ParticleBucketSoA& bucket =
            this->partinit->cell_particles_curr[(std::size_t)lc];
        for (std::size_t p = 0; p < bucket.size(); ++p)
        {
            if (bucket.p_mesh_serial[p] != lc)
            {
                cout << "PARTICLE_BUCKET_BAD_LOCAL_ID"
                     << " stage=" << stage
                     << " rank=" << this->c_rank
                     << " lc=" << lc
                     << " gid=" << gid
                     << " p=" << p
                     << " stored=" << bucket.p_mesh_serial[p]
                     << endl;
                return false;
            }
        }
    }
    return true;
}

void ProcessDSMC::clearNextParticleBuffers()
{
    if (this->partinit == nullptr) return;
    this->partinit->cell_particles_next.clear();
}

void ProcessDSMC::clearNextParticles(int globalCell)
{
    if (this->partinit == nullptr) return;
    auto it = this->partinit->cell_particles_next.find(globalCell);
    if (it != this->partinit->cell_particles_next.end())
    {
        it->second.clear();
        this->partinit->cell_particles_next.erase(it);
    }
}

void ProcessDSMC::macro_list_delete()
{
}

void ProcessDSMC::processFacesQuadNormals(const vector<int>& visList)
{
    for (int edgeid : visList)
    {
        if (edgeid < 0 || edgeid >= (int)edges.size()) continue;
        if (edges[edgeid].faceType != 4) continue;
        const int i0 = edges[edgeid].faceMap[0];
        const int i1 = edges[edgeid].faceMap[1];
        const int i2 = edges[edgeid].faceMap[2];
        const int i3 = edges[edgeid].faceMap[3];
        auto P = [&](int idx) -> std::array<double,AXIS_WIDTH>
        {
            const int pointBase = axisOffset(idx, AXIS_X);
            return {
                this->localPointXY[pointBase + AXIS_X],
                this->localPointXY[pointBase + AXIS_Y],
                this->localPointXY[pointBase + AXIS_Z]
            };
        };
        const auto A = P(i0);
        const auto B = P(i1);
        const auto C = P(i2);
        const auto D = P(i3);
        unsigned char tag = (edgeid >= 0 && edgeid < (int)this->faceSplitTag.size())
            ? this->faceSplitTag[(std::size_t)edgeid]
            : meshImport::FACE_SPLIT_02;
        int tri0[3], tri1[3];
        meshImport::decode_quad_split_tag(tag, tri0, tri1);
        if (tri0[0] < 0 || tri1[0] < 0)
        {
            tag = meshImport::FACE_SPLIT_02;
            meshImport::decode_quad_split_tag(tag, tri0, tri1);
        }
        if (tri0[0] < 0 || tri1[0] < 0)
        {
            cout << "FACE_SPLIT_INVALID wallMap"
                 << " faceId=" << edgeid
                 << " tag=" << (int)tag
                 << " faceType=" << edges[edgeid].faceType
                 << " faceTag=" << edges[edgeid].faceTag
                 << " nodes=(" << i0 << "," << i1 << "," << i2 << "," << i3 << ")"
                 << endl;
            continue;
        }
        const std::array<double,3> P4[4] = {A, B, C, D};
        auto makeNormal = [&](const int tri[3], double n[3])
        {
            const auto& X = P4[tri[0]];
            const auto& Y = P4[tri[1]];
            const auto& Z = P4[tri[2]];
            const double e1[3] = {Y[0]-X[0], Y[1]-X[1], Y[2]-X[2]};
            const double e2[3] = {Z[0]-X[0], Z[1]-X[1], Z[2]-X[2]};
            this->partinit->crossProduct(const_cast<double*>(e1), const_cast<double*>(e2), n);
            normalize(n);
            double* nref = edges[edgeid].edgeNormal;
            if (partinit->dotProduct(nref, n, DIM) < 0.0)
            {
                n[0] *= -1.0;
                n[1] *= -1.0;
                n[2] *= -1.0;
            }
        };
        double n1[3], n2[3];
        makeNormal(tri0, n1);
        makeNormal(tri1, n2);
        WallNormalPair normals{};
        normals[WALL_N1X] = n1[AXIS_X];
        normals[WALL_N1Y] = n1[AXIS_Y];
        normals[WALL_N1Z] = n1[AXIS_Z];
        normals[WALL_N2X] = n2[AXIS_X];
        normals[WALL_N2Y] = n2[AXIS_Y];
        normals[WALL_N2Z] = n2[AXIS_Z];
        wallMap[edgeid] = normals;
    }
}

bool ProcessDSMC :: ray_triangle_intersect(double* location, double* velocity, double* Apointer,double* Bpointer, double* Cpointer, double& t, double& u, double& v)
{
    double ABedge[3] = {Bpointer[0] - Apointer[0], Bpointer[1] - Apointer[1], Bpointer[2] - Apointer[2]};
    double ACedge[3] = {Cpointer[0] - Apointer[0], Cpointer[1] - Apointer[1], Cpointer[2] - Apointer[2]};
    double VCcross[3];
    const double eps = 1e-14;
    this->partinit->crossProduct(velocity,ACedge,VCcross);
    double det = this->partinit->dotProduct(VCcross,ABedge,3);
    if(fabs(det) < eps) return false;
    double invDet = 1.0 / det;
    double APedge[3] = {location[0] - Apointer[0], location[1] - Apointer[1], location[2] - Apointer[2]};
    u = this->partinit->dotProduct(APedge,VCcross,DIM)*invDet;
    if(u < 0.0 || u > 1.0) return false;
    double PBcross[3];
    this->partinit->crossProduct(APedge,ABedge,PBcross);
    v = this->partinit->dotProduct(velocity, PBcross,DIM) * invDet ;
    if(v < 0.0 || (u + v) > 1.0) return false;
    t = this->partinit->dotProduct(ACedge, PBcross,DIM) * invDet ;
    if(t > eps) {return true;}
    else{return false;}
}

bool ProcessDSMC::intersectCachedTri(const FaceTriCache& tri, const particle* part, double& t) const
{
    const double* location = part->p_location;
    const double* velocity = part->p_velocity;
    const double pvec0 = velocity[1] * tri.e2z - velocity[2] * tri.e2y;
    const double pvec1 = velocity[2] * tri.e2x - velocity[0] * tri.e2z;
    const double pvec2 = velocity[0] * tri.e2y - velocity[1] * tri.e2x;
    const double det = tri.e1x * pvec0 + tri.e1y * pvec1 + tri.e1z * pvec2;
    const double eps = 1e-14;
    if (std::fabs(det) < eps) return false;
    const double invDet = 1.0 / det;
    const double tvec0 = location[0] - tri.ax;
    const double tvec1 = location[1] - tri.ay;
    const double tvec2 = location[2] - tri.az;
    const double u = (tvec0 * pvec0 + tvec1 * pvec1 + tvec2 * pvec2) * invDet;
    if (u < 0.0 || u > 1.0) return false;
    const double qvec0 = tvec1 * tri.e1z - tvec2 * tri.e1y;
    const double qvec1 = tvec2 * tri.e1x - tvec0 * tri.e1z;
    const double qvec2 = tvec0 * tri.e1y - tvec1 * tri.e1x;
    const double v = (velocity[0] * qvec0 + velocity[1] * qvec1 + velocity[2] * qvec2) * invDet;
    if (v < 0.0 || (u + v) > 1.0) return false;
    t = (tri.e2x * qvec0 + tri.e2y * qvec1 + tri.e2z * qvec2) * invDet;
    return t > eps;
}

bool ProcessDSMC::tryCachedCellIntersection(
    particle* part, int Ncell_id, double& min_t, int acrossfid,
    int& tag_triangle, int& hit_face) const
{
    if (Ncell_id < 0 || (std::size_t)(Ncell_id + 1) >= this->cellTriOffset.size())
        return false;
    const int begin = this->cellTriOffset[(std::size_t)Ncell_id];
    const int end = this->cellTriOffset[(std::size_t)Ncell_id + 1u];
    if (begin < 0 || end < begin || end > (int)this->cellTriCache.size())
        return false;
    if (begin == end)
        return false;
    const int prev_tri = tag_triangle;
    double bestT = DBL_MAX;
    int bestFace = -2;
    int bestTri = -1;
    for (int i = begin; i < end; ++i)
    {
        const FaceTriCache& tri = this->cellTriCache[(std::size_t)i];
        if (tri.face == acrossfid)
        {
            if (tri.triTag <= 0 || (prev_tri != 1 && prev_tri != 2) || tri.triTag == prev_tri)
                continue;
        }
        double t = DBL_MAX;
        if (intersectCachedTri(tri, part, t) && t < bestT)
        {
            bestT = t;
            bestFace = tri.face;
            bestTri = tri.triTag;
        }
    }
    if (bestFace == -2)
    {
        min_t = DBL_MAX;
        return false;
    }
    hit_face = bestFace;
    tag_triangle = bestTri;
    min_t = bestT;
    return true;
}

void ProcessDSMC::tryTri(int faceid, int triTag, double *A,  double *B,  double *C, particle* part, double& bestT, int& bestFace, int& bestTri)
{
    double t, u, v;
    if (ray_triangle_intersect(part->p_location, part->p_velocity, A, B, C, t, u, v))
    {
        if (t < bestT)
        {
        bestT = t;
        bestFace = faceid;
        bestTri = triTag;
        }
    }
}

void ProcessDSMC::loadPoint(int idx, double *P)
{
    const int pointBase = axisOffset(idx, AXIS_X);
    P[AXIS_X] = this->localPointXY[pointBase + AXIS_X];
    P[AXIS_Y] = this->localPointXY[pointBase + AXIS_Y];
    P[AXIS_Z] = this->localPointXY[pointBase + AXIS_Z];
}
int ProcessDSMC :: tetra_intersection(particle* part, int Ncell_id, double& min_t, int acrossfid)
{
    min_t = DBL_MAX;
    int hit_face = -2;
    const int faceCount = this->cells[Ncell_id].num;
    for (int m = 0; m < faceCount && m < NN; m++)
    {
        const int faceid = this->cells[Ncell_id].cell2face[m];
        if (faceid < 0 || faceid >= (int)this->edges.size()) continue;
        if (acrossfid == faceid) continue;
        if (this->edges[faceid].faceType != 3) continue;
        const int index1 = this->edges[faceid].faceMap[0];
        const int index2 = this->edges[faceid].faceMap[1];
        const int index3 = this->edges[faceid].faceMap[2];
        double Apointer[3], Bpointer[3],Cpointer[3];
        this->loadPoint(index1, Apointer); this->loadPoint(index2, Bpointer); this->loadPoint(index3, Cpointer);
        double t, u, v;
        if(ray_triangle_intersect(part->p_location,part->p_velocity,Apointer,Bpointer,Cpointer,t, u, v))
        {
            if (t < min_t)
            {
                min_t  = t;
                hit_face = faceid;
            }
        }
    }
    return hit_face;
}

bool ProcessDSMC::loadFacePointsSafe(int faceid, FacePointSet& facePoints) const
{
    if (faceid < 0 || faceid >= (int)this->edges.size()) return false;
    const int facetype = this->edges[faceid].faceType;
    facePoints.faceType = facetype;
    for (int n = 0; n < 4; ++n)
        facePoints.nodeIds[n] = this->edges[faceid].faceMap[n];
    const int requiredNodes = (facetype == 4) ? 4 : 3;
    for (int n = 0; n < requiredNodes; ++n)
    {
        const int idx = facePoints.nodeIds[n];
        if (idx < 0) return false;
        const size_t p = static_cast<size_t>(axisOffset(idx, AXIS_X));
        if (p > this->localPointXY.size() || p + AXIS_Z >= this->localPointXY.size()) return false;
        facePoints.points[n][AXIS_X] = this->localPointXY[p + AXIS_X];
        facePoints.points[n][AXIS_Y] = this->localPointXY[p + AXIS_Y];
        facePoints.points[n][AXIS_Z] = this->localPointXY[p + AXIS_Z];
    }
    return true;
}

bool ProcessDSMC::decodeFaceTriangles(int faceid, const FacePointSet& facePoints,
                                      FaceTriangleList& triangles) const
{
    if (facePoints.faceType == 3)
    {
        triangles.count = 1;
        triangles.nodes[0][0] = 0;
        triangles.nodes[0][1] = 1;
        triangles.nodes[0][2] = 2;
        triangles.tag[0] = -1;
        return true;
    }
    if (facePoints.faceType != 4)
    {
        cout << "INTERSECT_FACE_TYPE_ERROR"
             << " faceId=" << faceid
             << " faceType=" << facePoints.faceType
             << endl;
        return false;
    }
    triangles.splitTag = (faceid >= 0 && faceid < (int)this->faceSplitTag.size())
        ? this->faceSplitTag[(std::size_t)faceid]
        : meshImport::FACE_SPLIT_02;
    int tri0[3], tri1[3];
    meshImport::decode_quad_split_tag(triangles.splitTag, tri0, tri1);
    if (tri0[0] < 0 || tri1[0] < 0)
    {
        triangles.splitTag = meshImport::FACE_SPLIT_02;
        meshImport::decode_quad_split_tag(triangles.splitTag, tri0, tri1);
    }
    if (tri0[0] < 0 || tri1[0] < 0)
    {
            cout << "FACE_SPLIT_INVALID check_intersection2"
             << " faceId=" << faceid
             << " tag=" << (int)triangles.splitTag
             << " faceType=" << facePoints.faceType
             << " faceTag=" << this->edges[faceid].faceTag
             << " nodes=(" << facePoints.nodeIds[0] << "," << facePoints.nodeIds[1] << "," << facePoints.nodeIds[2] << "," << facePoints.nodeIds[3] << ")"
             << endl;
        return false;
    }
    triangles.count = 2;
    triangles.tag[0] = 1;
    triangles.tag[1] = 2;
    for (int n = 0; n < 3; ++n)
    {
        triangles.nodes[0][n] = tri0[n];
        triangles.nodes[1][n] = tri1[n];
    }
    return true;
}

void ProcessDSMC::intersectFaceTriangles(int faceid, int skipTriTag, particle* part,
                                         IntersectionHit& bestHit)
{
    FacePointSet facePoints;
    if (!loadFacePointsSafe(faceid, facePoints))
    {
        if (faceid >= 0 && faceid < (int)this->edges.size())
        {
            cout << "INTERSECT_NODE_ERROR"
                 << " faceId=" << faceid
                 << " faceType=" << this->edges[faceid].faceType
                 << " nodes=(" << this->edges[faceid].faceMap[0] << "," << this->edges[faceid].faceMap[1] << "," << this->edges[faceid].faceMap[2] << "," << this->edges[faceid].faceMap[3] << ")"
                 << endl;
        }
        return;
    }
    FaceTriangleList triangles;
    if (!decodeFaceTriangles(faceid, facePoints, triangles)) return;
    for (int t = 0; t < triangles.count; ++t)
    {
        const int triTag = triangles.tag[t];
        if (triTag == skipTriTag) continue;
        const int* tri = triangles.nodes[t];
        this->tryTri(faceid, triTag,
                     facePoints.points[tri[0]],
                     facePoints.points[tri[1]],
                     facePoints.points[tri[2]],
                     part, bestHit.t, bestHit.face, bestHit.tri);
    }
}

int ProcessDSMC::intersectFallbackCell(particle* part, int cellLocal, int acrossFace,
                                       int previousTri, double& min_t, int& tag_triangle)
{
    int facenumber = this->cells[cellLocal].num;
    int hit_face = -2;
    if(facenumber == 4)
    {
        hit_face = tetra_intersection(part, cellLocal, min_t, acrossFace);
        tag_triangle = -1;
        return hit_face;
    }
    min_t = DBL_MAX;
    IntersectionHit bestHit;
    bestHit.face = -2;
    bestHit.tri = -1;
    bestHit.t = DBL_MAX;
    for (int m = 0; m < facenumber && m < NN; ++m)
    {
        int faceid = this->cells[cellLocal].cell2face[m];
        if (faceid < 0 || faceid >= (int)this->edges.size()) continue;
        if(acrossFace == faceid){continue;}
        intersectFaceTriangles(faceid, 0, part, bestHit);
    }
    if (acrossFace >= 0)
    {
        const int facetype = this->edges[acrossFace].faceType;
        if (facetype == 4 && (previousTri == 1 || previousTri == 2))
        {
            intersectFaceTriangles(acrossFace, previousTri, part, bestHit);
        }
    }
    if (bestHit.face == -2)
    {
        tag_triangle = -1;
        min_t = DBL_MAX;
        return -2;
    }
    tag_triangle = bestHit.tri;
    min_t = bestHit.t;
    return bestHit.face;
}

int ProcessDSMC :: check_intersection2(particle* part, int Ncell_id, double& min_t, int acrossfid,int& tag_triangle)
{
    if (Ncell_id < 0 || Ncell_id >= (int)this->cells.size())
    {
        min_t = DBL_MAX;
        cout<<" check is error "<< Ncell_id << " "<< this->iNcell
            <<" "<< this->cells.size() <<endl;
        return -2;
    }
    int cached_hit_face = -2;
    if (tryCachedCellIntersection(part, Ncell_id, min_t, acrossfid, tag_triangle, cached_hit_face))
        return cached_hit_face;
    int prev_tri = tag_triangle;
    return intersectFallbackCell(part, Ncell_id, acrossfid, prev_tri, min_t, tag_triangle);
}

void ProcessDSMC :: normalize(double vec[3])
{
    double length = sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
    vec[0] /= length;
    vec[1] /= length;
    vec[2] /= length;
}

void ProcessDSMC :: buildRotationMatrix(double* norm, double (*R)[3])
{
    double t1[3], t2[3];
    if((fabs(norm[0]) <= fabs(norm[1]))&& (fabs(norm[0]) <= fabs(norm[2])))
    {
        t1[0] = 0.0; t1[1] = norm[2]; t1[2] = -norm[1];
    }
    else if (fabs(norm[1]) < fabs(norm[2]))
    {
        t1[0] = norm[2]; t1[1] = 0.0; t1[2] = -norm[0];
    }
    else{
        t1[0] = norm[1]; t1[1] = -norm[0]; t1[2] = 0.0;
    }
    normalize(t1);
    partinit->crossProduct(norm,t1,t2);
    normalize(t2);
    R[0][0] = t1[0]; R[0][1] = t2[0]; R[0][2] = norm[0];
    R[1][0] = t1[1]; R[1][1] = t2[1]; R[1][2] = norm[1];
    R[2][0] = t1[2]; R[2][1] = t2[2]; R[2][2] = norm[2];
}

void ProcessDSMC :: scalar_product(double* norm1, double* norm2, double scalar, int dim)
{
    for(int i = 0; i < dim; i++)
    {
        norm1[i] += norm2[i] * scalar;
    }
}

bool ProcessDSMC::ownsBoundaryFace(int localFace) const
{
    if (this->mpi == nullptr || !this->mpi->active()) return false;
    if (localFace < 0 || localFace >= (int)this->edges.size()) return false;
    const DsmcEdge& edge = this->edges[(std::size_t)localFace];
    if (this->boundaryTable.byTag(edge.faceTag).dsmcRole != BoundaryRole::Wall) return false;
    const int g1 = edge.faceMap[4];
    const int g2 = edge.faceMap[5];
    const int ownedCell = (g1 >= 0 && g1 < this->mess.Ncell) ? g1 :
                          ((g2 >= 0 && g2 < this->mess.Ncell) ? g2 : -1);
    return ownedCell >= 0 && this->ownerOfGlobalCell(ownedCell) == this->c_rank;
}

void ProcessDSMC::recordBoundaryStressHeat(int localFace, const double velocityPre[3],
                                           const double velocityPost[3],
                                           double rotPre, double rotPost)
{
    if (!EnableBoundaryStressHeatStatistic) return;
    if (this->istep <= NSS) return;
    if (!ownsBoundaryFace(localFace)) return;
    int globalFace = localFace;
    if (localFace >= 0 && localFace < (int)this->face_gids.size())
        globalFace = this->face_gids[(std::size_t)localFace];
    if (globalFace < 0 || globalFace >= this->mess.Nface) return;
    double momentumDelta[3] = {
        velocityPost[0] - velocityPre[0],
        velocityPost[1] - velocityPre[1],
        velocityPost[2] - velocityPre[2]
    };
    const double vPre2 = velocityPre[0]*velocityPre[0] +
                         velocityPre[1]*velocityPre[1] +
                         velocityPre[2]*velocityPre[2];
    const double vPost2 = velocityPost[0]*velocityPost[0] +
                          velocityPost[1]*velocityPost[1] +
                          velocityPost[2]*velocityPost[2];
    const double energyTransDelta = 0.5 * (vPost2 - vPre2);
    const double energyRotDelta = rotPost - rotPre;
    BoundarySurfaceTally& tally = this->boundarySteadyTally[globalFace];
    tally.hits += 1.0;
    for (int d = 0; d < AXIS_WIDTH; ++d)
        tally.momentumDelta[(std::size_t)d] += momentumDelta[d];
    tally.energyTransDelta += energyTransDelta;
    tally.energyRotDelta += energyRotDelta;
}

void ProcessDSMC::CrossboundarySwitch(particle* part, int& cellid, int& ifout, double& dtleft,int crossfid,double t,int triangle_tag)
{
    auto& gen = partinit->thread_rng();
    auto& dis = partinit->get_uniform();
    const int tag = this->edges[crossfid].faceTag;
    const BoundaryCondition& boundary = this->boundaryTable.byTag(tag);
    auto getWallNormal = [&](int fid, int triTag, double n[3])
    {
        n[0] = edges[fid].edgeNormal[0];
        n[1] = edges[fid].edgeNormal[1];
        n[2] = edges[fid].edgeNormal[2];
        normalize(n);
        {
            auto it = wallMap.find(fid);
            if (it != wallMap.end())
            {
                const auto& wm = it->second;
                if (triTag == 1)
                {
                    n[AXIS_X] = wm[WALL_N1X];
                    n[AXIS_Y] = wm[WALL_N1Y];
                    n[AXIS_Z] = wm[WALL_N1Z];
                }
                else if (triTag == 2)
                {
                    n[AXIS_X] = wm[WALL_N2X];
                    n[AXIS_Y] = wm[WALL_N2Y];
                    n[AXIS_Z] = wm[WALL_N2Z];
                }
            }
        }
    };
    switch (boundary.dsmcRole)
    {
    case BoundaryRole::Interface:
    {
        ifout = 1;
        scalar_product(part->p_location, part->p_velocity, t, DIM);
        dtleft -= t;
        int gcur = -1;
        int gnext = -1;
        int lnext = -1;
        int dstRank = -1;
        const FaceCrossCache* crossCache = nullptr;
        if (crossfid >= 0 && crossfid < (int)this->faceCrossCache.size())
            crossCache = &this->faceCrossCache[(std::size_t)crossfid];
        if (crossCache != nullptr && cellid == crossCache->localCell0)
        {
            gcur = crossCache->globalCell0;
            gnext = crossCache->globalNextFromCell0;
            lnext = crossCache->localNextFromCell0;
            dstRank = crossCache->ownerNextFromCell0;
        }
        else if (crossCache != nullptr && cellid == crossCache->localCell1)
        {
            gcur = crossCache->globalCell1;
            gnext = crossCache->globalNextFromCell1;
            lnext = crossCache->localNextFromCell1;
            dstRank = crossCache->ownerNextFromCell1;
        }
        else
        {
            gcur = this->globalOfLocalCell(cellid);
            int g1   = edges[crossfid].faceMap[4];
            int g2   = edges[crossfid].faceMap[5];
            if      (gcur == g1) gnext = g2;
            else if (gcur == g2) gnext = g1;
            else{gnext = g1;cout<<"error in interface boundary tag="<<tag<<endl;}
            lnext = localOfGlobalCell(gnext);
            dstRank = ownerOfGlobalCell(gnext);
        }
        record_partition_flux(gcur, gnext, cellid);
        if (lnext >= 0)
        {
            cellid = lnext;
            ifout  = 1;
        }
        else
        {
            part->p_rank_serial = dstRank;
            cellid = gnext;
            ifout  = 2;
        }
    }break;
    case BoundaryRole::Inlet:
    case BoundaryRole::Outlet:
    {
        ifout = -1;
        scalar_product(part->p_location, part->p_velocity, t, DIM);
        cellid = -1; dtleft = -1;
    }break;
    case BoundaryRole::Wall:
    {
        scalar_product(part->p_location, part->p_velocity, t, DIM);
        dtleft -= t;
        double norm[3];
        getWallNormal(crossfid, triangle_tag, norm);
        const double velocityPre[3] = {
            part->p_velocity[AXIS_X],
            part->p_velocity[AXIS_Y],
            part->p_velocity[AXIS_Z]
        };
        const double rotPre = part->p_Ir;
        double R[3][3];
        buildRotationMatrix(norm,R);
        double vxp, vyp, vzp;
        double r1 = dis(gen);
        double r2 = dis(gen);
        double alpha = 2*M_PI*dis(gen);
        vxp = sqrt(-log(r2))*sin(alpha)*sqrt(this->mess.Twall_ref);
        vyp = sqrt(-log(r2))*cos(alpha)*sqrt(this->mess.Twall_ref);
        vzp = -sqrt(-log(r1))*sqrt(this->mess.Twall_ref);
        part->p_velocity[0] = R[0][0]*vxp + R[0][1]*vyp + R[0][2]*vzp;
        part->p_velocity[1] = R[1][0]*vxp + R[1][1]*vyp + R[1][2]*vzp;
        part->p_velocity[2] = R[2][0]*vxp + R[2][1]*vyp + R[2][2]*vzp;
        double r3 = dis(gen);
        part->p_Ir = -log(r3) * this->mess.Twall_ref * 0.5;
        const double velocityPost[3] = {
            part->p_velocity[AXIS_X],
            part->p_velocity[AXIS_Y],
            part->p_velocity[AXIS_Z]
        };
        recordBoundaryStressHeat(crossfid, velocityPre, velocityPost, rotPre, part->p_Ir);
    }break;
    case BoundaryRole::TopWall:
    {
        scalar_product(part->p_location, part->p_velocity, t, DIM);
        dtleft -= t;
        double norm[3];
        getWallNormal(crossfid, triangle_tag, norm);
        const double velocityPre[3] = {
            part->p_velocity[AXIS_X],
            part->p_velocity[AXIS_Y],
            part->p_velocity[AXIS_Z]
        };
        const double rotPre = part->p_Ir;
        // double* norm = mesh->edges[crossfid].edgeNormal;
        double R[3][3];
        buildRotationMatrix(norm,R);
        double vxp, vyp, vzp;
        double r1 = dis(gen);
        double r2 = dis(gen);
        double alpha = 2*M_PI*dis(gen);
        vxp = sqrt(-log(r2))*sin(alpha)*sqrt(this->mess.Twall_ref);
        vyp = sqrt(-log(r2))*cos(alpha)*sqrt(this->mess.Twall_ref);
        vzp = -sqrt(-log(r1))*sqrt(this->mess.Twall_ref);
        part->p_velocity[0] = R[0][0]*vxp + R[0][1]*vyp + R[0][2]*vzp  + this->mess.u_wall;
        part->p_velocity[1] = R[1][0]*vxp + R[1][1]*vyp + R[1][2]*vzp;
        part->p_velocity[2] = R[2][0]*vxp + R[2][1]*vyp + R[2][2]*vzp;
        double r3 = dis(gen);
        part->p_Ir = -log(r3) * this->mess.Twall_ref * 0.5;
        const double velocityPost[3] = {
            part->p_velocity[AXIS_X],
            part->p_velocity[AXIS_Y],
            part->p_velocity[AXIS_Z]
        };
        recordBoundaryStressHeat(crossfid, velocityPre, velocityPost, rotPre, part->p_Ir);
    }break;
    default:
        break;
    }
}

void ProcessDSMC::dt_ray_crosscell()
{
    while (true)
    {
        if ((int)this->dtleft_send_packets.size() < this->c_size)
            this->dtleft_send_packets.resize((std::size_t)this->c_size);
        if ((int)this->migrationPeerMask.size() != this->c_size)
            rebuild_migration_peer_cache();
        const bool all_empty = std::all_of(
            dtleft_send_packets.begin(), dtleft_send_packets.end(),
            [](const std::vector<DtleftPacket>& v){ return v.empty(); }
        );
        int localNeedFull = 0;
        if (!all_empty)
        {
            for (int r = 0; r < this->c_size; ++r)
            {
                if (r == this->c_rank) continue;
                if (this->dtleft_send_packets[(std::size_t)r].empty()) continue;
                if ((int)this->migrationPeerMask.size() != this->c_size ||
                    !this->migrationPeerMask[(std::size_t)r])
                {
                    localNeedFull = 1;
                    break;
                }
            }
        }
        int localState[2] = { all_empty ? 0 : 1, localNeedFull };
        int globalState[2] = {0, 0};
        MPI_Allreduce(localState, globalState, 2, MPI_INT, MPI_LOR, calGroup);
        if (globalState[0] == 0) break;
        cache2chain_dt(localNeedFull, globalState[1]);
        for (int src = 0; src < c_size; ++src)
        {
            const int recvN = (int)dtleft_recv_packets[src].size();
            if (recvN == 0) continue;
            for (int j = 0; j < recvN; ++j)
            {
                const DtleftPacket& packet = dtleft_recv_packets[src][j];
                particle parts = packet.p;
                const int gface = packet.gface;
                const int gcell = packet.gcell;
                int tag_triangle = packet.tri;
                int crossfid = -1;
                int cellid = -1;
                if (!(parts.dt_left > 0.0))
                {
                    cellid = this->localOfGlobalCell(gcell);
                    if (cellid >= 0 && this->isOwnedLocalCell(cellid))
                    {
                        parts.p_mesh_serial = cellid;
                        parts.p_rank_serial = this->c_rank;
                        parts.dt_left = 0.0;
                        this->recv_cache.push_back(parts);
                    }
                    else
                    {
                        cout << "DTLEFT_FINAL_BAD_CELL rank=" << this->c_rank
                             << " gcell=" << gcell
                             << " local=" << cellid
                             << " owner=" << this->ownerOfGlobalCell(gcell)
                             << endl;
                    }
                    continue;
                }
                if (!ParticleTransportOps::mapIncomingFaceCellLocal(*this, gface, gcell, crossfid, cellid))
                {
                    cout << " cell is error " << endl;
                    continue;
                }
                parts.p_mesh_serial = gcell;
                int ifout = 0;
                double dtleft = parts.dt_left;
                ParticleTransportOps::traceParticleDtleft(*this, parts, cellid, crossfid, tag_triangle, dtleft, ifout, true);
                if (ifout == 0 || ifout == 1)
                {
                    if (this->globalOfLocalCell(cellid) >= 0)
                    {
                        parts.p_mesh_serial = cellid;
                        parts.p_rank_serial = this->c_rank;
                        parts.dt_left = 0.0;
                        this->recv_cache.push_back(parts);
                    }
                }
            }
        }
        for (int i = 0; i < c_size; ++i)
        {
            dtleft_recv_packets[i].clear();
        }
    }
}

void ProcessDSMC::advection(int istep)
{
    this->istep = istep;
    for (int i = 0; i < this->iNcell; ++i)
    {
        const int meshindex = this->globalOfLocalCell(i);
        if (meshindex < 0 || this->partinit == nullptr) continue;
        const double cell_time_begin = MPI_Wtime();
        ParticleBucketSoA &bucket = this->currParticles(meshindex);
        size_t keep_count = 0;
        const size_t bucketSize = bucket.size();
        for (size_t p = 0; p < bucketSize; ++p)
        {
            double dtleft = this->mess.dtime;
            particle part;
            part.p_serial = bucket.p_serial[p];
            part.p_rank_serial = bucket.p_rank_serial[p];
            part.p_mesh_serial = bucket.p_mesh_serial[p];
            part.p_velocity[0] = bucket.vx[p];
            part.p_velocity[1] = bucket.vy[p];
            part.p_velocity[2] = bucket.vz[p];
            part.p_location[0] = bucket.px[p];
            part.p_location[1] = bucket.py[p];
            part.p_location[2] = bucket.pz[p];
            part.p_Ir = bucket.p_Ir[p];
            part.dt_left = bucket.dt_left[p];
            int cellid = i;
            int crossfid = -1;
            int ifout = 0;
            int tag_triangle = -1;
            ParticleTransportOps::traceParticleDtleft(*this, part, cellid, crossfid, tag_triangle, dtleft, ifout, true);
            if (ifout == 1)
            {
                const int gdst = this->globalOfLocalCell(cellid);
                if (gdst >= 0)
                {
                    part.p_mesh_serial = cellid;
                    part.p_rank_serial = this->c_rank;
                    part.dt_left = 0.0;
                    this->recv_cache.push_back(part);
                }
                continue;
            }
            if (ifout == -1 || ifout == 2)
                continue;
            bucket.p_serial[keep_count] = (int)keep_count;
            bucket.p_rank_serial[keep_count] = this->c_rank;
            bucket.p_mesh_serial[keep_count] = i;
            bucket.vx[keep_count] = part.p_velocity[0];
            bucket.vy[keep_count] = part.p_velocity[1];
            bucket.vz[keep_count] = part.p_velocity[2];
            bucket.px[keep_count] = part.p_location[0];
            bucket.py[keep_count] = part.p_location[1];
            bucket.pz[keep_count] = part.p_location[2];
            bucket.p_Ir[keep_count] = part.p_Ir;
            bucket.dt_left[keep_count] = 0.0;
            ++keep_count;
        }
        bucket.resize(keep_count);
        if (meshindex < (int)this->partinit->cell_particle_reserve_hint.size())
        {
            this->partinit->cell_particle_reserve_hint[(std::size_t)meshindex] =
                std::max(this->partinit->cell_particle_reserve_hint[(std::size_t)meshindex],
                         (int)bucket.size());
        }
        accumulate_cell_time_weight(meshindex, MPI_Wtime() - cell_time_begin);
    }
    dt_ray_crosscell();
    if (!migrateParticlesOnce())
        cout << "PARTICLE_MIGRATE_ONCE_STAGE_FAIL rank=" << this->c_rank
             << " stage=advection" << endl;
}

void ProcessDSMC::cache2chain()
{
    if (this->partinit == nullptr)
    {
        this->recv_cache.clear();
        return;
    }
    const double cache_time_begin = MPI_Wtime();
    this->cacheTouchedCells.clear();
    if (this->enable_partition_time_weights && this->mess.Ncell > 0)
    {
        if ((int)this->cacheTouchedStamp.size() != this->mess.Ncell)
            this->cacheTouchedStamp.assign((std::size_t)this->mess.Ncell, 0u);
        ++this->cacheTouchedEpoch;
        if (this->cacheTouchedEpoch == 0u)
        {
            std::fill(this->cacheTouchedStamp.begin(), this->cacheTouchedStamp.end(), 0u);
            this->cacheTouchedEpoch = 1u;
        }
    }
    auto markTouched = [&](int globalCell)
    {
        if (!this->enable_partition_time_weights) return;
        if (globalCell < 0 || globalCell >= (int)this->cacheTouchedStamp.size()) return;
        unsigned int& stamp = this->cacheTouchedStamp[(std::size_t)globalCell];
        if (stamp == this->cacheTouchedEpoch) return;
        stamp = this->cacheTouchedEpoch;
        this->cacheTouchedCells.push_back(globalCell);
    };
    const int cache_num = (int)this->recv_cache.size();
    for (int j = 0; j < cache_num; ++j)
    {
        particle parts = this->recv_cache[(std::size_t)j];
        const int localCell = parts.p_mesh_serial;
        const int globalCell = this->globalOfLocalCell(localCell);
        if (globalCell < 0) continue;
        if (!this->isOwnedLocalCell(localCell))
        {
            cout << "CACHE2CHAIN_SKIP_HALO_PARTICLE rank=" << this->c_rank
                 << " local=" << localCell
                 << " global=" << globalCell
                 << " owner=" << this->ownerOfGlobalCell(globalCell)
                 << endl;
            continue;
        }
        markTouched(globalCell);
        ParticleBucketSoA &bucket = this->currParticles(globalCell);
        parts.p_serial = (int)bucket.size();
        parts.p_rank_serial = this->c_rank;
        parts.p_mesh_serial = localCell;
        parts.dt_left = 0.0;
        bucket.push_back(parts);
        if (globalCell < (int)this->partinit->cell_particle_reserve_hint.size())
        {
            this->partinit->cell_particle_reserve_hint[(std::size_t)globalCell] =
                std::max(this->partinit->cell_particle_reserve_hint[(std::size_t)globalCell],
                         (int)bucket.size());
        }
    }
    for (auto it = this->partinit->cell_particles_next.begin();
         it != this->partinit->cell_particles_next.end(); )
    {
        const int gid = it->first;
        const int localCell = this->localOfGlobalCell(gid);
        ParticleBucketSoA &nextBuf = it->second;
        if (gid < 0 || localCell < 0 || !this->isOwnedLocalCell(localCell))
        {
            it = this->partinit->cell_particles_next.erase(it);
            continue;
        }
        if (nextBuf.empty())
        {
            it = this->partinit->cell_particles_next.erase(it);
            continue;
        }
        markTouched(gid);
        ParticleBucketSoA &bucket = this->currParticles(gid);
        const size_t oldSize = bucket.size();
        bucket.append_bucket(nextBuf);
        for (size_t p = oldSize; p < bucket.size(); ++p)
        {
            bucket.p_serial[p] = (int)p;
            bucket.p_rank_serial[p] = this->c_rank;
            bucket.p_mesh_serial[p] = localCell;
            bucket.dt_left[p] = 0.0;
        }
        if (gid < (int)this->partinit->cell_particle_reserve_hint.size())
        {
            this->partinit->cell_particle_reserve_hint[(std::size_t)gid] =
                std::max(this->partinit->cell_particle_reserve_hint[(std::size_t)gid],
                         (int)bucket.size());
        }
        it = this->partinit->cell_particles_next.erase(it);
    }
    this->recv_cache.clear();
    const int touched_count = (int)this->cacheTouchedCells.size();
    if (touched_count > 0)
    {
        const double share = (MPI_Wtime() - cache_time_begin) / (double)touched_count;
        for (int gid : this->cacheTouchedCells)
            accumulate_cell_time_weight(gid, share);
    }
#ifdef CHECK_PARTICLE_BUCKETS
    checkParticleBucketConsistency("cache2chain");
#endif
}

bool ProcessDSMC::migrateParticlesOnce()
{
    if (this->partinit == nullptr || this->partinit->mpass == nullptr || this->mpi == nullptr)
        return false;
    if (!this->mpi->active())
        return true;
    if ((int)this->migrate_send_particles.size() < this->c_size)
        this->migrate_send_particles.resize((std::size_t)this->c_size);
    if ((int)this->migrate_recv_particles.size() < this->c_size)
        this->migrate_recv_particles.resize((std::size_t)this->c_size);
    if ((int)this->migrationPeerMask.size() != this->c_size)
        rebuild_migration_peer_cache();
    const bool allEmpty = std::all_of(
        this->migrate_send_particles.begin(),
        this->migrate_send_particles.end(),
        [](const std::vector<particle>& v){ return v.empty(); });
    int localNeedFull = 0;
    if (!allEmpty)
    {
        for (int r = 0; r < this->c_size; ++r)
        {
            if (r == this->c_rank) continue;
            if (this->migrate_send_particles[(std::size_t)r].empty()) continue;
            if ((int)this->migrationPeerMask.size() != this->c_size ||
                !this->migrationPeerMask[(std::size_t)r])
            {
                localNeedFull = 1;
                break;
            }
        }
    }
    int localState[2] = { allEmpty ? 0 : 1, localNeedFull };
    int globalState[2] = {0, 0};
    MPI_Allreduce(localState, globalState, 2, MPI_INT, MPI_LOR, this->calGroup);
    if (globalState[0] == 0)
        return true;
    const int globalNeedFull = globalState[1];
    if (globalNeedFull != 0 && localNeedFull != 0)
    {
        int targetCount = 0;
        int packetCount = 0;
        for (int r = 0; r < this->c_size; ++r)
        {
            if (r == this->c_rank) continue;
            const int n = (int)this->migrate_send_particles[(std::size_t)r].size();
            if (n <= 0) continue;
            ++targetCount;
            packetCount += n;
        }
        cout << "PARTICLE_MIGRATE_PEER_FALLBACK rank=" << this->c_rank
             << " targets=" << targetCount
             << " packets=" << packetCount << endl;
    }
    const bool ok = (globalNeedFull == 0)
        ? this->partinit->mpass->exchangeParticleVectorsOnPeers(
              this->migrate_send_particles,
              this->migrate_recv_particles,
              *this->mpi,
              this->migrationPeerRanks,
              3201)
        : this->partinit->mpass->exchangeParticleVectors(
              this->migrate_send_particles,
              this->migrate_recv_particles,
              *this->mpi,
              3201);
    if (!ok)
    {
        cout << "PARTICLE_MIGRATE_ONCE_FAIL rank=" << this->c_rank << endl;
        return false;
    }
    for (int r = 0; r < this->c_size; ++r)
        this->migrate_send_particles[(std::size_t)r].clear();
    for (int src = 0; src < this->c_size; ++src)
    {
        std::vector<particle>& incoming = this->migrate_recv_particles[(std::size_t)src];
        const int recvN = (int)incoming.size();
        if (recvN == 0) continue;
        for (int j = 0; j < recvN; ++j)
        {
            particle part = incoming[(std::size_t)j];
            const int globalCell = part.p_mesh_serial;
            const int localCell = this->localOfGlobalCell(globalCell);
            if (localCell < 0 || !this->isOwnedLocalCell(localCell))
            {
                cout << "PARTICLE_MIGRATE_ONCE_BAD_CELL rank=" << this->c_rank
                     << " src=" << src
                     << " gcell=" << globalCell
                     << " local=" << localCell
                     << " owner=" << this->ownerOfGlobalCell(globalCell)
                     << endl;
                continue;
            }
            part.p_mesh_serial = localCell;
            part.p_rank_serial = this->c_rank;
            part.dt_left = 0.0;
            this->recv_cache.push_back(part);
        }
        incoming.clear();
    }
    return true;
}

bool ProcessDSMC::cache2chain_dt(int localNeedFull, int globalNeedFull)
{
    if (this->partinit == nullptr || this->partinit->mpass == nullptr || this->mpi == nullptr)
        return false;
    if ((int)this->dtleft_send_packets.size() < this->c_size)
        this->dtleft_send_packets.resize((std::size_t)this->c_size);
    if ((int)this->dtleft_recv_packets.size() < this->c_size)
        this->dtleft_recv_packets.resize((std::size_t)this->c_size);
    bool hadTraffic = false;
    if (globalNeedFull != 0 && localNeedFull != 0)
    {
        int targetCount = 0;
        int packetCount = 0;
        for (int r = 0; r < this->c_size; ++r)
        {
            if (r == this->c_rank) continue;
            const int n = (int)this->dtleft_send_packets[(std::size_t)r].size();
            if (n <= 0) continue;
            ++targetCount;
            packetCount += n;
        }
        cout << "DTLEFT_PEER_FALLBACK rank=" << this->c_rank
             << " targets=" << targetCount
             << " packets=" << packetCount << endl;
    }
    const bool ok = (globalNeedFull == 0)
        ? this->partinit->mpass->exchangeDtleftPacketVectorsOnPeers(this->dtleft_send_packets,
                                                                     this->dtleft_recv_packets,
                                                                     *this->mpi,
                                                                     this->migrationPeerRanks,
                                                                     1,
                                                                     &hadTraffic)
        : this->partinit->mpass->exchangeDtleftPacketVectors(this->dtleft_send_packets,
                                                              this->dtleft_recv_packets,
                                                              *this->mpi,
                                                              1,
                                                              &hadTraffic);
    if (!ok)
    {
        cout << "DTLEFT_PACKET_EXCHANGE_FAIL rank=" << this->c_rank << endl;
        return false;
    }
    for (int i = 0; i < c_size; ++i)
        this->dtleft_send_packets[i].clear();
    return hadTraffic;
}
ProcessDSMC::BoundaryGasState ProcessDSMC::computeBoundaryGasState(const BoundaryEmitPatch& patch)
{
    BoundaryGasState gas;
    const BoundaryCondition& boundary = patch.boundary;
    double rhoNormalized = boundary.rho;
    double uxNormalized = boundary.ux;
    double uyNormalized = boundary.uy;
    double uzNormalized = boundary.uz;
    double tNormalized = boundary.temperature;
    switch (boundary.dsmcModel)
    {
        case BoundaryModel::FreestreamInlet:
            uxNormalized = boundary.freestreamUxScale * this->mess.Ma * sqrt(this->mess.gamma / 2.0) *this->mess.v_in/this->mess.v_rms * sqrt(3)/2;
            uyNormalized = boundary.freestreamUxScale * this->mess.Ma * sqrt(this->mess.gamma / 2.0) *this->mess.v_in/this->mess.v_rms /2;
            tNormalized = this->mess.T_in / this->mess.T_ref;
            break;
        default:
        {
            const DsmcReservoirBoundaryConfig& config = reservoirConfigForState(patch.reservoirState);
            rhoNormalized = config.rho;
            uxNormalized = config.useFreestreamUx
                ? config.freestreamUxScale * this->mess.Ma * sqrt(this->mess.gamma / 2.0) *this->mess.v_in/this->mess.v_rms * sqrt(3)/2
                : config.ux;
            uyNormalized = config.useFreestreamUx
                ? config.freestreamUxScale * this->mess.Ma * sqrt(this->mess.gamma / 2.0) *this->mess.v_in/this->mess.v_rms /2
                : config.ux;
            uzNormalized = config.uz;
            tNormalized = config.useInletTemperature
                ? this->mess.T_in / this->mess.T_ref
                : config.temperature;
            break;
        }
    }
    gas.rhoPhysical = rhoNormalized * this->mess.n_ref;
    gas.uxPhysical = uxNormalized * this->mess.v_rms;
    gas.uyPhysical = uyNormalized * this->mess.v_rms;
    gas.uzPhysical = uzNormalized * this->mess.v_rms;
    gas.tNormalized = tNormalized;
    gas.tPhysical = gas.tNormalized * this->mess.T_ref;
    double beta = 1.0 / sqrt(2 * this->mess.kB / this->mess.p_mass * gas.tPhysical);
    gas.sx = gas.uxPhysical * beta;
    gas.sy = gas.uyPhysical * beta;
    gas.sz = gas.uzPhysical * beta;
    gas.stp1 = patch.rotation[0][0] * gas.sx + patch.rotation[1][0] * gas.sy + patch.rotation[2][0] * gas.sz;
    gas.stp2 = patch.rotation[0][1] * gas.sx + patch.rotation[1][1] * gas.sy + patch.rotation[2][1] * gas.sz;
    gas.snp  = patch.rotation[0][2] * gas.sx + patch.rotation[1][2] * gas.sy + patch.rotation[2][2] * gas.sz;
    gas.fs1 = gas.snp + sqrt(gas.snp*gas.snp + 2.0);
    gas.fs2 = 0.5 * (1.0 + gas.snp * (2.0 * gas.snp - gas.fs1));
    return gas;
}

int ProcessDSMC::computeBoundaryInsertCount(const BoundaryEmitPatch& patch, BoundaryGasState& gas)
{
    double cos_theta = 0.0;
    double s = 0.0;
    double add_num;
    double Umag = sqrt(gas.uxPhysical * gas.uxPhysical + gas.uyPhysical * gas.uyPhysical + gas.uzPhysical * gas.uzPhysical);
    double beta = 1.0 / sqrt(2 * this->mess.kB / this->mess.p_mass * gas.tPhysical);
    if (Umag > 0.0 && patch.normalMag > 0.0)
    {
        cos_theta = -((gas.uxPhysical * patch.normal[0] +
                       gas.uyPhysical * patch.normal[1] +
                       gas.uzPhysical * patch.normal[2]) / (Umag * patch.normalMag));
    }
    s = beta * Umag;
    const double scos = s * cos_theta;
    add_num = this->mess.dtime * patch.area * gas.rhoPhysical / 2.0 / sqrt(M_PI) *
              sqrt(gas.tPhysical / this->mess.T_ref) *
              (exp(-(scos * scos)) +
               sqrt(M_PI) * scos * (1.0 + erf(scos)));
    add_num = add_num / this->mess.Neff + partinit->remainderinpre[(std::size_t)patch.cellLocal];
    int N_add = floor(add_num);
    partinit->remainderinpre[(std::size_t)patch.cellLocal] = add_num - N_add;
    return N_add;
}

particle ProcessDSMC::sampleBoundaryParticle(const BoundaryEmitPatch& patch, const BoundaryGasState& gas)
{
    auto& gen = partinit->thread_rng();
    auto& dis = partinit->get_uniform();
    particle part;
    double r;
    double vxp_absolute, vyp_absolute, vzp_absolute;
    double vxp_relative, vyp_relative, vzp_relative;
    double QA = 3.0;
    if (abs(gas.snp) > 3.0) { QA = abs(gas.snp) + 1.0; }
    int loops = 0;
    while (true)
    {
        loops++;
        int loopas = 0;
        while (true)
        {
            loopas++;
            r = dis(gen);
            vzp_relative = (-QA + 2.0 * QA * r) * sqrt(gas.tNormalized);
            vzp_absolute = vzp_relative / sqrt(gas.tNormalized) + gas.snp;
            if (vzp_absolute > 0) { break; }
            if (loopas == 10001) { cout << "Warning: too many loops in prepro" << endl; }
        }
        const double vzp_norm = vzp_relative / sqrt(gas.tNormalized);
        double P_Pmax = (2.0 * vzp_absolute / gas.fs1) * exp(gas.fs2 - vzp_norm * vzp_norm);
        if (loops == 10001) { cout << "Warning: too many loops in preprocessp_max" << endl; }
        if (dis(gen) < P_Pmax) { break; }
    }
    r = dis(gen);
    double theta = 2 * M_PI * dis(gen);
    vxp_relative = sqrt(-log(r)) * sin(theta);
    vyp_relative = sqrt(-log(r)) * cos(theta);
    vxp_absolute = (vxp_relative + gas.stp1) * sqrt(gas.tNormalized);
    vyp_absolute = (vyp_relative + gas.stp2) * sqrt(gas.tNormalized);
    vzp_absolute *= sqrt(gas.tNormalized);
    part.p_velocity[0] = patch.rotation[0][0] * vxp_absolute + patch.rotation[0][1] * vyp_absolute + patch.rotation[0][2] * vzp_absolute;
    part.p_velocity[1] = patch.rotation[1][0] * vxp_absolute + patch.rotation[1][1] * vyp_absolute + patch.rotation[1][2] * vzp_absolute;
    part.p_velocity[2] = patch.rotation[2][0] * vxp_absolute + patch.rotation[2][1] * vyp_absolute + patch.rotation[2][2] * vzp_absolute;
    r = dis(gen);
    part.p_Ir = -log(r) * gas.tNormalized * 0.5;
    return part;
}

void ProcessDSMC::traceBoundaryParticle(particle& part, const BoundaryEmitPatch& patch, double dtleft)
{
    int ifout = 0;
    int cellid = patch.cellLocal;
    int crossedgeid = patch.edgeid;
    int legacy_tag_triangle = patch.triTag;
    ParticleTransportOps::traceParticleDtleft(*this, part, cellid, crossedgeid, legacy_tag_triangle, dtleft, ifout, true);
    if (ifout == 0 || ifout == 1)
    {
        if (this->globalOfLocalCell(cellid) >= 0)
        {
            part.p_mesh_serial = cellid;
            part.p_rank_serial = this->c_rank;
            part.dt_left = 0.0;
            this->recv_cache.push_back(part);
        }
    }
}

int ProcessDSMC::emitParticlesFromBoundaryPatch(const BoundaryEmitPatch& patch, double& expectedCount)
{
    expectedCount = 0.0;
    if (!patch.boundary.injectParticles) return 0;
    if (patch.edgeid < 0 || patch.edgeid >= (int)this->edges.size()) return 0;
    if (patch.cellLocal < 0 || patch.cellLocal >= (int)this->partinit->remainderinpre.size()) return 0;
    auto& gen = partinit->thread_rng();
    auto& dis = partinit->get_uniform();
    BoundaryGasState gas = computeBoundaryGasState(patch);
    int N_add = computeBoundaryInsertCount(patch, gas);
    expectedCount = N_add + partinit->remainderinpre[(std::size_t)patch.cellLocal];
    int k = 0;
    int loop = 0;
    while (k < N_add)
    {
        particle part;
        part.p_serial = 0;
        part.p_rank_serial = this->c_rank;
        part.dt_left = 0;
        double u = dis(gen);
        double v = dis(gen);
        if (u + v > 1.0) { u = 1.0 - u; v = 1.0 - v; }
        part.p_location[0] = patch.p0[0] + u * patch.e10[0] + v * patch.e20[0];
        part.p_location[1] = patch.p0[1] + u * patch.e10[1] + v * patch.e20[1];
        part.p_location[2] = patch.p0[2] + u * patch.e10[2] + v * patch.e20[2];
        loop++;
        if (loop > 10000)
        {
            cout << "Warning: too many loops in preprocess, state=0, N_add=" << N_add << endl;
            break;
        }
        particle sampledPart = sampleBoundaryParticle(patch, gas);
        part.p_velocity[0] = sampledPart.p_velocity[0];
        part.p_velocity[1] = sampledPart.p_velocity[1];
        part.p_velocity[2] = sampledPart.p_velocity[2];
        part.p_Ir = sampledPart.p_Ir;
        double r = dis(gen);
        double dtleft = r * this->mess.dtime;
        traceBoundaryParticle(part, patch, dtleft);
        k++;
    }
    return N_add;
}

void ProcessDSMC::preprocesseffquad(int istep)
{
    this->istep = istep;
    if (boundaryEmitCacheDirty)
        rebuildBoundaryEmitCache();
    for (const BoundaryEmitPatch& patch : boundaryEmitCache)
    {
        double expectedCount = 0.0;
        emitParticlesFromBoundaryPatch(patch, expectedCount);
    }
    dt_ray_crosscell();
    if (!migrateParticlesOnce())
        cout << "PARTICLE_MIGRATE_ONCE_STAGE_FAIL rank=" << this->c_rank
             << " stage=preprocess" << endl;
}

double ProcessDSMC::collision_diameter(double cr)
{
    const double con_denominator = tgamma(5.0/2.0 - this->mess.omega);
    const double d_ref2 = this->mess.d_ref * this->mess.d_ref;
    const double collisionDiameterPrefactor =
        M_PI * d_ref2 * pow(2*this->mess.kB*this->mess.T_ref/this->mess.p_mass_r, this->mess.eta) /
        con_denominator;
    return collision_diameter(cr, collisionDiameterPrefactor);
}

double ProcessDSMC::collision_diameter(double cr, double collisionDiameterPrefactor)
{
    const double cr2 = cr * cr;
    return cr * collisionDiameterPrefactor * pow(1.0 / cr2, this->mess.eta);
}

void ProcessDSMC::collisionDSMC()
{
    tol_collision_pairs = 0;
    tol_collision_times = 0;
    const double vrms = this->mess.v_rms;
    const double con_denominator = tgamma(5.0/2.0 - this->mess.omega);
    const double d_ref2 = this->mess.d_ref * this->mess.d_ref;
    const double collisionDiameterPrefactor =
        M_PI * d_ref2 * pow(2*this->mess.kB*this->mess.T_ref/this->mess.p_mass_r, this->mess.eta) /
        con_denominator;
    for (int i = 0; i < this->iNcell; ++i)
    {
        auto& gen = partinit->thread_rng();
        int meshindex = this->globalOfLocalCell(i);
        if (meshindex < 0 || this->partinit == nullptr) continue;
        const double cell_time_begin = MPI_Wtime();
        ParticleBucketSoA &bucket = this->currParticles(meshindex);
        int Np_cell = (int)bucket.size();
        if (Np_cell <= 1)
        {
            accumulate_cell_time_weight(meshindex, MPI_Wtime() - cell_time_begin);
            continue;
        }
        const double S = this->cells[i].area;
        double crm = partinit->crmax[i];
        const double consant_crm = collision_diameter(crm * vrms, collisionDiameterPrefactor);
        const double pairs = 0.5 * Np_cell * (Np_cell - 1) * this->mess.Neff * consant_crm * this->mess.dtime * this->mess.dt_ref / S + partinit->remainderincoll[i];
        const int npairs = (int)floor(pairs);
        partinit->remainderincoll[i] = pairs - npairs;
        tol_collision_pairs += npairs;
        std::uniform_int_distribution<int> dist(0, Np_cell - 1);
        for (int j = 0; j < npairs; ++j)
        {
            const int kp1 = dist(gen);
            int kp2 = 0;
            if (Np_cell == 2) kp2 = (kp1 == 0) ? 1 : 0;
            else
            {
                do { kp2 = dist(gen); } while (kp1 == kp2);
            }
            collisionVelocity(bucket, kp1, kp2, tol_collision_times, consant_crm, crm, collisionDiameterPrefactor);
        }
        partinit->crmax[i] = crm;
        accumulate_cell_time_weight(meshindex, MPI_Wtime() - cell_time_begin);
    }
}

void ProcessDSMC::collisionVelocity(ParticleBucketSoA& bucket, int idx1, int idx2, int& tol_collision_times,
                                    double consant_crm, double& crm, double collisionDiameterPrefactor)
{
    if (idx1 < 0 || idx2 < 0) return;
    if (idx1 >= (int)bucket.size() || idx2 >= (int)bucket.size()) return;
    if (!(consant_crm > 0.0)) return;
    auto& gen = partinit->thread_rng();
    auto& dis = partinit->get_uniform();
    const std::size_t p1 = (std::size_t)idx1;
    const std::size_t p2 = (std::size_t)idx2;
    double &vx1 = bucket.vx[p1];
    double &vy1 = bucket.vy[p1];
    double &vz1 = bucket.vz[p1];
    double &vx2 = bucket.vx[p2];
    double &vy2 = bucket.vy[p2];
    double &vz2 = bucket.vz[p2];
    double &Ir1 = bucket.p_Ir[p1];
    double &Ir2 = bucket.p_Ir[p2];
    const double vr0 = vx1 - vx2;
    const double vr1 = vy1 - vy2;
    const double vr2 = vz1 - vz2;
    const double alpha = this->mess.alpha;
    const double cr2 = vr0*vr0 + vr1*vr1 + vr2*vr2;
    const double cr = sqrt(cr2);
    double pre_Ert = 0.5 * 0.5 * cr2;
    if (cr > crm) crm = cr;
    const double consant_cr = collision_diameter(cr*this->mess.v_rms, collisionDiameterPrefactor);
    const double accept = consant_cr / consant_crm;
    if (dis(gen) >= accept) return;
    for (int h = 0; h < 2; ++h)
    {
        double *Ir_relax = (h == 0) ? &Ir1 : &Ir2;
        if (dis(gen) <= this->mess.P_relax)
        {
            const double pre_Er = *Ir_relax;
            const double r = dis(gen);
            const double post_Er = (pre_Ert + pre_Er) * (1.0-pow(r,(1.0/(2.5-this->mess.omega))));
            *Ir_relax = post_Er;
            pre_Ert = pre_Ert + pre_Er - post_Er;
        }
    }
    const double post_Vr2 = 2.0 * pre_Ert / 0.5;
    const double post_Vr = sqrt(post_Vr2);
    const double vc0 = 0.5 * (vx1 + vx2);
    const double vc1 = 0.5 * (vy1 + vy2);
    const double vc2 = 0.5 * (vz1 + vz2);
    const double r = dis(gen);
    const double cos_chi = 2.0*pow(r,1.0/alpha)-1.0;
    const double sin_chi = sqrt(1.0-cos_chi*cos_chi);
    const double theta = 2*M_PI*dis(gen);
    const double vrp2 = vr1*vr1 + vr2*vr2;
    const double vrp  = sqrt(vrp2);
    const double scale = post_Vr / std::max(cr, 1.0e-300);
    double vref1 = scale * (cos_chi*vr0+sin_chi*sin(theta)*vrp);
    double vref2 = scale * (cos_chi*vr1+sin_chi*(cr*vr2*cos(theta)-vr0*vr1*sin(theta))/std::max(vrp, 1.0e-300));
    double vref3 = scale * (cos_chi*vr2-sin_chi*(cr*vr1*cos(theta)+vr0*vr2*sin(theta))/std::max(vrp, 1.0e-300));
    if (vrp < 1e-6)
    {
        vref1 = scale * cos_chi*vr0;
        vref2 = scale * sin_chi*vr0*cos(theta);
        vref3 = scale * sin_chi*vr0*sin(theta);
    }
    vx1 = vc0 + 0.5*vref1; vy1 = vc1 + 0.5*vref2; vz1 = vc2 + 0.5*vref3;
    vx2 = vc0 - 0.5*vref1; vy2 = vc1 - 0.5*vref2; vz2 = vc2 - 0.5*vref3;
    tol_collision_times++;
}

void ProcessDSMC::statistic_macro(int istep)
{
    current_macro_zero();
    const double stepWindowSamples = static_cast<double>(NSCHEME);
    const int stepIntervalSamples = std::max(1, Nrepeat);
    const bool dsmc2nsExchangeStep = (istep%NGSIS==0);
    if ((int)this->dsmc2ns_sparse_state.size() != this->iNcell)
        this->dsmc2ns_sparse_state.assign((std::size_t)this->iNcell, DSMC2NS_SPARSE_NORMAL);
    if ((int)this->dsmc2ns_sparse_accum_steps.size() != this->iNcell)
        this->dsmc2ns_sparse_accum_steps.assign((std::size_t)this->iNcell, 0);
    if (dsmc2nsExchangeStep)
    {
        if ((int)this->dsmc2ns_window_samples.size() != this->iNcell)
            this->dsmc2ns_window_samples.assign((std::size_t)this->iNcell, 0.0);
        if ((int)this->dsmc2ns_window_valid.size() != this->iNcell)
            this->dsmc2ns_window_valid.assign((std::size_t)this->iNcell, 0);
    }
    for (int i = 0;i< this->iNcell;i++)
    {
        int meshindex = this->globalOfLocalCell(i);
        if (meshindex < 0)
        {
            if (dsmc2nsExchangeStep)
            {
                this->dsmc2ns_window_samples[(std::size_t)i] = 0.0;
                this->dsmc2ns_window_valid[(std::size_t)i] = 0;
            }
            continue;
        }
        ParticleBucketSoA &bucket = currParticles(meshindex);
        MacroMoments stepMoments;
        accumulateCellMoments(i, bucket, stepMoments);
        updateStepWindowMoments(i, istep);
        MacroMoments windowMoments = loadMacroMoments(
            i, stepsum_rho, stepsum_U, stepsum_T, stepsum_sigma, stepsum_q, stepsum_qr);
        writeAveragedMacro(i, windowMoments, stepWindowSamples, this->local);
        if (istep>NSS)
        {
            accumulateStepMomentsInto(
                i, steady_rho, steady_U, steady_T, steady_sigma, steady_q, steady_qr, 1.0);
        }
        if (istep%Nevery==0)
        {
            accumulateStepMomentsInto(
                i, stepinter_rho, stepinter_U, stepinter_T,
                stepinter_sigma, stepinter_q, stepinter_qr, 1.0);
            unsigned char sparseState =
                this->dsmc2ns_sparse_state[(std::size_t)i];
            if (sparseState == DSMC2NS_SPARSE_ACCUMULATING ||
                sparseState == DSMC2NS_SPARSE_RELEASED)
            {
                int &accumSteps =
                    this->dsmc2ns_sparse_accum_steps[(std::size_t)i];
                if (accumSteps < 0) accumSteps = 0;
                ++accumSteps;
            }
        }
        if((istep > NSS))
        {
            MacroMoments steadyMoments = loadMacroMoments(
                i, steady_rho, steady_U, steady_T, steady_sigma, steady_q, steady_qr);
            writeAveragedMacro(
                i, steadyMoments, static_cast<double>(istep - NSS), this->final_record);
        }
        if(dsmc2nsExchangeStep)
        {
            MacroMoments stepIntervalMoments = loadMacroMoments(
                i, stepinter_rho, stepinter_U, stepinter_T,
                stepinter_sigma, stepinter_q, stepinter_qr);
            const double windowSamples = stepIntervalMoments.rho;
            const bool windowSamplesFinite = std::isfinite(windowSamples);
            unsigned char &sparseState =
                this->dsmc2ns_sparse_state[(std::size_t)i];
            int &sparseAccumSteps =
                this->dsmc2ns_sparse_accum_steps[(std::size_t)i];
            bool sparseCell =
                (sparseState == DSMC2NS_SPARSE_ACCUMULATING ||
                 sparseState == DSMC2NS_SPARSE_RELEASED);
            if (!sparseCell && windowSamplesFinite &&
                windowSamples < kDsmc2NsMinSampleCount)
            {
                sparseState = DSMC2NS_SPARSE_ACCUMULATING;
                sparseAccumSteps = stepIntervalSamples;
                sparseCell = true;
            }
            const bool windowValid = (windowSamples >= kDsmc2NsMinSampleCount) &&
                                     windowSamplesFinite;
            this->dsmc2ns_window_samples[(std::size_t)i] = windowSamples;
            this->dsmc2ns_window_valid[(std::size_t)i] = windowValid ? 1 : 0;
            if (sparseCell)
            {
                const int accumSteps = std::max(1, sparseAccumSteps);
                writeAveragedMacro(
                    i, stepIntervalMoments, static_cast<double>(accumSteps), this->record);
            }
            else if (windowValid)
            {
                writeAveragedMacro(
                    i, stepIntervalMoments, static_cast<double>(Nrepeat), this->record);
            }
            if (!sparseCell)
                stepinter_macro_zero(i);
        }
    }
}

void ProcessDSMC::stepinter_macro_zero(int icell)
{
    this->stepinter_rho[icell] = 0.0;
    this->stepinter_T[icell] = 0.0;
    const int axisBase = axisOffset(icell, AXIS_X);
    const int heatBase = heatOffset(icell, HEAT_X);
    const int stressBase = stressOffset(icell, STRESS_XX);
    const int rotBase = rotHeatOffset(icell, ROT_ENERGY);
    for (int j = 0; j < AXIS_WIDTH; j++)
    {
        this->stepinter_U[axisBase+j] = 0.0;
    }
    for (int j = 0; j < HEAT_WIDTH; j++)
    {
        this->stepinter_q[heatBase+j] = 0.0;
    }
    for (int j = 0; j < STRESS_WIDTH; j++)
    {
        this->stepinter_sigma[stressBase+j] = 0.0;
    }
    for (int j = 0; j < ROT_HEAT_WIDTH; j++)
    {
        this->stepinter_qr[rotBase+j] = 0.0;
    }
}

void ProcessDSMC::current_macro_zero()
{
    for (int i = 0;i<this->iNcell;i++)
    {
        this->step_rho[i] = 0.0;
        this->step_T[i] = 0.0;
        const int axisBase = axisOffset(i, AXIS_X);
        const int heatBase = heatOffset(i, HEAT_X);
        const int rotBase = rotHeatOffset(i, ROT_ENERGY);
        const int stressBase = stressOffset(i, STRESS_XX);
        for (int j = 0;j<AXIS_WIDTH;j++)
        {
            this->step_U[axisBase+j] = 0.0;
        }
        for (int j = 0;j<HEAT_WIDTH;j++)
        {
            this->step_q[heatBase+j] = 0.0;
        }
        for (int j = 0;j<ROT_HEAT_WIDTH;j++)
        {
            this->step_qr[rotBase+j] = 0.0;
        }
        for (int j = 0;j<STRESS_WIDTH;j++)
        {
            this->step_sigma[stressBase+j] = 0.0;
        }
    }
}

void ProcessDSMC::accumulateCellMoments(int localCell, const ParticleBucketSoA& bucket,
                                        MacroMoments& moments)
{
    const std::size_t bucketSize = bucket.size();
    const std::vector<double> &vx = bucket.vx;
    const std::vector<double> &vy = bucket.vy;
    const std::vector<double> &vz = bucket.vz;
    const std::vector<double> &pIr = bucket.p_Ir;
    const std::vector<int> &prank = bucket.p_rank_serial;
    const std::vector<int> &pserial = bucket.p_serial;
    for (std::size_t pi = 0; pi < bucketSize; ++pi)
    {
        const double vxi = vx[pi];
        const double vyi = vy[pi];
        const double vzi = vz[pi];
        const double Ir = pIr[pi];
        const double v2 = vxi*vxi + vyi*vyi + vzi*vzi;
        moments.rho += 1.0;
        moments.velocity[AXIS_X] += vxi;
        moments.velocity[AXIS_Y] += vyi;
        moments.velocity[AXIS_Z] += vzi;
        moments.kinetic += v2;
        moments.stress[STRESS_XX] += vxi*vxi;
        moments.stress[STRESS_XY] += vxi*vyi;
        moments.stress[STRESS_YY] += vyi*vyi;
        moments.stress[STRESS_YZ] += vyi*vzi;
        moments.stress[STRESS_ZZ] += vzi*vzi;
        moments.stress[STRESS_ZX] += vzi*vxi;
        moments.heat[HEAT_X] += vxi*v2;
        moments.heat[HEAT_Y] += vyi*v2;
        moments.heat[HEAT_Z] += vzi*v2;
        moments.rot[ROT_ENERGY] += Ir;
        moments.rot[ROT_HEAT_X] += vxi*Ir;
        moments.rot[ROT_HEAT_Y] += vyi*Ir;
        moments.rot[ROT_HEAT_Z] += vzi*Ir;
        if (prank[pi] != this->c_rank) {cout << "Error!: statistic_macro----rank is not equivelent"<<"rank "<< this->rank<<"particles "<<prank[pi]<< " " <<pserial[pi] <<endl;}
    }
    this->step_rho[localCell] = moments.rho;
    this->step_U[axisOffset(localCell, AXIS_X)] = moments.velocity[AXIS_X];
    this->step_U[axisOffset(localCell, AXIS_Y)] = moments.velocity[AXIS_Y];
    this->step_U[axisOffset(localCell, AXIS_Z)] = moments.velocity[AXIS_Z];
    this->step_T[localCell] = moments.kinetic;
    this->step_sigma[stressOffset(localCell, STRESS_XX)] = moments.stress[STRESS_XX];
    this->step_sigma[stressOffset(localCell, STRESS_XY)] = moments.stress[STRESS_XY];
    this->step_sigma[stressOffset(localCell, STRESS_YY)] = moments.stress[STRESS_YY];
    this->step_sigma[stressOffset(localCell, STRESS_YZ)] = moments.stress[STRESS_YZ];
    this->step_sigma[stressOffset(localCell, STRESS_ZZ)] = moments.stress[STRESS_ZZ];
    this->step_sigma[stressOffset(localCell, STRESS_ZX)] = moments.stress[STRESS_ZX];
    this->step_q[heatOffset(localCell, HEAT_X)] = moments.heat[HEAT_X];
    this->step_q[heatOffset(localCell, HEAT_Y)] = moments.heat[HEAT_Y];
    this->step_q[heatOffset(localCell, HEAT_Z)] = moments.heat[HEAT_Z];
    this->step_qr[rotHeatOffset(localCell, ROT_ENERGY)] = moments.rot[ROT_ENERGY];
    this->step_qr[rotHeatOffset(localCell, ROT_HEAT_X)] = moments.rot[ROT_HEAT_X];
    this->step_qr[rotHeatOffset(localCell, ROT_HEAT_Y)] = moments.rot[ROT_HEAT_Y];
    this->step_qr[rotHeatOffset(localCell, ROT_HEAT_Z)] = moments.rot[ROT_HEAT_Z];
}

ProcessDSMC::MacroMoments ProcessDSMC::loadMacroMoments(
    int localCell, const vector<double>& rho, const vector<double>& velocity,
    const vector<double>& kinetic, const vector<double>& stress,
    const vector<double>& heat, const vector<double>& rot) const
{
    MacroMoments moments;
    moments.rho = rho[localCell];
    moments.kinetic = kinetic[localCell];
    const int axisBase = axisOffset(localCell, AXIS_X);
    const int heatBase = heatOffset(localCell, HEAT_X);
    const int stressBase = stressOffset(localCell, STRESS_XX);
    const int rotBase = rotHeatOffset(localCell, ROT_ENERGY);
    for (int j = 0; j < AXIS_WIDTH; j++)
    {
        moments.velocity[j] = velocity[axisBase+j];
    }
    for (int j = 0; j < HEAT_WIDTH; j++)
    {
        moments.heat[j] = heat[heatBase+j];
    }
    for (int j = 0; j < STRESS_WIDTH; j++)
    {
        moments.stress[j] = stress[stressBase+j];
    }
    for (int j = 0; j < ROT_HEAT_WIDTH; j++)
    {
        moments.rot[j] = rot[rotBase+j];
    }
    return moments;
}

void ProcessDSMC::accumulateStepMomentsInto(
    int localCell, vector<double>& rho, vector<double>& velocity,
    vector<double>& kinetic, vector<double>& stress, vector<double>& heat,
    vector<double>& rot, double historyWeight)
{
    auto blend = [historyWeight](double& target, double value)
    {
        if (historyWeight == 1.0) target += value;
        else target = historyWeight * target + value;
    };
    blend(rho[localCell], this->step_rho[localCell]);
    blend(kinetic[localCell], this->step_T[localCell]);
    const int axisBase = axisOffset(localCell, AXIS_X);
    const int heatBase = heatOffset(localCell, HEAT_X);
    const int stressBase = stressOffset(localCell, STRESS_XX);
    const int rotBase = rotHeatOffset(localCell, ROT_ENERGY);
    for (int j = 0; j < AXIS_WIDTH; j++)
    {
        blend(velocity[axisBase+j], this->step_U[axisBase+j]);
    }
    for (int j = 0; j < HEAT_WIDTH; j++)
    {
        blend(heat[heatBase+j], this->step_q[heatBase+j]);
    }
    for (int j = 0; j < STRESS_WIDTH; j++)
    {
        blend(stress[stressBase+j], this->step_sigma[stressBase+j]);
    }
    for (int j = 0; j < ROT_HEAT_WIDTH; j++)
    {
        blend(rot[rotBase+j], this->step_qr[rotBase+j]);
    }
}

void ProcessDSMC::updateStepWindowMoments(int localCell, int istep)
{
    const double na = static_cast<double>(NSCHEME);
    const double historyWeight = (istep <= NSCHEME) ? 1.0 : (na - 1.0)/na;
    accumulateStepMomentsInto(
        localCell, stepsum_rho, stepsum_U, stepsum_T,
        stepsum_sigma, stepsum_q, stepsum_qr, historyWeight);
}

void ProcessDSMC::writeAveragedMacro(
    int localCell, const MacroMoments& moments, double sampleCount,
    vector<double>& target)
{
    const double Np = moments.rho;
    const double Neff = this->mess.Neff;
    const double S = this->cells[localCell].area;
    double* out = &target[macroOffset(localCell, MACRO_RHO)];
    if (!(Np > kMacroMinDivisor) || !std::isfinite(Np) ||
        !(sampleCount > 0.0) || !std::isfinite(sampleCount) ||
        !(S > 0.0) || !std::isfinite(S) ||
        !(Neff > 0.0) || !std::isfinite(Neff) ||
        !(this->mess.n_ref > 0.0) || !std::isfinite(this->mess.n_ref) ||
        !(this->mess.dr > 0.0) || !std::isfinite(this->mess.dr))
    {
        for (int j = 0; j < MACRO_WIDTH; ++j)
        {
            out[j] = 0.0;
        }
        return;
    }
    out[MACRO_RHO] = Neff/S*Np/sampleCount/this->mess.n_ref;
    out[MACRO_UX] = moments.velocity[AXIS_X]/Np;
    out[MACRO_UY] = moments.velocity[AXIS_Y]/Np;
    out[MACRO_UZ] = moments.velocity[AXIS_Z]/Np;
    const double ux2 = out[MACRO_UX] * out[MACRO_UX];
    const double uy2 = out[MACRO_UY] * out[MACRO_UY];
    const double uz2 = out[MACRO_UZ] * out[MACRO_UZ];
    const double u2 = ux2 + uy2 + uz2;
    out[MACRO_T] = 2.0/3.0/Np*(moments.kinetic-Np*u2);
    out[MACRO_SXX] = 2.0*out[MACRO_RHO]/Np*(2.0/3.0*moments.stress[STRESS_XX]-1.0/3.0*moments.stress[STRESS_YY]-1.0/3.0*moments.stress[STRESS_ZZ]-2.0/3.0*Np*ux2+1.0/3.0*Np*uy2+1.0/3.0*Np*uz2);
    out[MACRO_SXY] = 2.0*out[MACRO_RHO]/Np*(moments.stress[STRESS_XY]-Np*out[MACRO_UX]*out[MACRO_UY]);
    out[MACRO_SYY] = 2.0*out[MACRO_RHO]/Np*(2.0/3.0*moments.stress[STRESS_YY]-1.0/3.0*moments.stress[STRESS_XX]-1.0/3.0*moments.stress[STRESS_ZZ]-2.0/3.0*Np*uy2+1.0/3.0*Np*ux2+1.0/3.0*Np*uz2);
    out[MACRO_SYZ] = 2.0*out[MACRO_RHO]/Np*(moments.stress[STRESS_YZ]-Np*out[MACRO_UY]*out[MACRO_UZ]);
    out[MACRO_SZZ] = 2.0*out[MACRO_RHO]/Np*(2.0/3.0*moments.stress[STRESS_ZZ]-1.0/3.0*moments.stress[STRESS_XX]-1.0/3.0*moments.stress[STRESS_YY]-2.0/3.0*Np*uz2+1.0/3.0*Np*ux2+1.0/3.0*Np*uy2);
    out[MACRO_SZX] = 2.0*out[MACRO_RHO]/Np*(moments.stress[STRESS_ZX]-Np*out[MACRO_UZ]*out[MACRO_UX]);
    out[MACRO_QX] = out[MACRO_RHO]/Np*(moments.heat[HEAT_X]-2*out[MACRO_UX]*moments.stress[STRESS_XX]-2*out[MACRO_UY]*moments.stress[STRESS_XY]-2*out[MACRO_UZ]*moments.stress[STRESS_ZX]+u2*moments.velocity[AXIS_X])-out[MACRO_UX]*1.5*out[MACRO_T]*out[MACRO_RHO];
    out[MACRO_QY] = out[MACRO_RHO]/Np*(moments.heat[HEAT_Y]-2*out[MACRO_UX]*moments.stress[STRESS_XY]-2*out[MACRO_UY]*moments.stress[STRESS_YY]-2*out[MACRO_UZ]*moments.stress[STRESS_YZ]+u2*moments.velocity[AXIS_Y])-out[MACRO_UY]*1.5*out[MACRO_T]*out[MACRO_RHO];
    out[MACRO_QZ] = out[MACRO_RHO]/Np*(moments.heat[HEAT_Z]-2*out[MACRO_UX]*moments.stress[STRESS_ZX]-2*out[MACRO_UY]*moments.stress[STRESS_YZ]-2*out[MACRO_UZ]*moments.stress[STRESS_ZZ]+u2*moments.velocity[AXIS_Z])-out[MACRO_UZ]*1.5*out[MACRO_T]*out[MACRO_RHO];
    out[MACRO_TR] = 4.0/this->mess.dr/Np*moments.rot[ROT_ENERGY];
    out[MACRO_QRX] = 2.0*out[MACRO_RHO]/Np*(moments.rot[ROT_HEAT_X]-out[MACRO_UX]*moments.rot[ROT_ENERGY]);
    out[MACRO_QRY] = 2.0*out[MACRO_RHO]/Np*(moments.rot[ROT_HEAT_Y]-out[MACRO_UY]*moments.rot[ROT_ENERGY]);
    out[MACRO_QRZ] = 2.0*out[MACRO_RHO]/Np*(moments.rot[ROT_HEAT_Z]-out[MACRO_UZ]*moments.rot[ROT_ENERGY]);
    guardInvalidMacro(target, localCell);
}

void ProcessDSMC::guardInvalidMacro(vector<double>& target, int localCell)
{
    double* out = &target[macroOffset(localCell, MACRO_RHO)];
    bool valid = (out[MACRO_RHO] > 0.0) && std::isfinite(out[MACRO_RHO]) &&
                 (out[MACRO_T] > 0.0) && std::isfinite(out[MACRO_T]) &&
                 (out[MACRO_TR] > 0.0) && std::isfinite(out[MACRO_TR]);
    for (int j = 0; valid && j < MACRO_WIDTH; ++j)
    {
        valid = std::isfinite(out[j]);
    }
    if (!valid)
    {
        for (int j = 0; j < MACRO_WIDTH; j++)
        {
            out[j] = 0.0;
        }
    }
}

void ProcessDSMC::writeLocalMacro(int localCell, const MacroMoments& moments)
{
    double Nl = moments.rho;
    double* out = &this->local[macroOffset(localCell, MACRO_RHO)];
    if (!(Nl > 0.0) || !std::isfinite(Nl))
    {
        for (int j = 0; j < MACRO_WIDTH; ++j)
        {
            out[j] = 0.0;
        }
        return;
    }
    out[MACRO_RHO] = this->mess.Neff/this->cells[localCell].area*Nl/this->mess.n_ref;
    out[MACRO_UX] = moments.velocity[AXIS_X]/Nl;
    out[MACRO_UY] = moments.velocity[AXIS_Y]/Nl;
    out[MACRO_UZ] = moments.velocity[AXIS_Z]/Nl;
    const double ux2 = out[MACRO_UX] * out[MACRO_UX];
    const double uy2 = out[MACRO_UY] * out[MACRO_UY];
    const double uz2 = out[MACRO_UZ] * out[MACRO_UZ];
    double ul2 = ux2 + uy2 + uz2;
    out[MACRO_T] = 2.0/3.0/Nl*(moments.kinetic-Nl*ul2);
    out[MACRO_SXX] = 2.0*out[MACRO_RHO]/Nl*(2.0/3.0*moments.stress[STRESS_XX]-1.0/3.0*moments.stress[STRESS_YY]-1.0/3.0*moments.stress[STRESS_ZZ]-2.0/3.0*Nl*ux2+1.0/3.0*Nl*uy2+1.0/3.0*Nl*uz2);
    out[MACRO_SXY] = 2.0*out[MACRO_RHO]/Nl*(moments.stress[STRESS_XY]-Nl*out[MACRO_UX]*out[MACRO_UY]);
    out[MACRO_SYY] = 2.0*out[MACRO_RHO]/Nl*(2.0/3.0*moments.stress[STRESS_YY]-1.0/3.0*moments.stress[STRESS_XX]-1.0/3.0*moments.stress[STRESS_ZZ]-2.0/3.0*Nl*uy2+1.0/3.0*Nl*ux2+1.0/3.0*Nl*uz2);
    out[MACRO_SYZ] = 2.0*out[MACRO_RHO]/Nl*(moments.stress[STRESS_YZ]-Nl*out[MACRO_UY]*out[MACRO_UZ]);
    out[MACRO_SZZ] = 2.0*out[MACRO_RHO]/Nl*(2.0/3.0*moments.stress[STRESS_ZZ]-1.0/3.0*moments.stress[STRESS_XX]-1.0/3.0*moments.stress[STRESS_YY]-2.0/3.0*Nl*uz2+1.0/3.0*Nl*ux2+1.0/3.0*Nl*uy2);
    out[MACRO_SZX] = 2.0*out[MACRO_RHO]/Nl*(moments.stress[STRESS_ZX]-Nl*out[MACRO_UZ]*out[MACRO_UX]);
    out[MACRO_QX] = out[MACRO_RHO]/Nl*(moments.heat[HEAT_X]-2*out[MACRO_UX]*moments.stress[STRESS_XX]-2*out[MACRO_UY]*moments.stress[STRESS_XY]-2*out[MACRO_UZ]*moments.stress[STRESS_ZX]+ul2*moments.velocity[AXIS_X])-out[MACRO_UX]*1.5*out[MACRO_T]*out[MACRO_RHO];
    out[MACRO_QY] = out[MACRO_RHO]/Nl*(moments.heat[HEAT_Y]-2*out[MACRO_UX]*moments.stress[STRESS_XY]-2*out[MACRO_UY]*moments.stress[STRESS_YY]-2*out[MACRO_UZ]*moments.stress[STRESS_YZ]+ul2*moments.velocity[AXIS_Y])-out[MACRO_UY]*1.5*out[MACRO_T]*out[MACRO_RHO];
    out[MACRO_QZ] = out[MACRO_RHO]/Nl*(moments.heat[HEAT_Z]-2*out[MACRO_UX]*moments.stress[STRESS_ZX]-2*out[MACRO_UY]*moments.stress[STRESS_YZ]-2*out[MACRO_UZ]*moments.stress[STRESS_ZZ]+ul2*moments.velocity[AXIS_Z])-out[MACRO_UZ]*1.5*out[MACRO_T]*out[MACRO_RHO];
    out[MACRO_TR] = 4.0/this->mess.dr/Nl*(moments.rot[ROT_ENERGY]);
    out[MACRO_QRX] = 2.0*out[MACRO_RHO]/Nl*(moments.rot[ROT_HEAT_X]-out[MACRO_UX]*moments.rot[ROT_ENERGY]);
    out[MACRO_QRY] = 2.0*out[MACRO_RHO]/Nl*(moments.rot[ROT_HEAT_Y]-out[MACRO_UY]*moments.rot[ROT_ENERGY]);
    out[MACRO_QRZ] = 2.0*out[MACRO_RHO]/Nl*(moments.rot[ROT_HEAT_Z]-out[MACRO_UZ]*moments.rot[ROT_ENERGY]);
}

void ProcessDSMC::writeRecordMacro(int localCell, const MacroMoments& moments)
{
    double Np = moments.rho;
    double Neff = this->mess.Neff;
    double S = this->cells[localCell].area;
    double* out = &this->record[macroOffset(localCell, MACRO_RHO)];
    out[MACRO_RHO] = Neff/S*Np/this->mess.n_ref;
    out[MACRO_UX] = moments.velocity[AXIS_X]/Np;
    out[MACRO_UY] = moments.velocity[AXIS_Y]/Np;
    out[MACRO_UZ] = moments.velocity[AXIS_Z]/Np;
    const double ux2 = out[MACRO_UX] * out[MACRO_UX];
    const double uy2 = out[MACRO_UY] * out[MACRO_UY];
    const double uz2 = out[MACRO_UZ] * out[MACRO_UZ];
    double u2 = ux2 + uy2 + uz2;
    out[MACRO_T] = 2.0/3.0/Np*(moments.kinetic-Np*u2);
    out[MACRO_SXX] = 2.0*out[MACRO_RHO]/Np*(2.0/3.0*moments.stress[STRESS_XX]-1.0/3.0*moments.stress[STRESS_YY]-1.0/3.0*moments.stress[STRESS_ZZ]-2.0/3.0*Np*ux2+1.0/3.0*Np*uy2+1.0/3.0*Np*uz2);
    out[MACRO_SXY] = 2.0*out[MACRO_RHO]/Np*(moments.stress[STRESS_XY]-Np*out[MACRO_UX]*out[MACRO_UY]);
    out[MACRO_SYY] = 2.0*out[MACRO_RHO]/Np*(2.0/3.0*moments.stress[STRESS_YY]-1.0/3.0*moments.stress[STRESS_XX]-1.0/3.0*moments.stress[STRESS_ZZ]-2.0/3.0*Np*uy2+1.0/3.0*Np*ux2+1.0/3.0*Np*uz2);
    out[MACRO_SYZ] = 2.0*out[MACRO_RHO]/Np*(moments.stress[STRESS_YZ]-Np*out[MACRO_UY]*out[MACRO_UZ]);
    out[MACRO_SZZ] = 2.0*out[MACRO_RHO]/Np*(2.0/3.0*moments.stress[STRESS_ZZ]-1.0/3.0*moments.stress[STRESS_XX]-1.0/3.0*moments.stress[STRESS_YY]-2.0/3.0*Np*uz2+1.0/3.0*Np*ux2+1.0/3.0*Np*uy2);
    out[MACRO_SZX] = 2.0*out[MACRO_RHO]/Np*(moments.stress[STRESS_ZX]-Np*out[MACRO_UZ]*out[MACRO_UX]);
    out[MACRO_QX] = out[MACRO_RHO]/Np*(moments.heat[HEAT_X]-2*out[MACRO_UX]*moments.stress[STRESS_XX]-2*out[MACRO_UY]*moments.stress[STRESS_XY]-2*out[MACRO_UZ]*moments.stress[STRESS_ZX]+u2*moments.velocity[AXIS_X])-out[MACRO_UX]*1.5*out[MACRO_T]*out[MACRO_RHO];
    out[MACRO_QY] = out[MACRO_RHO]/Np*(moments.heat[HEAT_Y]-2*out[MACRO_UX]*moments.stress[STRESS_XY]-2*out[MACRO_UY]*moments.stress[STRESS_YY]-2*out[MACRO_UZ]*moments.stress[STRESS_YZ]+u2*moments.velocity[AXIS_Y])-out[MACRO_UY]*1.5*out[MACRO_T]*out[MACRO_RHO];
    out[MACRO_QZ] = out[MACRO_RHO]/Np*(moments.heat[HEAT_Z]-2*out[MACRO_UX]*moments.stress[STRESS_ZX]-2*out[MACRO_UY]*moments.stress[STRESS_YZ]-2*out[MACRO_UZ]*moments.stress[STRESS_ZZ]+u2*moments.velocity[AXIS_Z])-out[MACRO_UZ]*1.5*out[MACRO_T]*out[MACRO_RHO];
    out[MACRO_TR] = 4.0/this->mess.dr/Np*moments.rot[ROT_ENERGY];
    out[MACRO_QRX] = 2.0*out[MACRO_RHO]/Np*(moments.rot[ROT_HEAT_X]-out[MACRO_UX]*moments.rot[ROT_ENERGY]);
    out[MACRO_QRY] = 2.0*out[MACRO_RHO]/Np*(moments.rot[ROT_HEAT_Y]-out[MACRO_UY]*moments.rot[ROT_ENERGY]);
    out[MACRO_QRZ] = 2.0*out[MACRO_RHO]/Np*(moments.rot[ROT_HEAT_Z]-out[MACRO_UZ]*moments.rot[ROT_ENERGY]);
}

void ProcessDSMC::statistic_macroPre()
{
    if (!this->mpi->active()) return;
    current_macro_zero();
    for (int i = 0;i< this->iNcell;i++)
    {
        int meshindex = this->globalOfLocalCell(i);
        if (meshindex < 0) continue;
        ParticleBucketSoA &bucket = currParticles(meshindex);
        MacroMoments moments;
        accumulateCellMoments(i, bucket, moments);
        writeLocalMacro(i, moments);
        writeRecordMacro(i, moments);
    }
}
bool ProcessDSMC::out2dat(int istep)
{
    int ncell = this->mess.Ncell, Madata = this->Madata;
    const int localCount = this->mpi->active() ? this->iNcell : 0;
    vector<int> localGids((std::size_t)localCount, -1);
    vector<double> localValues((std::size_t)localCount * (std::size_t)Madata, 0.0);
    if (this->mpi->active())
    {
        for(int i = 0; i < this->iNcell; i++)
        {
            int meshindex = this->globalOfLocalCell(i);
            if (meshindex < 0) continue;
            localGids[(std::size_t)i] = meshindex;
            const bool sparseCell =
                (i < (int)this->dsmc2ns_sparse_state.size()) &&
                (this->dsmc2ns_sparse_state[(std::size_t)i] ==
                     DSMC2NS_SPARSE_ACCUMULATING ||
                 this->dsmc2ns_sparse_state[(std::size_t)i] ==
                     DSMC2NS_SPARSE_RELEASED);
            for(int j=0; j < Madata; j++)
            {
                if (istep <= NSS || sparseCell)
                {
                    localValues[(std::size_t)i * (std::size_t)Madata + (std::size_t)j] =
                        this->record[i*Madata+j];
                }else
                {
                    localValues[(std::size_t)i * (std::size_t)Madata + (std::size_t)j] =
                        this->final_record[i*Madata+j];
                }
            }
        }
    }
    vector<int> recvCounts;
    if (this->mpi->root())
        recvCounts.assign((std::size_t)this->size, 0);
    if (MPI_Gather(&localCount, 1, MPI_INT,
                   this->mpi->root() ? recvCounts.data() : nullptr, 1, MPI_INT,
                   0, comm) != MPI_SUCCESS)
        return false;
    if (this->mpi->root())
    {
        vector<int> gidDispls((std::size_t)this->size, 0);
        vector<int> valueCounts((std::size_t)this->size, 0);
        vector<int> valueDispls((std::size_t)this->size, 0);
        int totalCells = 0;
        int totalValues = 0;
        for (int r = 0; r < this->size; ++r)
        {
            gidDispls[(std::size_t)r] = totalCells;
            valueDispls[(std::size_t)r] = totalValues;
            valueCounts[(std::size_t)r] = recvCounts[(std::size_t)r] * Madata;
            totalCells += recvCounts[(std::size_t)r];
            totalValues += valueCounts[(std::size_t)r];
        }
        vector<int> allGids((std::size_t)totalCells, -1);
        vector<double> allValues((std::size_t)totalValues, 0.0);
        if (MPI_Gatherv(localGids.empty() ? nullptr : localGids.data(), localCount, MPI_INT,
                        allGids.empty() ? nullptr : allGids.data(),
                        recvCounts.empty() ? nullptr : recvCounts.data(),
                        gidDispls.empty() ? nullptr : gidDispls.data(),
                        MPI_INT, 0, comm) != MPI_SUCCESS)
            return false;
        const int localValueCount = localCount * Madata;
        if (MPI_Gatherv(localValues.empty() ? nullptr : localValues.data(), localValueCount, MPI_DOUBLE,
                        allValues.empty() ? nullptr : allValues.data(),
                        valueCounts.empty() ? nullptr : valueCounts.data(),
                        valueDispls.empty() ? nullptr : valueDispls.data(),
                        MPI_DOUBLE, 0, comm) != MPI_SUCCESS)
            return false;
        this->out2dat_buffer.assign((std::size_t)ncell * (std::size_t)Madata, 0.0);
        double *w = this->out2dat_buffer.data();
        for (int i = 0; i < totalCells; ++i)
        {
            const int gid = allGids[(std::size_t)i];
            if (gid < 0 || gid >= ncell) continue;
            for (int j = 0; j < Madata; ++j)
            {
                w[(std::size_t)gid * (std::size_t)Madata + (std::size_t)j] =
                    allValues[(std::size_t)i * (std::size_t)Madata + (std::size_t)j];
            }
        }
        char filename[100];
        sprintf(filename, "./statisticResults/DSMCKn%.1f_iter%d_CELL%d.dat", this->mess.Kn, istep, ncell);
        if (this->mesh == NULL)
            return false;
        return this->mesh->out2Fluent_dsmc(filename, w);
    }
    else
    {
        const int localValueCount = localCount * Madata;
        if (MPI_Gatherv(localGids.empty() ? nullptr : localGids.data(), localCount, MPI_INT,
                        nullptr, nullptr, nullptr, MPI_INT, 0, comm) != MPI_SUCCESS)
            return false;
        if (MPI_Gatherv(localValues.empty() ? nullptr : localValues.data(), localValueCount, MPI_DOUBLE,
                        nullptr, nullptr, nullptr, MPI_DOUBLE, 0, comm) != MPI_SUCCESS)
            return false;
        vector<double>().swap(this->out2dat_buffer);
    }
    return true;
}

bool ProcessDSMC::outBoundaryStressHeat(int istep)
{
    if (!EnableBoundaryStressHeatStatistic) return true;
    const int nface = this->mess.Nface;
    const int rawWidth = 6;
    if (nface <= 0) return false;
    std::vector<double> rawData((std::size_t)nface * (std::size_t)rawWidth, 0.0);
    if (this->mpi->active())
    {
        for (const auto& item : this->boundarySteadyTally)
        {
            const int face = item.first;
            if (face < 0 || face >= nface) continue;
            const BoundarySurfaceTally& tally = item.second;
            if (tally.hits == 0.0 &&
                tally.momentumDelta[0] == 0.0 &&
                tally.momentumDelta[1] == 0.0 &&
                tally.momentumDelta[2] == 0.0 &&
                tally.energyTransDelta == 0.0 &&
                tally.energyRotDelta == 0.0)
                continue;
            double* out = &rawData[(std::size_t)face * (std::size_t)rawWidth];
            out[0] = tally.hits;
            out[1] = tally.momentumDelta[AXIS_X];
            out[2] = tally.momentumDelta[AXIS_Y];
            out[3] = tally.momentumDelta[AXIS_Z];
            out[4] = tally.energyTransDelta;
            out[5] = tally.energyRotDelta;
        }
    }
    if (this->mpi->root())
    {
        MPI_Reduce(MPI_IN_PLACE, rawData.data(), nface * rawWidth, MPI_DOUBLE, MPI_SUM, 0, comm);
    }
    else
    {
        MPI_Reduce(rawData.data(), nullptr, nface * rawWidth, MPI_DOUBLE, MPI_SUM, 0, comm);
    }
    if (!this->mpi->root()) return true;
    if (this->mesh == nullptr ||
        (int)this->mesh->Dsmcedges.size() < nface ||
        this->mesh->zonemap.empty())
        return false;
    const bool useSteady = (istep > NSS);
    const double sampleSteps = useSteady ? static_cast<double>(istep - NSS) : 1.0;
    const double timeScale = (this->mess.dtime > 0.0 && sampleSteps > 0.0)
        ? this->mess.Neff / (this->mess.dtime * sampleSteps)
        : 0.0;
    struct WallFaceOutput
    {
        double stress[3] = {0.0, 0.0, 0.0};
        double normal[3] = {0.0, 0.0, 0.0};
        double pressure = 0.0;
        double shear[3] = {0.0, 0.0, 0.0};
        double qTrans = 0.0;
        double qRot = 0.0;
        double qTotal = 0.0;
    };
    std::vector<WallFaceOutput> faceOutput((std::size_t)nface);
    for (int face = 0; face < nface; ++face)
    {
        const DsmcEdge& edge = this->mesh->Dsmcedges[(std::size_t)face];
        if (this->boundaryTable.byTag(edge.faceTag).dsmcRole != BoundaryRole::Wall) continue;
        double normalOut[3] = {0.0, 0.0, 0.0};
        normalOut[AXIS_X] = edge.edgeNormal[AXIS_X];
        normalOut[AXIS_Y] = edge.edgeNormal[AXIS_Y];
        normalOut[AXIS_Z] = edge.edgeNormal[AXIS_Z];
        const double normalLen = std::sqrt(normalOut[AXIS_X] * normalOut[AXIS_X] +
                                           normalOut[AXIS_Y] * normalOut[AXIS_Y] +
                                           normalOut[AXIS_Z] * normalOut[AXIS_Z]);
        if (normalLen > 0.0 && std::isfinite(normalLen))
            normalize(normalOut);
        else
        {
            normalOut[AXIS_X] = 1.0;
            normalOut[AXIS_Y] = 0.0;
            normalOut[AXIS_Z] = 0.0;
        }
        double normalIn[3] = {-normalOut[AXIS_X], -normalOut[AXIS_Y], -normalOut[AXIS_Z]};
        normalize(normalIn);
        const double* in = &rawData[(std::size_t)face * (std::size_t)rawWidth];
        const double pnorm = in[1] * normalIn[AXIS_X] +
                             in[2] * normalIn[AXIS_Y] +
                             in[3] * normalIn[AXIS_Z];
        const double ptang[3] = {
            in[1] - pnorm * normalIn[AXIS_X],
            in[2] - pnorm * normalIn[AXIS_Y],
            in[3] - pnorm * normalIn[AXIS_Z]
        };
        const double area = (edge.length > 0.0 && std::isfinite(edge.length)) ? edge.length : 1.0;
        const double scale = (area > 0.0 && this->mess.n_ref > 0.0)
            ? 2.0 * timeScale / area / this->mess.n_ref
            : 0.0;
        WallFaceOutput& out = faceOutput[(std::size_t)face];
        out.normal[AXIS_X] = normalOut[AXIS_X];
        out.normal[AXIS_Y] = normalOut[AXIS_Y];
        out.normal[AXIS_Z] = normalOut[AXIS_Z];
        out.pressure = pnorm * scale;
        out.shear[AXIS_X] = -ptang[AXIS_X] * scale;
        out.shear[AXIS_Y] = -ptang[AXIS_Y] * scale;
        out.shear[AXIS_Z] = -ptang[AXIS_Z] * scale;
        out.stress[AXIS_X] = out.pressure * out.normal[AXIS_X] + out.shear[AXIS_X];
        out.stress[AXIS_Y] = out.pressure * out.normal[AXIS_Y] + out.shear[AXIS_Y];
        out.stress[AXIS_Z] = out.pressure * out.normal[AXIS_Z] + out.shear[AXIS_Z];
        out.qTrans = -in[4] * scale;
        out.qRot = -in[5] * scale;
        out.qTotal = out.qTrans + out.qRot;
    }
    struct WallZoneBlock
    {
        int zoneId = -1;
        int firstFace = 0;
        int lastFace = -1;
    };
    std::vector<WallZoneBlock> wallZones;
    wallZones.reserve(this->mesh->zonemap.size());
    for (const ZONE& zone : this->mesh->zonemap)
    {
        const int firstFace = std::max(1, zone.firstidx);
        const int lastFace = std::min(zone.lastidx, nface);
        if (firstFace > lastFace) continue;
        bool allWallFaces = true;
        for (int face = firstFace - 1; face < lastFace; ++face)
        {
            const DsmcEdge& edge = this->mesh->Dsmcedges[(std::size_t)face];
            if (this->boundaryTable.byTag(edge.faceTag).dsmcRole != BoundaryRole::Wall)
            {
                allWallFaces = false;
                break;
            }
        }
        if (!allWallFaces) continue;
        WallZoneBlock block;
        block.zoneId = zone.id;
        block.firstFace = firstFace;
        block.lastFace = lastFace;
        wallZones.push_back(block);
    }
    if (wallZones.empty()) return true;
    char filename[200];
    std::snprintf(filename, sizeof(filename),
                  "./statisticResults/BoundaryStressHeatFluentKn%.1f_iter%d.dat",
                  this->mess.Kn, istep);
    std::ofstream fout(filename);
    if (!fout.is_open()) return false;
    fout << "(0 \" (300 (var-id zone-id var-size 0 0 first-id last-id)(......))\" )" << endl;
    fout << "(0 \"DSMC normalized wall fields: 700 stress vector, 701 q_trans, 702 q_rot, 703 normal vector, 704 pressure, 705 shear vector, 706 q_total\")" << endl;
    fout << std::setprecision(16);
    for (const WallZoneBlock& zone : wallZones)
    {
        auto writeScalarBlock = [&](int variableId, double WallFaceOutput::*field)
        {
            fout << "(300 (" << variableId << " " << zone.zoneId << " " << 1
                 << " 0 0 " << zone.firstFace << " " << zone.lastFace << ")" << endl;
            fout << "(" << endl;
            for (int face = zone.firstFace - 1; face < zone.lastFace; ++face)
                fout << faceOutput[(std::size_t)face].*field << endl;
            fout << "))" << endl;
        };
        auto writeVectorBlock = [&](int variableId, double (WallFaceOutput::*field)[3])
        {
            fout << "(300 (" << variableId << " " << zone.zoneId << " " << 3
                 << " 0 0 " << zone.firstFace << " " << zone.lastFace << ")" << endl;
            fout << "(" << endl;
            for (int face = zone.firstFace - 1; face < zone.lastFace; ++face)
            {
                const double* value = faceOutput[(std::size_t)face].*field;
                fout << value[AXIS_X] << " " << value[AXIS_Y] << " "
                     << value[AXIS_Z] << endl;
            }
            fout << "))" << endl;
        };
        writeVectorBlock(700, &WallFaceOutput::stress);
        writeScalarBlock(701, &WallFaceOutput::qTrans);
        writeScalarBlock(702, &WallFaceOutput::qRot);
        writeVectorBlock(703, &WallFaceOutput::normal);
        writeScalarBlock(704, &WallFaceOutput::pressure);
        writeVectorBlock(705, &WallFaceOutput::shear);
        writeScalarBlock(706, &WallFaceOutput::qTotal);
    }
    return true;
}
