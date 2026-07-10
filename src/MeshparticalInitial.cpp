/*
 * Particle count adjustment, cell sampling, and initial particle creation.
 */

# include "MeshparticalInitial.h"
# include "meshImport.h"
# include "math.h"
#include <chrono>
#include <algorithm>
#include <cmath>
#include <random>
using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace
{
    constexpr int kMigrationHaloRings = 3;
    struct InitTet
    {
        int node[4] = {-1, -1, -1, -1};
        double volume = 0.0;
    };
}

struct ParticleInitialLocalOps
{
/*
 * validLocalNodeIndex: works with mesh topology or geometric intersections.
 * Params: idx, xyz; returns: success or decision flag.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 */
    static bool validLocalNodeIndex(int idx, const vector<double>& xyz)
    {
        return idx >= 0 && (size_t)idx < xyz.size() / 3u;
    }
/*
 * tetAbsVolume: works with mesh topology or geometric intersections.
 * Params: xyz, n0, n1, n2, n3; returns: computed scalar.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 */
    static double tetAbsVolume(const vector<double>& xyz,
                               int n0, int n1, int n2, int n3)
    {
        const double ax = xyz[3 * n1 + 0] - xyz[3 * n0 + 0];
        const double ay = xyz[3 * n1 + 1] - xyz[3 * n0 + 1];
        const double az = xyz[3 * n1 + 2] - xyz[3 * n0 + 2];
        const double bx = xyz[3 * n2 + 0] - xyz[3 * n0 + 0];
        const double by = xyz[3 * n2 + 1] - xyz[3 * n0 + 1];
        const double bz = xyz[3 * n2 + 2] - xyz[3 * n0 + 2];
        const double cx = xyz[3 * n3 + 0] - xyz[3 * n0 + 0];
        const double cy = xyz[3 * n3 + 1] - xyz[3 * n0 + 1];
        const double cz = xyz[3 * n3 + 2] - xyz[3 * n0 + 2];
        const double crx = by * cz - bz * cy;
        const double cry = bz * cx - bx * cz;
        const double crz = bx * cy - by * cx;
        return std::fabs(ax * crx + ay * cry + az * crz) / 6.0;
    }
/*
 * positiveUniform: performs one solver support operation.
 * Params: dis, gen; returns: computed scalar.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 */
    static double positiveUniform(std::uniform_real_distribution<double>& dis,
                                  std::mt19937& gen)
    {
        const double r = dis(gen);
        return (r > 0.0) ? r : 1.0e-300;
    }
/*
 * randomBarycentric4: works with mesh topology or geometric intersections.
 * Params: dis, gen, a, b, c, d; returns: none.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 */
    static void randomBarycentric4(std::uniform_real_distribution<double>& dis,
                                   std::mt19937& gen,
                                   double& a,
                                   double& b,
                                   double& c,
                                   double& d)
    {
        const double r0 = positiveUniform(dis, gen);
        const double r1 = positiveUniform(dis, gen);
        const double r2 = positiveUniform(dis, gen);
        const double r3 = positiveUniform(dis, gen);
        const double sum = r0 + r1 + r2 + r3;
        if (!(sum > 0.0) || !std::isfinite(sum))
        {
            a = b = c = d = 0.25;
            return;
        }
        a = r0 / sum;
        b = r1 / sum;
        c = r2 / sum;
        d = r3 / sum;
    }
};

/*
 * appendVertexTet: works with mesh topology or geometric intersections.
 * Params: init, n0, n1, n2, n3, tets; returns: success or decision flag.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 */
static bool appendVertexTet(const MeshparticalInitial& init,
                            int n0, int n1, int n2, int n3,
                            vector<InitTet>& tets)
{
    const int nodes[4] = {n0, n1, n2, n3};
    for (int k = 0; k < 4; ++k)
    {
        if (!ParticleInitialLocalOps::validLocalNodeIndex(nodes[k], init.localPointXY))
            return false;
    }
    InitTet tet;
    for (int k = 0; k < 4; ++k)
        tet.node[k] = nodes[k];
    tet.volume = ParticleInitialLocalOps::tetAbsVolume(init.localPointXY, n0, n1, n2, n3);
    if (!(tet.volume > 0.0) || !std::isfinite(tet.volume)) return false;
    tets.push_back(tet);
    return true;
}

/*
 * collectCellUniqueNodes: works with mesh topology or geometric intersections.
 * Params: init, cell, uniqueNodes; returns: success or decision flag.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 */
static bool collectCellUniqueNodes(const MeshparticalInitial& init,
                                   const DsmcCell& cell,
                                   vector<int>& uniqueNodes)
{
    uniqueNodes.clear();
    for (int m = 0; m < cell.num && m < NN; ++m)
    {
        const int faceLocal = cell.cell2face[m];
        if (faceLocal < 0 || faceLocal >= (int)init.edges.size()) return false;
        const DsmcEdge& face = init.edges[(std::size_t)faceLocal];
        if (face.faceType != 3 && face.faceType != 4) return false;
        for (int k = 0; k < face.faceType; ++k)
        {
            const int node = face.faceMap[k];
            if (!ParticleInitialLocalOps::validLocalNodeIndex(node, init.localPointXY))
                return false;
            if (std::find(uniqueNodes.begin(), uniqueNodes.end(), node) == uniqueNodes.end())
                uniqueNodes.push_back(node);
        }
    }
    return true;
}

/*
 * appendTetCellTets: works with mesh topology or geometric intersections.
 * Params: init, cell, tets; returns: success or decision flag.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 */
static bool appendTetCellTets(const MeshparticalInitial& init,
                              const DsmcCell& cell,
                              vector<InitTet>& tets)
{
    vector<int> uniqueNodes;
    if (!collectCellUniqueNodes(init, cell, uniqueNodes)) return false;
    if (uniqueNodes.size() != 4u) return false;
    return appendVertexTet(init,
                           uniqueNodes[0], uniqueNodes[1],
                           uniqueNodes[2], uniqueNodes[3],
                           tets);
}

/*
 * appendPyramidCellTets: works with mesh topology or geometric intersections.
 * Params: init, cell, tets; returns: success or decision flag.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 */
static bool appendPyramidCellTets(const MeshparticalInitial& init,
                                  const DsmcCell& cell,
                                  vector<InitTet>& tets)
{
    int quadFace = -1;
    for (int m = 0; m < cell.num && m < NN; ++m)
    {
        const int faceLocal = cell.cell2face[m];
        if (faceLocal < 0 || faceLocal >= (int)init.edges.size()) return false;
        const DsmcEdge& face = init.edges[(std::size_t)faceLocal];
        if (face.faceType == 4)
        {
            if (quadFace >= 0) return false;
            quadFace = faceLocal;
        }
    }
    if (quadFace < 0) return false;
    vector<int> uniqueNodes;
    if (!collectCellUniqueNodes(init, cell, uniqueNodes)) return false;
    if (uniqueNodes.size() != 5u) return false;
    const DsmcEdge& base = init.edges[(std::size_t)quadFace];
    int apex = -1;
    for (int node : uniqueNodes)
    {
        bool onBase = false;
        for (int k = 0; k < 4; ++k)
        {
            if (base.faceMap[k] == node)
            {
                onBase = true;
                break;
            }
        }
        if (!onBase)
        {
            apex = node;
            break;
        }
    }
    if (apex < 0) return false;
    unsigned char tag = (quadFace >= 0 && quadFace < (int)init.faceSplitTag.size())
        ? init.faceSplitTag[(std::size_t)quadFace]
        : meshImport::FACE_SPLIT_02;
    int tri0[3], tri1[3];
    meshImport::decode_quad_split_tag(tag, tri0, tri1);
    if (tri0[0] < 0 || tri1[0] < 0)
        meshImport::decode_quad_split_tag(meshImport::FACE_SPLIT_02, tri0, tri1);
    if (tri0[0] < 0 || tri1[0] < 0) return false;
    return appendVertexTet(init,
                           base.faceMap[tri0[0]],
                           base.faceMap[tri0[1]],
                           base.faceMap[tri0[2]],
                           apex,
                           tets) &&
           appendVertexTet(init,
                           base.faceMap[tri1[0]],
                           base.faceMap[tri1[1]],
                           base.faceMap[tri1[2]],
                           apex,
                           tets);
}

/*
 * appendCenterFaceTet: works with mesh topology or geometric intersections.
 * Params: init, cellLocal, faceLocal, tri, tets; returns: success or decision flag.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 */
static bool appendCenterFaceTet(const MeshparticalInitial& init,
                                int cellLocal,
                                int faceLocal,
                                const int tri[3],
                                vector<InitTet>& tets)
{
    if (faceLocal < 0 || faceLocal >= (int)init.edges.size()) return false;
    const DsmcEdge& face = init.edges[(std::size_t)faceLocal];
    const double* center = init.cells[(std::size_t)cellLocal].cellXY;
    InitTet tet;
    for (int k = 0; k < 3; ++k)
    {
        const int nodeSlot = tri[k];
        if (nodeSlot < 0 || nodeSlot >= face.faceType) return false;
        tet.node[k] = face.faceMap[nodeSlot];
        if (!ParticleInitialLocalOps::validLocalNodeIndex(tet.node[k], init.localPointXY))
            return false;
    }
    tet.node[3] = -1;
    const double ax = init.localPointXY[3 * tet.node[0] + 0] - center[0];
    const double ay = init.localPointXY[3 * tet.node[0] + 1] - center[1];
    const double az = init.localPointXY[3 * tet.node[0] + 2] - center[2];
    const double bx = init.localPointXY[3 * tet.node[1] + 0] - center[0];
    const double by = init.localPointXY[3 * tet.node[1] + 1] - center[1];
    const double bz = init.localPointXY[3 * tet.node[1] + 2] - center[2];
    const double cx = init.localPointXY[3 * tet.node[2] + 0] - center[0];
    const double cy = init.localPointXY[3 * tet.node[2] + 1] - center[1];
    const double cz = init.localPointXY[3 * tet.node[2] + 2] - center[2];
    const double crx = by * cz - bz * cy;
    const double cry = bz * cx - bx * cz;
    const double crz = bx * cy - by * cx;
    tet.volume = std::fabs(ax * crx + ay * cry + az * crz) / 6.0;
    if (!(tet.volume > 0.0) || !std::isfinite(tet.volume)) return false;
    tets.push_back(tet);
    return true;
}

/*
 * appendCenterFaceTets: works with mesh topology or geometric intersections.
 * Params: init, cellLocal, cell, tets; returns: success or decision flag.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 */
static bool appendCenterFaceTets(const MeshparticalInitial& init,
                                 int cellLocal,
                                 const DsmcCell& cell,
                                 vector<InitTet>& tets)
{
    for (int m = 0; m < cell.num && m < NN; ++m)
    {
        const int faceLocal = cell.cell2face[m];
        if (faceLocal < 0 || faceLocal >= (int)init.edges.size()) return false;
        const DsmcEdge& face = init.edges[(std::size_t)faceLocal];
        if (face.faceType == 3)
        {
            const int tri[3] = {0, 1, 2};
            if (!appendCenterFaceTet(init, cellLocal, faceLocal, tri, tets)) return false;
        }
        else if (face.faceType == 4)
        {
            unsigned char tag = (faceLocal >= 0 && faceLocal < (int)init.faceSplitTag.size())
                ? init.faceSplitTag[(std::size_t)faceLocal]
                : meshImport::FACE_SPLIT_02;
            int tri0[3], tri1[3];
            meshImport::decode_quad_split_tag(tag, tri0, tri1);
            if (tri0[0] < 0 || tri1[0] < 0)
                meshImport::decode_quad_split_tag(meshImport::FACE_SPLIT_02, tri0, tri1);
            if (!appendCenterFaceTet(init, cellLocal, faceLocal, tri0, tets)) return false;
            if (!appendCenterFaceTet(init, cellLocal, faceLocal, tri1, tets)) return false;
        }
        else
        {
            return false;
        }
    }
    return !tets.empty();
}

/*
 * buildCellInitTets: works with mesh topology or geometric intersections.
 * Params: init, cellLocal, tets; returns: success or decision flag.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 */
static bool buildCellInitTets(const MeshparticalInitial& init,
                              int cellLocal,
                              vector<InitTet>& tets)
{
    tets.clear();
    if (cellLocal < 0 || cellLocal >= (int)init.cells.size()) return false;
    const DsmcCell& cell = init.cells[(std::size_t)cellLocal];
    if (appendTetCellTets(init, cell, tets))
        return true;
    tets.clear();
    if (appendPyramidCellTets(init, cell, tets))
        return true;
    tets.clear();
    return appendCenterFaceTets(init, cellLocal, cell, tets);
}

/*
 * MeshparticalInitial: prepares derived solver state.
 * Params: none; returns: MeshparticalInitial ::.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 */
MeshparticalInitial :: MeshparticalInitial()
{    
}

/*
 * ~MeshparticalInitial: releases owned buffers and MPI helper state.
 * Params: none; returns: none.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 */
MeshparticalInitial :: ~MeshparticalInitial()
{
}

/*
 * MeshparticalInitial: prepares derived solver state.
 * Params: mesh, mess, mpass, mpiCtx; returns: MeshparticalInitial ::.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 */
MeshparticalInitial :: MeshparticalInitial(meshImport *mesh, meshMessage mess, MessagePassing *mpass, const MpiContext& mpiCtx)
{
    this->mpi = &mpiCtx;
    this->mesh = mesh;
    this->mess = mess;
    this->mpass = mpass;
    this->comm = mpiCtx.comm;
    this->calGroup = mpiCtx.calGroup;
    this->rank = mpiCtx.rank;
    this->size = mpiCtx.size;
    this->c_rank = mpiCtx.c_rank;
    this->c_size = mpiCtx.c_size;
    const bool active = this->mpi != nullptr && this->mpi->active();
    module_variables();
    MeshPartitionTransfer3D transfer(this->mesh, this, this->mpass, *this->mpi);
    if (!transfer.initialPartitionAndDistribute(kMigrationHaloRings))
    {
        cout << "MESH_PARTITION_INITIAL_FAIL"
             << " rank=" << this->rank << endl;
        return;
    }
    this->allocateOwnedParticleBuckets();
    vector<int> Np_exp;
    if (!transfer.broadcastInitialParticleCounts(Np_exp))
    {
        cout << "MESH_PARTITION_INITIAL_NPC_BCAST_FAIL"
             << " rank=" << this->rank << endl;
        return;
    }
    this->partitionReady = true;
    if(active)
    {
        Collision_constant_initial();
        for(int i = 0; i < this->my_owned_ncell; i++)
        {
            int Npc_exa = Np_exp[this->local_cells[i]];
            vector<double> Np_tri;
            if(Npc_exa > 0) { Np_tri_inititial(i, Np_tri, Npc_exa); }
            if(Npc_exa > 0 && !Np_tri.empty()) { meshfillpar(i, Np_tri); }
        }
    }
}

/*
 * module_variables: performs one solver support operation.
 * Params: none; returns: void MeshparticalInitial.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 */
void MeshparticalInitial ::module_variables()
{
    this->rank_cell_all.reserve(this->mess.Ncell);
    this->rank_cell_all.resize(this->mess.Ncell);
    this->cell_particles_curr.clear();
    this->cell_particles_next.clear();
    this->cell_particle_reserve_hint.assign((std::size_t)this->mess.Ncell, 0);
}

/*
 * allocateOwnedParticleBuckets: updates particles or particle-derived state.
 * Params: none; returns: none.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 */
void MeshparticalInitial::allocateOwnedParticleBuckets()
{
    const int nOwned = std::max(0, this->my_owned_ncell);
    this->cell_particles_curr.clear();
    this->cell_particles_curr.resize((std::size_t)nOwned);
    for (int lc = 0; lc < nOwned; ++lc)
    {
        if (lc >= (int)this->local_cells.size()) continue;
        const int gid = this->local_cells[(std::size_t)lc];
        if (gid < 0 || gid >= (int)this->cell_particle_reserve_hint.size()) continue;
        const int hint = this->cell_particle_reserve_hint[(std::size_t)gid];
        if (hint > 0)
            this->cell_particles_curr[(std::size_t)lc].reserve((std::size_t)hint);
    }
}

/*
 * Np_tri_inititial: works with mesh topology or geometric intersections.
 * Params: i, Np_tri, Np_exa; returns: none.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 */
void MeshparticalInitial::Np_tri_inititial(int i, std::vector<double>& Np_tri, int Np_exa)
{
    vector<InitTet> initTets;
    if (!buildCellInitTets(*this, i, initTets))
    {
        cout << "[INIT_TET_BUILD_ERROR] cellLocal=" << i
                  << " cellGlobal=" << this->local_cells[i]
                  << endl;
        return;
    }
    double sumSubVol = 0.0;
    for (const InitTet& tet : initTets)
        sumSubVol += tet.volume;
    if (!(sumSubVol > 0.0) || !std::isfinite(sumSubVol))
    {
        cout << "[INIT_TET_VOLUME_ERROR] cellLocal=" << i
                  << " cellGlobal=" << this->local_cells[i]
                  << " sumVol=" << sumSubVol << endl;
        return;
    }
    Np_tri.clear();
    Np_tri.reserve(initTets.size());
    for (const InitTet& tet : initTets)
    {
        double tetNp = static_cast<double>(Np_exa) * tet.volume / sumSubVol;
        tetNp = Iround(tetNp);
        Np_tri.push_back(tetNp);
    }
    int sumN = 0;
    for (double n : Np_tri)
        sumN += static_cast<int>(n);
    const int selectN = Np_exa - sumN;
    adjust_particles(Np_tri, (int)Np_tri.size(), selectN);
}

/*
 * CreateParticlesnode: works with mesh topology or geometric intersections.
 * Params: j, i, index0, index1, index2, center; returns: void MeshparticalInitial ::.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 */
void MeshparticalInitial :: CreateParticlesnode(int j,int i, int index0, int index1, int index2, const double center[DIM])
{
    particle parts = this->velocityLocationGenerate(j, index0, index1, index2, center);
    if (isinf(parts.p_velocity[0]) || isinf(parts.p_velocity[1]) || isinf(parts.p_velocity[2]) )
    {
        cout << "The initialization procedure failed" << endl;
    }
    const int localCell = i;
    const int gid = (localCell >= 0 && localCell < (int)this->local_cells.size())
        ? this->local_cells[(std::size_t)localCell]
        : -1;
    if (localCell >= 0 && localCell < (int)this->cell_particles_curr.size())
    {
        ParticleBucketSoA& bucket = this->cell_particles_curr[(std::size_t)localCell];
        parts.p_serial = (int)bucket.size();
        parts.p_rank_serial = this->c_rank;
        parts.p_mesh_serial = localCell;
        parts.dt_left = 0.0;
        bucket.push_back(parts);
        if (gid >= 0 && gid < (int)this->cell_particle_reserve_hint.size())
        {
            this->cell_particle_reserve_hint[(std::size_t)gid] =
                std::max(this->cell_particle_reserve_hint[(std::size_t)gid],
                         (int)bucket.size());
        }
    }
}

/*
 * CreateParticlesnode: works with mesh topology or geometric intersections.
 * Params: j, i, index0, index1, index2, index3; returns: void MeshparticalInitial ::.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 */
void MeshparticalInitial :: CreateParticlesnode(int j,int i, int index0, int index1, int index2, int index3)
{
    particle parts = this->velocityLocationGenerate(j, index0, index1, index2, index3);
    if (isinf(parts.p_velocity[0]) || isinf(parts.p_velocity[1]) || isinf(parts.p_velocity[2]) )
    {
        cout << "The initialization procedure failed" << endl;
    }
    const int localCell = i;
    const int gid = (localCell >= 0 && localCell < (int)this->local_cells.size())
        ? this->local_cells[(std::size_t)localCell]
        : -1;
    if (localCell >= 0 && localCell < (int)this->cell_particles_curr.size())
    {
        ParticleBucketSoA& bucket = this->cell_particles_curr[(std::size_t)localCell];
        parts.p_serial = (int)bucket.size();
        parts.p_rank_serial = this->c_rank;
        parts.p_mesh_serial = localCell;
        parts.dt_left = 0.0;
        bucket.push_back(parts);
        if (gid >= 0 && gid < (int)this->cell_particle_reserve_hint.size())
        {
            this->cell_particle_reserve_hint[(std::size_t)gid] =
                std::max(this->cell_particle_reserve_hint[(std::size_t)gid],
                         (int)bucket.size());
        }
    }
}

/*
 * meshfillpar: performs one solver support operation.
 * Params: i, Np_tri; returns: void MeshparticalInitial ::.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 */
void MeshparticalInitial :: meshfillpar(int i,vector<double>& Np_tri)
{
    vector<InitTet> initTets;
    if (!buildCellInitTets(*this, i, initTets))
    {
        cout << "[INIT_TET_BUILD_ERROR] meshfill cellLocal=" << i
             << " cellGlobal=" << this->local_cells[i]
             << endl;
        return;
    }
    if (Np_tri.size() != initTets.size())
    {
        cout << "[INIT_TET_COUNT_ERROR] cellLocal=" << i
             << " cellGlobal=" << this->local_cells[i]
             << " expected=" << initTets.size()
             << " actual=" << Np_tri.size() << endl;
        return;
    }
    int partOffset = 0;
    const double* center = this->cells[(std::size_t)i].cellXY;
    for (int it = 0; it < (int)initTets.size(); ++it)
    {
        const int npTet = static_cast<int>(Np_tri[it]);
        for (int j = 0; j < npTet; ++j)
        {
            if (initTets[(std::size_t)it].node[3] >= 0)
            {
                CreateParticlesnode(j + partOffset, i,
                                    initTets[(std::size_t)it].node[0],
                                    initTets[(std::size_t)it].node[1],
                                    initTets[(std::size_t)it].node[2],
                                    initTets[(std::size_t)it].node[3]);
            }
            else
            {
                CreateParticlesnode(j + partOffset, i,
                                    initTets[(std::size_t)it].node[0],
                                    initTets[(std::size_t)it].node[1],
                                    initTets[(std::size_t)it].node[2],
                                    center);
            }
        }
        partOffset += npTet;
    }
}

/*
 * velocityLocationGenerate: updates particles or particle-derived state.
 * Params: i, index0, index1, index2, center; returns: updated particle.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 */
particle MeshparticalInitial :: velocityLocationGenerate(int i, int index0, int index1, int index2, const double center[DIM])
{
    particle ipart;
    auto& gen = thread_rng();
    auto& dis = get_uniform();
    double r = ParticleInitialLocalOps::positiveUniform(dis, gen);
    double theta = 2 * M_PI * dis(gen);
    ipart.p_velocity[0] = sqrt(-2 * log(r)) * sin(theta);
    ipart.p_velocity[1] = sqrt(-2 * log(r)) * cos(theta);
    r = ParticleInitialLocalOps::positiveUniform(dis, gen);
    theta = 2 * M_PI * dis(gen);
    ipart.p_velocity[2] = sqrt(-2 * log(r)) * sin(theta);
    r = ParticleInitialLocalOps::positiveUniform(dis, gen);
    ipart.p_Ir = -0.5 * log(r);
    double u, v, w, t;
    ParticleInitialLocalOps::randomBarycentric4(dis, gen, u, v, w, t);
    ipart.p_location[0] =
        u * center[0] +
        v * this->localPointXY[index0 * 3 + 0] +
        w * this->localPointXY[index1 * 3 + 0] +
        t * this->localPointXY[index2 * 3 + 0];
    ipart.p_location[1] =
        u * center[1] +
        v * this->localPointXY[index0 * 3 + 1] +
        w * this->localPointXY[index1 * 3 + 1] +
        t * this->localPointXY[index2 * 3 + 1];
    ipart.p_location[2] =
        u * center[2] +
        v * this->localPointXY[index0 * 3 + 2] +
        w * this->localPointXY[index1 * 3 + 2] +
        t * this->localPointXY[index2 * 3 + 2];
    ipart.p_serial = i;
    return ipart;
}

/*
 * velocityLocationGenerate: updates particles or particle-derived state.
 * Params: i, index0, index1, index2, index3; returns: updated particle.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 */
particle MeshparticalInitial :: velocityLocationGenerate(int i, int index0, int index1, int index2, int index3)
{
    particle ipart;
    auto& gen = thread_rng();
    auto& dis = get_uniform();
    double r = ParticleInitialLocalOps::positiveUniform(dis, gen);
    double theta = 2 * M_PI * dis(gen);
    ipart.p_velocity[0] = sqrt(-2 * log(r)) * sin(theta);
    ipart.p_velocity[1] = sqrt(-2 * log(r)) * cos(theta);
    r = ParticleInitialLocalOps::positiveUniform(dis, gen);
    theta = 2 * M_PI * dis(gen);
    ipart.p_velocity[2] = sqrt(-2 * log(r)) * sin(theta);
    r = ParticleInitialLocalOps::positiveUniform(dis, gen);
    ipart.p_Ir = -0.5 * log(r);
    double u, v, w, t;
    ParticleInitialLocalOps::randomBarycentric4(dis, gen, u, v, w, t);
    ipart.p_location[0] =
        u * this->localPointXY[index0 * 3 + 0] +
        v * this->localPointXY[index1 * 3 + 0] +
        w * this->localPointXY[index2 * 3 + 0] +
        t * this->localPointXY[index3 * 3 + 0];
    ipart.p_location[1] =
        u * this->localPointXY[index0 * 3 + 1] +
        v * this->localPointXY[index1 * 3 + 1] +
        w * this->localPointXY[index2 * 3 + 1] +
        t * this->localPointXY[index3 * 3 + 1];
    ipart.p_location[2] =
        u * this->localPointXY[index0 * 3 + 2] +
        v * this->localPointXY[index1 * 3 + 2] +
        w * this->localPointXY[index2 * 3 + 2] +
        t * this->localPointXY[index3 * 3 + 2];
    ipart.p_serial = i;
    return ipart;
}

/*
 * Iround: performs one solver support operation.
 * Params: x; returns: int MeshparticalInitial ::.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 */
int MeshparticalInitial :: Iround(double x)
{
    int x_f = 0;
    double r = static_cast<double>(rand()) / (RAND_MAX) + 0.5 / (RAND_MAX);
    if (r<(x-floor(x)))
    {
        x_f = floor(x) + 1;
    } 
    else
    {
        x_f = floor(x);
    }
    return x_f;
}

/*
 * adjust_particles: updates particles or particle-derived state.
 * Params: regions, size, delta; returns: none.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 */
void MeshparticalInitial::adjust_particles(vector<double>& regions, int size, int delta) {
/*
 * rng: performs one solver support operation.
 * Params: std::random_device{}(); returns: std::mt19937.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 */
    static std::mt19937 rng(std::random_device{}());
    const int abs_delta = std::abs(delta);
    const bool add = delta > 0;
    if(abs_delta > size) { 
        cout<<"delta exceeds region count!"<<endl; 
    }
    auto select_regions = [size](int count) {
        vector<int> indices(size);
        iota(indices.begin(), indices.end(), 0);
        for(int i=0; i<count; ++i) {
            uniform_int_distribution<int> dist(i, size-1);
            int r = dist(rng);
            swap(indices[i], indices[r]);
        }
        return vector<int>(indices.begin(), indices.begin() + count);
    };
    if(abs_delta == size) {
        for(int i=0; i<size; ++i) {
            regions[i] += (add ? 1.0 : -1.0);
        }
    } else {
        const auto selected = select_regions(abs_delta);
        for(int idx : selected) {
            regions[idx] += (add ? 1.0 : -1.0);
        }
    }
    for(int i=0; i<size; ++i) {
        regions[i] = std::max(regions[i], 0.0);
    }
}

/*
 * Collision_constant_initial: updates particles or particle-derived state.
 * Params: none; returns: void MeshparticalInitial ::.
 * Flow:
 *   - select collision pairs.
 *   - apply acceptance model.
 */
void MeshparticalInitial :: Collision_constant_initial()
{
    const int nOwned = this->my_owned_ncell; 
    this->crmax.resize(nOwned);
    this->remainderincoll.resize(nOwned);
    this->remainderinpre.resize(nOwned);
    for (int i = 0;i < nOwned;i++)
    {
        crmax[i] = 3.0;
        remainderincoll[i] = 0.0;
        remainderinpre[i] = 0.0;
    }
}

/*
 * crossProduct: performs one solver support operation.
 * Params: vecA, vecB, result; returns: void MeshparticalInitial ::.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 */
void MeshparticalInitial :: crossProduct(const double* vecA, const double* vecB, double* result) {
    result[0] = vecA[1]*vecB[2] - vecA[2]*vecB[1];
    result[1] = vecA[2]*vecB[0] - vecA[0]*vecB[2];
    result[2] = vecA[0]*vecB[1] - vecA[1]*vecB[0];
}

/*
 * dotProduct: performs one solver support operation.
 * Params: vecA, vecB, size; returns: double MeshparticalInitial ::.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 */
double MeshparticalInitial :: dotProduct(const double* vecA, const double* vecB, int size) {
    double result = 0.0;
    for (int i = 0; i < size; i++) {
        result += vecA[i] * vecB[i];
    }
    return result;
}
