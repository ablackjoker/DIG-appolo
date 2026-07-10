#include "ProcessDSMC.h"

#include <algorithm>
#include <cmath>
#include <utility>

void ProcessDSMC::installLocalGeometry(LocalGeometry3D&& geometry)
{
    this->iNcell = geometry.ownedCount;
    this->nface = (int)geometry.face_gids.size();
    this->cells = std::move(geometry.cells);
    this->edges = std::move(geometry.edges);
    this->local_cells = std::move(geometry.local_cells);
    this->face_gids = std::move(geometry.face_gids);
    this->gid2local = std::move(geometry.gid2local);
    this->face_gid2local = std::move(geometry.face_gid2local);
    this->localPointXY = std::move(geometry.localPointXY);
    this->faceSplitTag = std::move(geometry.faceSplitTag);
    this->dsmc2ns_window_samples.assign((std::size_t)this->iNcell, 0.0);
    this->dsmc2ns_window_valid.assign((std::size_t)this->iNcell, 0);
    this->boundaryEmitCacheDirty = true;
    rebuildFaceLookup();
    rebuildFaceCrossCache();
    rebuildFaceTraversalCache();
    rebuild_migration_peer_cache();
    ensure_partition_flux_storage();
    ensure_partition_time_storage();
}

void ProcessDSMC::rebuild_migration_peer_cache()
{
    this->migrationPeerRanks.clear();
    this->migrationPeerMask.clear();
    if (this->mpi == nullptr || !this->mpi->active() || this->c_size <= 0) return;
    std::vector<char> need((std::size_t)this->c_size, 0);
    const int nlocal = (int)this->local_cells.size();
    for (int lc = 0; lc < nlocal; ++lc)
    {
        const int gid = this->local_cells[(std::size_t)lc];
        const int owner = this->ownerOfGlobalCell(gid);
        if (owner >= 0 && owner < this->c_size && owner != this->c_rank)
            need[(std::size_t)owner] = 1;
    }
    std::vector<char> allNeed((std::size_t)this->c_size * (std::size_t)this->c_size, 0);
    MPI_Allgather(need.data(), this->c_size, MPI_CHAR,
                  allNeed.data(), this->c_size, MPI_CHAR,
                  this->calGroup);
    this->migrationPeerMask.assign((std::size_t)this->c_size, 0);
    for (int r = 0; r < this->c_size; ++r)
    {
        if (r == this->c_rank) continue;
        const bool iNeedR = need[(std::size_t)r] != 0;
        const bool rNeedsMe = allNeed[(std::size_t)r * (std::size_t)this->c_size +
                                      (std::size_t)this->c_rank] != 0;
        if (iNeedR || rNeedsMe)
        {
            this->migrationPeerMask[(std::size_t)r] = 1;
            this->migrationPeerRanks.push_back(r);
        }
    }
}

void ProcessDSMC::rebuildFaceLookup()
{
    const int denseSize = std::max(this->mess.Nface, (int)this->face_gids.size());
    if (denseSize > 0)
        this->faceGid2LocalDense.assign((std::size_t)denseSize, -1);
    else
        this->faceGid2LocalDense.clear();
    this->face_gid2local.clear();
    this->face_gid2local.reserve(this->face_gids.size());
    for (int lf = 0; lf < (int)this->face_gids.size(); ++lf)
    {
        const int gid = this->face_gids[(std::size_t)lf];
        if (gid < 0) continue;
        if (gid < (int)this->faceGid2LocalDense.size())
            this->faceGid2LocalDense[(std::size_t)gid] = lf;
        this->face_gid2local[gid] = lf;
    }
}

void ProcessDSMC::rebuildFaceCrossCache()
{
    this->faceCrossCache.assign(this->edges.size(), FaceCrossCache());
    for (int lf = 0; lf < (int)this->edges.size(); ++lf)
    {
        const DsmcEdge& edge = this->edges[(std::size_t)lf];
        FaceCrossCache& cache = this->faceCrossCache[(std::size_t)lf];
        cache.gface = (lf < (int)this->face_gids.size()) ? this->face_gids[(std::size_t)lf] : lf;
        if (this->boundaryTable.byTag(edge.faceTag).dsmcRole != BoundaryRole::Interface) continue;
        const int g0 = edge.faceMap[4];
        const int g1 = edge.faceMap[5];
        cache.globalCell0 = g0;
        cache.globalCell1 = g1;
        cache.localCell0 = this->localOfGlobalCell(g0);
        cache.localCell1 = this->localOfGlobalCell(g1);
        cache.globalNextFromCell0 = g1;
        cache.globalNextFromCell1 = g0;
        cache.localNextFromCell0 = cache.localCell1;
        cache.localNextFromCell1 = cache.localCell0;
        cache.ownerNextFromCell0 = this->ownerOfGlobalCell(g1);
        cache.ownerNextFromCell1 = this->ownerOfGlobalCell(g0);
    }
}

void ProcessDSMC::rebuildFaceTraversalCache()
{
    this->cellTriOffset.assign(this->cells.size() + 1u, 0);
    this->cellTriCache.clear();
    this->cellTriCache.reserve(this->cells.size() * 8u);
    auto loadPointSafe = [&](int idx, double P[3]) -> bool
    {
        if (idx < 0) return false;
        const std::size_t p = (std::size_t)idx * 3u;
        if (p + 2u >= this->localPointXY.size()) return false;
        P[0] = this->localPointXY[p + 0u];
        P[1] = this->localPointXY[p + 1u];
        P[2] = this->localPointXY[p + 2u];
        return true;
    };
    auto appendTri = [&](int faceid, int triTag, int ia, int ib, int ic) -> bool
    {
        double A[3], B[3], C[3];
        if (!loadPointSafe(ia, A) || !loadPointSafe(ib, B) || !loadPointSafe(ic, C))
            return false;
        FaceTriCache tri;
        tri.face = faceid;
        tri.triTag = triTag;
        tri.ax = A[0];
        tri.ay = A[1];
        tri.az = A[2];
        tri.e1x = B[0] - A[0];
        tri.e1y = B[1] - A[1];
        tri.e1z = B[2] - A[2];
        tri.e2x = C[0] - A[0];
        tri.e2y = C[1] - A[1];
        tri.e2z = C[2] - A[2];
        this->cellTriCache.push_back(tri);
        return true;
    };
    for (int lc = 0; lc < (int)this->cells.size(); ++lc)
    {
        this->cellTriOffset[(std::size_t)lc] = (int)this->cellTriCache.size();
        const int faceCount = this->cells[(std::size_t)lc].num;
        for (int m = 0; m < faceCount && m < NN; ++m)
        {
            const int faceid = this->cells[(std::size_t)lc].cell2face[m];
            if (faceid < 0 || faceid >= (int)this->edges.size()) continue;
            const DsmcEdge& face = this->edges[(std::size_t)faceid];
            const int ids[4] = {
                face.faceMap[0],
                face.faceMap[1],
                face.faceMap[2],
                face.faceMap[3]
            };
            if (face.faceType == 3)
            {
                appendTri(faceid, -1, ids[0], ids[1], ids[2]);
                continue;
            }
            if (face.faceType != 4) continue;
            unsigned char tag = (faceid >= 0 && faceid < (int)this->faceSplitTag.size())
                ? this->faceSplitTag[(std::size_t)faceid]
                : meshImport::FACE_SPLIT_02;
            int tri0[3], tri1[3];
            meshImport::decode_quad_split_tag(tag, tri0, tri1);
            if (tri0[0] < 0 || tri1[0] < 0)
                meshImport::decode_quad_split_tag(meshImport::FACE_SPLIT_02, tri0, tri1);
            if (tri0[0] < 0 || tri1[0] < 0) continue;
            appendTri(faceid, 1, ids[tri0[0]], ids[tri0[1]], ids[tri0[2]]);
            appendTri(faceid, 2, ids[tri1[0]], ids[tri1[1]], ids[tri1[2]]);
        }
    }
    this->cellTriOffset[this->cells.size()] = (int)this->cellTriCache.size();
}

int ProcessDSMC::localOfGlobalFace(int gid) const
{
    if (gid >= 0 && gid < (int)this->faceGid2LocalDense.size())
        return this->faceGid2LocalDense[(std::size_t)gid];
    auto it = this->face_gid2local.find(gid);
    if (it == this->face_gid2local.end()) return -1;
    return it->second;
}

void ProcessDSMC::ensure_partition_flux_storage()
{
    if (!this->enable_partition_flux_weights) return;
    if (this->mess.Ncell <= 0 && this->mesh == nullptr) return;
}

void ProcessDSMC::reset_partition_flux_counts()
{
    this->partition_edge_flux.clear();
}

void ProcessDSMC::ensure_partition_time_storage()
{
    const int ncell = (this->mess.Ncell > 0)
        ? this->mess.Ncell
        : ((this->mesh != nullptr) ? this->mesh->Ncell : 0);
    if (ncell <= 0) return;
    if (this->cell_time_weight_accum.size() > (std::size_t)ncell)
        this->cell_time_weight_accum.clear();
    if (this->mpi != nullptr && this->mpi->root() &&
        (int)this->cell_time_weight_ema.size() != ncell)
        this->cell_time_weight_ema.assign((std::size_t)ncell, 0.0);
}

void ProcessDSMC::reset_partition_time_weights()
{
    this->cell_time_weight_accum.clear();
}

void ProcessDSMC::accumulate_cell_time_weight(int globalCell, double dt)
{
    if (!this->enable_partition_time_weights) return;
    if (!(dt > 0.0) || !std::isfinite(dt)) return;
    if (globalCell < 0) return;
    const int ncell = (this->mess.Ncell > 0)
        ? this->mess.Ncell
        : ((this->mesh != nullptr) ? this->mesh->Ncell : 0);
    if (globalCell >= ncell) return;
    if (this->cell_time_weight_accum.empty() && this->iNcell > 0)
        this->cell_time_weight_accum.reserve((std::size_t)this->iNcell);
    this->cell_time_weight_accum[globalCell] += dt;
}

void ProcessDSMC::record_partition_flux(int oldCell, int endOldCell, int startLocalCell)
{
    if (!this->enable_partition_flux_weights) return;
    if (oldCell < 0 || endOldCell < 0 || oldCell == endOldCell) return;
    const int ncell = (this->mess.Ncell > 0)
        ? this->mess.Ncell
        : ((this->mesh != nullptr) ? this->mesh->Ncell : 0);
    if (ncell <= 0 || oldCell >= ncell || endOldCell >= ncell) return;
    auto recordFromCell = [&](const DsmcCell& src) -> bool
    {
        const int slots = (src.num > 0 && src.num < NN) ? src.num : NN;
        for (int j = 0; j < slots; ++j)
        {
            if (src.cell2cell[j] == endOldCell)
            {
                const std::size_t idx = (std::size_t)oldCell * (std::size_t)NN + (std::size_t)j;
                this->partition_edge_flux[idx] += 1;
                return true;
            }
        }
        return false;
    };
    if (startLocalCell >= 0 && startLocalCell < (int)this->cells.size())
    {
        if (recordFromCell(this->cells[(std::size_t)startLocalCell]))
            return;
    }
    if (this->mesh != nullptr &&
        oldCell >= 0 &&
        oldCell < (int)this->mesh->Dsmccells.size())
    {
        recordFromCell(this->mesh->Dsmccells[(std::size_t)oldCell]);
    }
}
