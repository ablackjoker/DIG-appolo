#include "MeshPartitionTransfer3D.h"

#include "MeshparticalInitial.h"
#include "ProcessDSMC.h"

#include <algorithm>
#include <climits>
#include <iostream>
#include <unordered_map>

namespace
{
    const bool kReportHaloAdjacency = false;
    const int kGeomCountsTag = 4300;
    const int kGeomCellGidsTag = 4301;
    const int kGeomCellsTag = 4302;
    const int kGeomFaceGidsTag = 4303;
    const int kGeomEdgesTag = 4304;
    const int kGeomFaceSplitTag = 4305;
    const int kGeomNodeXyzTag = 4306;

    template <typename T>
    inline T *partition3dDataOrNull(std::vector<T> &values)
    {
        return values.empty() ? nullptr : values.data();
    }

    template <typename T>
    inline T *partition3dDataOrNull(const std::vector<T> &values)
    {
        return values.empty() ? nullptr : const_cast<T *>(values.data());
    }

    inline void prefixSumCounts(const std::vector<int> &counts, std::vector<int> &displs)
    {
        displs.assign(counts.size(), 0);
        for (std::size_t i = 1; i < counts.size(); ++i)
            displs[i] = displs[i - 1] + counts[i - 1];
    }
}

MeshPartitionTransfer3D::MeshPartitionTransfer3D(meshImport *mesh,
                                                 MeshparticalInitial *partinit,
                                                 MessagePassing *mpass,
                                                 const MpiContext &mpiCtx)
    : m_mesh(mesh), m_partinit(partinit), m_mpass(mpass), m_mpi(&mpiCtx)
{
}

bool MeshPartitionTransfer3D::broadcastMeshMessage(meshMessage &mess) const
{
    if (m_mpass == nullptr || m_mpi == nullptr) return false;

    MPI_Datatype mpiMessage = MPI_DATATYPE_NULL;
    if (!m_mpass->commitMyMesssge(mpiMessage)) return false;
    const int rc = MPI_Bcast(&mess, 1, mpiMessage, 0, m_mpi->comm);
    MPI_Type_free(&mpiMessage);
    return rc == MPI_SUCCESS;
}

bool MeshPartitionTransfer3D::initialPartitionAndDistribute(int haloRings)
{
    if (m_partinit == nullptr || m_mpi == nullptr) return false;

    m_partinit->rank_cell_all.assign((std::size_t)m_partinit->mess.Ncell, 0);
    bool ok = true;
    if (isRoot())
        ok = initialPartitionRoot(m_partinit->rank_cell_all);

    int okInt = ok ? 1 : 0;
    MPI_Bcast(&okInt, 1, MPI_INT, 0, m_mpi->comm);
    if (!okInt) return false;

    MPI_Bcast(m_partinit->rank_cell_all.data(), m_partinit->mess.Ncell, MPI_INT, 0, m_mpi->comm);
    m_partinit->partitionState.assign(m_partinit->rank_cell_all, m_mpi->c_size);

    if (isRoot() && m_mesh != nullptr)
        m_mesh->rootCaptureGlobalDsmcAndReleaseMesh();

    return distributeGeometryByOwners(m_partinit->partitionState, haloRings);
}

bool MeshPartitionTransfer3D::distributeGeometryByOwners(const PartitionState3D &state, int haloRings)
{
    if (m_partinit == nullptr || m_mpass == nullptr || m_mpi == nullptr) return false;

    m_partinit->partitionState = state;
    m_partinit->rank_cell_all = state.ownerByCell;

    MPI_Datatype mpiCell = MPI_DATATYPE_NULL;
    MPI_Datatype mpiEdge = MPI_DATATYPE_NULL;
    if (!m_mpass->commitMyDsmcCell(mpiCell)) return false;
    if (!m_mpass->commitMyDsmcEdge(mpiEdge))
    {
        MPI_Type_free(&mpiCell);
        return false;
    }

    bool ok = true;
    if (isRoot())
    {
        const int rings = std::max(0, haloRings);
        for (int calRank = 0; calRank < m_mpi->c_size; ++calRank)
        {
            MeshPartitionPackage3D package;
            if (ok)
                ok = buildPackageForRank(calRank, state, rings, package);

            if (!ok)
            {
                MeshPartitionPackage3D failedPackage;
                failedPackage.ownedCellCount = -1;
                if (!sendPackageToWorld(worldFromCal(calRank), failedPackage, mpiCell, mpiEdge))
                    break;
                continue;
            }

            if (!sendPackageToWorld(worldFromCal(calRank), package, mpiCell, mpiEdge))
            {
                ok = false;
                break;
            }
        }
    }
    else if (m_mpi->active())
    {
        ok = recvPackageFromRoot(mpiCell, mpiEdge);
    }

    MPI_Type_free(&mpiCell);
    MPI_Type_free(&mpiEdge);

    int failInt = ok ? 0 : 1;
    MPI_Allreduce(MPI_IN_PLACE, &failInt, 1, MPI_INT, MPI_LOR, m_mpi->comm);
    MPI_Barrier(m_mpi->comm);
    return failInt == 0;
}

LocalGeometry3D MeshPartitionTransfer3D::takeLocalGeometry()
{
    LocalGeometry3D geometry;
    if (m_partinit == nullptr) return geometry;

    geometry.ownedCount = m_partinit->my_owned_ncell;
    geometry.cells = std::move(m_partinit->cells);
    geometry.edges = std::move(m_partinit->edges);
    geometry.local_cells = std::move(m_partinit->local_cells);
    geometry.face_gids = std::move(m_partinit->face_gids);
    geometry.gid2local = std::move(m_partinit->gid2local);
    geometry.face_gid2local = std::move(m_partinit->face_gid2local);
    geometry.localPointXY = std::move(m_partinit->localPointXY);
    geometry.faceSplitTag = std::move(m_partinit->faceSplitTag);
    return geometry;
}

bool MeshPartitionTransfer3D::installLocalGeometryTo(ProcessDSMC &process)
{
    if (m_partinit == nullptr) return false;
    process.installLocalGeometry(takeLocalGeometry());
    return true;
}

bool MeshPartitionTransfer3D::broadcastInitialParticleCounts(std::vector<int> &npcByCell) const
{
    if (m_partinit == nullptr || m_mpi == nullptr) return false;

    npcByCell.assign((std::size_t)m_partinit->mess.Ncell, 0);
    if (isRoot())
    {
        if (m_mesh == nullptr || m_mesh->Npc_exa == nullptr) return false;
        for (int i = 0; i < m_partinit->mess.Ncell; ++i)
            npcByCell[(std::size_t)i] = m_mesh->Npc_exa[i];
    }

    return MPI_Bcast(npcByCell.data(), m_partinit->mess.Ncell, MPI_INT, 0, m_mpi->comm) == MPI_SUCCESS;
}

bool MeshPartitionTransfer3D::isRoot() const
{
    return m_mpi != nullptr && m_mpi->root();
}

int MeshPartitionTransfer3D::worldFromCal(int calRank) const
{
    return calRank + 1;
}

bool MeshPartitionTransfer3D::initialPartitionRoot(std::vector<int> &rankCellAll) const
{
    if (m_mesh == nullptr || m_partinit == nullptr || m_mpi == nullptr) return false;

    restoreOriginalMesh();
    m_mesh->metisPartition((idx_t)m_mpi->c_size);

    rankCellAll.assign((std::size_t)m_partinit->mess.Ncell, 0);
    for (int i = 0; i < m_partinit->mess.Ncell; ++i)
        rankCellAll[(std::size_t)i] = m_mesh->cells[i].no;

    syncDsmcCellOwners(rankCellAll);
    return true;
}

void MeshPartitionTransfer3D::syncDsmcCellOwners(const std::vector<int> &rankCellAll) const
{
    if (m_mesh == nullptr) return;
    const int n = std::min((int)m_mesh->Dsmccells.size(), (int)rankCellAll.size());
    for (int i = 0; i < n; ++i)
        m_mesh->Dsmccells[(std::size_t)i].no = rankCellAll[(std::size_t)i];
}

void MeshPartitionTransfer3D::restoreOriginalMesh() const
{
    if (m_mesh == nullptr || m_partinit == nullptr) return;

    if ((int)m_mesh->originalCells.size() >= m_partinit->mess.Ncell)
    {
        for (int i = 0; i < m_partinit->mess.Ncell; ++i)
            m_mesh->cells[i] = m_mesh->originalCells[(std::size_t)i];
        std::vector<cell>().swap(m_mesh->originalCells);
    }

    if ((int)m_mesh->originalEdges.size() >= m_partinit->mess.Nface)
    {
        for (int i = 0; i < m_partinit->mess.Nface; ++i)
            m_mesh->edges[i] = m_mesh->originalEdges[(std::size_t)i];
        std::vector<edge>().swap(m_mesh->originalEdges);
    }
}

void MeshPartitionTransfer3D::initializeCounts(RootScatterBuffers &buffers) const
{
    const std::size_t nworld = (m_mpi == nullptr) ? 0u : (std::size_t)m_mpi->size;
    buffers.cellCounts.assign(nworld, 0);
    buffers.ownedCounts.assign(nworld, 0);
    buffers.faceCounts.assign(nworld, 0);
    buffers.xyzCounts.assign(nworld, 0);
}

void MeshPartitionTransfer3D::appendPackageToRootBuffers(int worldRank,
                                                         const MeshPartitionPackage3D &package,
                                                         RootScatterBuffers &buffers) const
{
    buffers.cellCounts[(std::size_t)worldRank] = (int)package.cells.size();
    buffers.ownedCounts[(std::size_t)worldRank] = package.ownedCellCount;
    buffers.faceCounts[(std::size_t)worldRank] = (int)package.edges.size();
    buffers.xyzCounts[(std::size_t)worldRank] = (int)package.nodeXyz.size();

    buffers.cellGids.insert(buffers.cellGids.end(), package.cellGids.begin(), package.cellGids.end());
    buffers.cells.insert(buffers.cells.end(), package.cells.begin(), package.cells.end());
    buffers.faceGids.insert(buffers.faceGids.end(), package.faceGids.begin(), package.faceGids.end());
    buffers.edges.insert(buffers.edges.end(), package.edges.begin(), package.edges.end());
    buffers.faceSplitTags.insert(buffers.faceSplitTags.end(), package.faceSplitTags.begin(), package.faceSplitTags.end());
    buffers.nodeXyz.insert(buffers.nodeXyz.end(), package.nodeXyz.begin(), package.nodeXyz.end());
}

void MeshPartitionTransfer3D::resizeLocalGeometryStorage(int myNcell,
                                                         int myOwnedNcell,
                                                         int myNface,
                                                         int myNxyz) const
{
    m_partinit->my_ncell = myNcell;
    m_partinit->my_nface = myNface;
    m_partinit->my_owned_ncell = myOwnedNcell;

    m_partinit->cells.assign((std::size_t)myNcell, DsmcCell());
    m_partinit->edges.assign((std::size_t)myNface, DsmcEdge());
    m_partinit->localPointXY.assign((std::size_t)myNxyz, 0.0);
    m_partinit->faceSplitTag.assign((std::size_t)myNface, meshImport::FACE_SPLIT_INVALID);
    m_partinit->local_cells.assign((std::size_t)myNcell, 0);
    m_partinit->face_gids.assign((std::size_t)myNface, 0);
}

bool MeshPartitionTransfer3D::installPackageLocal(const MeshPartitionPackage3D &package) const
{
    if (m_partinit == nullptr) return false;
    if (package.ownedCellCount < 0) return false;

    resizeLocalGeometryStorage((int)package.cells.size(),
                               package.ownedCellCount,
                               (int)package.edges.size(),
                               (int)package.nodeXyz.size());
    m_partinit->local_cells = package.cellGids;
    m_partinit->cells = package.cells;
    m_partinit->face_gids = package.faceGids;
    m_partinit->edges = package.edges;
    m_partinit->faceSplitTag = package.faceSplitTags;
    m_partinit->localPointXY = package.nodeXyz;
    rebuildInitialMaps();
    return true;
}

bool MeshPartitionTransfer3D::sendPackageToWorld(int worldRank,
                                                 const MeshPartitionPackage3D &package,
                                                 MPI_Datatype mpiCell,
                                                 MPI_Datatype mpiEdge) const
{
    LocalGeometryCounts3D counts;
    counts.ownedCellCount = package.ownedCellCount;
    if (package.ownedCellCount >= 0)
    {
        counts.ncell = (int)package.cells.size();
        counts.nface = (int)package.edges.size();
        counts.nxyz = (int)package.nodeXyz.size();
    }

    int header[4] = {
        counts.ownedCellCount,
        counts.ncell,
        counts.nface,
        counts.nxyz
    };
    MPI_Request req = MPI_REQUEST_NULL;
    if (MPI_Isend(header, 4, MPI_INT, worldRank, kGeomCountsTag, m_mpi->comm, &req) != MPI_SUCCESS)
        return false;
    if (MPI_Waitall(1, &req, MPI_STATUSES_IGNORE) != MPI_SUCCESS)
        return false;
    if (counts.ownedCellCount < 0) return true;

    if (counts.ncell > 0)
    {
        if (MPI_Isend(partition3dDataOrNull(package.cellGids), counts.ncell, MPI_INT,
                      worldRank, kGeomCellGidsTag, m_mpi->comm, &req) != MPI_SUCCESS)
            return false;
        if (MPI_Waitall(1, &req, MPI_STATUSES_IGNORE) != MPI_SUCCESS)
            return false;
        if (MPI_Isend(partition3dDataOrNull(package.cells), counts.ncell, mpiCell,
                      worldRank, kGeomCellsTag, m_mpi->comm, &req) != MPI_SUCCESS)
            return false;
        if (MPI_Waitall(1, &req, MPI_STATUSES_IGNORE) != MPI_SUCCESS)
            return false;
    }
    if (counts.nface > 0)
    {
        if (MPI_Isend(partition3dDataOrNull(package.faceGids), counts.nface, MPI_INT,
                      worldRank, kGeomFaceGidsTag, m_mpi->comm, &req) != MPI_SUCCESS)
            return false;
        if (MPI_Waitall(1, &req, MPI_STATUSES_IGNORE) != MPI_SUCCESS)
            return false;
        if (MPI_Isend(partition3dDataOrNull(package.edges), counts.nface, mpiEdge,
                      worldRank, kGeomEdgesTag, m_mpi->comm, &req) != MPI_SUCCESS)
            return false;
        if (MPI_Waitall(1, &req, MPI_STATUSES_IGNORE) != MPI_SUCCESS)
            return false;
        if (MPI_Isend(partition3dDataOrNull(package.faceSplitTags), counts.nface, MPI_UNSIGNED_CHAR,
                      worldRank, kGeomFaceSplitTag, m_mpi->comm, &req) != MPI_SUCCESS)
            return false;
        if (MPI_Waitall(1, &req, MPI_STATUSES_IGNORE) != MPI_SUCCESS)
            return false;
    }
    if (counts.nxyz > 0)
    {
        if (MPI_Isend(partition3dDataOrNull(package.nodeXyz), counts.nxyz, MPI_DOUBLE,
                      worldRank, kGeomNodeXyzTag, m_mpi->comm, &req) != MPI_SUCCESS)
            return false;
        if (MPI_Waitall(1, &req, MPI_STATUSES_IGNORE) != MPI_SUCCESS)
            return false;
    }
    return true;
}

bool MeshPartitionTransfer3D::recvPackageFromRoot(MPI_Datatype mpiCell,
                                                  MPI_Datatype mpiEdge) const
{
    int header[4] = {0, 0, 0, 0};
    MPI_Request req = MPI_REQUEST_NULL;
    if (MPI_Irecv(header, 4, MPI_INT, 0, kGeomCountsTag, m_mpi->comm, &req) != MPI_SUCCESS)
        return false;
    if (MPI_Waitall(1, &req, MPI_STATUSES_IGNORE) != MPI_SUCCESS)
        return false;

    LocalGeometryCounts3D counts;
    counts.ownedCellCount = header[0];
    counts.ncell = header[1];
    counts.nface = header[2];
    counts.nxyz = header[3];
    if (counts.ownedCellCount < 0) return false;

    resizeLocalGeometryStorage(counts.ncell, counts.ownedCellCount, counts.nface, counts.nxyz);

    if (counts.ncell > 0)
    {
        if (MPI_Irecv(partition3dDataOrNull(m_partinit->local_cells), counts.ncell, MPI_INT,
                      0, kGeomCellGidsTag, m_mpi->comm, &req) != MPI_SUCCESS)
            return false;
        if (MPI_Waitall(1, &req, MPI_STATUSES_IGNORE) != MPI_SUCCESS)
            return false;
        if (MPI_Irecv(partition3dDataOrNull(m_partinit->cells), counts.ncell, mpiCell,
                      0, kGeomCellsTag, m_mpi->comm, &req) != MPI_SUCCESS)
            return false;
        if (MPI_Waitall(1, &req, MPI_STATUSES_IGNORE) != MPI_SUCCESS)
            return false;
    }
    if (counts.nface > 0)
    {
        if (MPI_Irecv(partition3dDataOrNull(m_partinit->face_gids), counts.nface, MPI_INT,
                      0, kGeomFaceGidsTag, m_mpi->comm, &req) != MPI_SUCCESS)
            return false;
        if (MPI_Waitall(1, &req, MPI_STATUSES_IGNORE) != MPI_SUCCESS)
            return false;
        if (MPI_Irecv(partition3dDataOrNull(m_partinit->edges), counts.nface, mpiEdge,
                      0, kGeomEdgesTag, m_mpi->comm, &req) != MPI_SUCCESS)
            return false;
        if (MPI_Waitall(1, &req, MPI_STATUSES_IGNORE) != MPI_SUCCESS)
            return false;
        if (MPI_Irecv(partition3dDataOrNull(m_partinit->faceSplitTag), counts.nface, MPI_UNSIGNED_CHAR,
                      0, kGeomFaceSplitTag, m_mpi->comm, &req) != MPI_SUCCESS)
            return false;
        if (MPI_Waitall(1, &req, MPI_STATUSES_IGNORE) != MPI_SUCCESS)
            return false;
    }
    if (counts.nxyz > 0)
    {
        if (MPI_Irecv(partition3dDataOrNull(m_partinit->localPointXY), counts.nxyz, MPI_DOUBLE,
                      0, kGeomNodeXyzTag, m_mpi->comm, &req) != MPI_SUCCESS)
            return false;
        if (MPI_Waitall(1, &req, MPI_STATUSES_IGNORE) != MPI_SUCCESS)
            return false;
    }

    rebuildInitialMaps();
    return true;
}

bool MeshPartitionTransfer3D::buildRootBuffers(const PartitionState3D &state,
                                               int haloRings,
                                               RootScatterBuffers &buffers) const
{
    initializeCounts(buffers);
    if (!isRoot()) return true;
    if (m_mesh == nullptr || m_partinit == nullptr) return false;

    const int rings = std::max(0, haloRings);
    for (int calRank = 0; calRank < m_mpi->c_size; ++calRank)
    {
        MeshPartitionPackage3D package;
        if (!buildPackageForRank(calRank, state, rings, package))
            return false;

        const int worldRank = worldFromCal(calRank);
        appendPackageToRootBuffers(worldRank, package, buffers);
    }

    prefixSumCounts(buffers.cellCounts, buffers.cellDispls);
    prefixSumCounts(buffers.faceCounts, buffers.faceDispls);
    prefixSumCounts(buffers.xyzCounts, buffers.xyzDispls);
    return true;
}

MeshPartitionTransfer3D::HaloBuildResult MeshPartitionTransfer3D::buildOwnedHaloCells(
    int calRank,
    const PartitionState3D &state,
    int haloRings) const
{
    HaloBuildResult result;
    if (m_mesh == nullptr) return result;

    const int ncell = (int)m_mesh->Dsmccells.size();
    if ((int)state.ownerByCell.size() < ncell) return result;

    if ((int)m_cellVisitStamp.size() < ncell)
        m_cellVisitStamp.assign((std::size_t)ncell, 0);
    if (m_cellVisitEpoch >= INT_MAX)
    {
        std::fill(m_cellVisitStamp.begin(), m_cellVisitStamp.end(), 0);
        m_cellVisitEpoch = 1;
    }
    else
    {
        ++m_cellVisitEpoch;
    }
    const int visitEpoch = m_cellVisitEpoch;
    auto visited = [&](int gid) -> bool
    {
        return gid >= 0 && gid < ncell &&
               m_cellVisitStamp[(std::size_t)gid] == visitEpoch;
    };
    auto markVisited = [&](int gid)
    {
        if (gid >= 0 && gid < ncell)
            m_cellVisitStamp[(std::size_t)gid] = visitEpoch;
    };

    std::vector<int> frontier;
    frontier.reserve((std::size_t)ncell / 4u + 16u);

    if (calRank >= 0 && calRank < (int)state.cellsByRank.size())
    {
        const std::vector<int> &ownedSeed = state.cellsByRank[(std::size_t)calRank];
        for (int gid : ownedSeed)
        {
            if (gid < 0 || gid >= ncell) continue;
            result.ownedCellGids.push_back(gid);
            result.ownedCellLayers.push_back(0);
            frontier.push_back(gid);
            markVisited(gid);
        }
    }
    else
    {
        for (int gid = 0; gid < ncell; ++gid)
        {
            if (state.ownerByCell[(std::size_t)gid] == calRank)
            {
                result.ownedCellGids.push_back(gid);
                result.ownedCellLayers.push_back(0);
                frontier.push_back(gid);
                markVisited(gid);
            }
        }
    }

    result.ownedCount = (int)result.ownedCellGids.size();

    std::vector<int> next;
    next.reserve(frontier.size());
    for (int ring = 0; ring < haloRings; ++ring)
    {
        next.clear();
        for (int gid : frontier)
        {
            const DsmcCell &cell = m_mesh->Dsmccells[(std::size_t)gid];
            for (int k = 0; k < cell.num && k < NN; ++k)
            {
                const int neighbor = cell.cell2cell[k];
                if (neighbor < 0 || neighbor >= ncell) continue;
                if (visited(neighbor)) continue;

                markVisited(neighbor);
                next.push_back(neighbor);

                if (state.ownerByCell[(std::size_t)neighbor] != calRank)
                {
                    result.haloCellGids.push_back(neighbor);
                    result.haloCellLayers.push_back(ring + 1);
                }
            }
        }

        frontier.swap(next);
        if (frontier.empty()) break;
    }

    result.allCellGids.reserve(result.ownedCellGids.size() + result.haloCellGids.size());
    result.allCellLayers.reserve(result.ownedCellLayers.size() + result.haloCellLayers.size());
    result.allCellGids.insert(result.allCellGids.end(), result.ownedCellGids.begin(), result.ownedCellGids.end());
    result.allCellGids.insert(result.allCellGids.end(), result.haloCellGids.begin(), result.haloCellGids.end());
    result.allCellLayers.insert(result.allCellLayers.end(), result.ownedCellLayers.begin(), result.ownedCellLayers.end());
    result.allCellLayers.insert(result.allCellLayers.end(), result.haloCellLayers.begin(), result.haloCellLayers.end());
    return result;
}

bool MeshPartitionTransfer3D::buildPackageForRank(int calRank,
                                                  const PartitionState3D &state,
                                                  int haloRings,
                                                  MeshPartitionPackage3D &package) const
{
    package = MeshPartitionPackage3D();
    if (m_mesh == nullptr || m_partinit == nullptr) return false;

    const HaloBuildResult halo = buildOwnedHaloCells(calRank, state, haloRings);

    if (kReportHaloAdjacency &&
        haloRings > 0 &&
        halo.allCellLayers.size() == halo.allCellGids.size())
        reportHaloAdjacencyMisses(calRank, state, haloRings, halo);

    if (!fillPackageCells(halo, package))
        return false;
    if (!fillPackageFaces(package))
        return false;
    return fillPackageNodes(package);
}

bool MeshPartitionTransfer3D::fillPackageCells(const HaloBuildResult &halo,
                                               MeshPartitionPackage3D &package) const
{
    package.ownedCellCount = halo.ownedCount;
    package.cellGids = halo.allCellGids;
    package.cells.resize(package.cellGids.size());
    for (std::size_t i = 0; i < package.cellGids.size(); ++i)
    {
        const int gid = package.cellGids[i];
        if (gid < 0 || gid >= (int)m_mesh->Dsmccells.size()) return false;
        package.cells[i] = m_mesh->Dsmccells[(std::size_t)gid];
    }
    return true;
}

bool MeshPartitionTransfer3D::fillPackageFaces(MeshPartitionPackage3D &package) const
{
    const int globalFaceCount = std::max(m_partinit->mess.Nface, (int)m_mesh->Dsmcedges.size());
    if ((int)m_faceVisitStamp.size() < globalFaceCount)
    {
        m_faceVisitStamp.assign((std::size_t)globalFaceCount, 0);
        m_faceLocalMap.assign((std::size_t)globalFaceCount, -1);
    }
    if (m_faceVisitEpoch >= INT_MAX)
    {
        std::fill(m_faceVisitStamp.begin(), m_faceVisitStamp.end(), 0);
        m_faceVisitEpoch = 1;
    }
    else
    {
        ++m_faceVisitEpoch;
    }
    const int faceEpoch = m_faceVisitEpoch;

    package.faceGids.reserve(package.cellGids.size() * 8u);
    for (DsmcCell &cell : package.cells)
    {
        for (int j = 0; j < cell.num && j < NN; ++j)
        {
            const int fid = cell.cell2face[j];
            if (fid == -1) break;
            if (fid >= 0 &&
                fid < m_partinit->mess.Nface &&
                fid < (int)m_mesh->Dsmcedges.size() &&
                m_faceVisitStamp[(std::size_t)fid] != faceEpoch)
            {
                m_faceVisitStamp[(std::size_t)fid] = faceEpoch;
                package.faceGids.push_back(fid);
            }
        }
    }

    for (int lf = 0; lf < (int)package.faceGids.size(); ++lf)
    {
        const int gfid = package.faceGids[(std::size_t)lf];
        m_faceVisitStamp[(std::size_t)gfid] = faceEpoch;
        m_faceLocalMap[(std::size_t)gfid] = lf;
    }

    for (DsmcCell &cell : package.cells)
    {
        for (int j = 0; j < cell.num && j < NN; ++j)
        {
            const int fid = cell.cell2face[j];
            if (fid == -1) break;
            if (fid >= 0 &&
                fid < (int)m_faceVisitStamp.size() &&
                m_faceVisitStamp[(std::size_t)fid] == faceEpoch)
                cell.cell2face[j] = m_faceLocalMap[(std::size_t)fid];
        }
    }

    package.edges.resize(package.faceGids.size());
    package.faceSplitTags.resize(package.faceGids.size(), meshImport::FACE_SPLIT_INVALID);
    for (int lf = 0; lf < (int)package.faceGids.size(); ++lf)
    {
        const int gfid = package.faceGids[(std::size_t)lf];
        if (gfid < 0 || gfid >= (int)m_mesh->Dsmcedges.size()) return false;
        package.edges[(std::size_t)lf] = m_mesh->Dsmcedges[(std::size_t)gfid];
        if (gfid < (int)m_mesh->DsmcfaceSplitTag.size())
            package.faceSplitTags[(std::size_t)lf] = m_mesh->DsmcfaceSplitTag[(std::size_t)gfid];
    }
    return true;
}

bool MeshPartitionTransfer3D::fillPackageNodes(MeshPartitionPackage3D &package) const
{
    const int globalNodeCount = (int)(m_mesh->localPointXY.size() / 3u);
    if ((int)m_nodeVisitStamp.size() < globalNodeCount)
    {
        m_nodeVisitStamp.assign((std::size_t)globalNodeCount, 0);
        m_nodeLocalMap.assign((std::size_t)globalNodeCount, -1);
    }
    if (m_nodeVisitEpoch >= INT_MAX)
    {
        std::fill(m_nodeVisitStamp.begin(), m_nodeVisitStamp.end(), 0);
        m_nodeVisitEpoch = 1;
    }
    else
    {
        ++m_nodeVisitEpoch;
    }
    const int nodeEpoch = m_nodeVisitEpoch;

    package.nodeGids.reserve(package.edges.size() * 4u);
    for (const auto &edge : package.edges)
    {
        const int nfaceNode = edge.faceType;
        for (int k = 0; k < nfaceNode; ++k)
        {
            const int ngid = edge.faceMap[k];
            if (ngid >= 0 &&
                ngid < globalNodeCount &&
                m_nodeVisitStamp[(std::size_t)ngid] != nodeEpoch)
            {
                m_nodeVisitStamp[(std::size_t)ngid] = nodeEpoch;
                package.nodeGids.push_back(ngid);
            }
        }
    }

    for (int ln = 0; ln < (int)package.nodeGids.size(); ++ln)
    {
        const int ngid = package.nodeGids[(std::size_t)ln];
        m_nodeVisitStamp[(std::size_t)ngid] = nodeEpoch;
        m_nodeLocalMap[(std::size_t)ngid] = ln;
    }

    for (auto &edge : package.edges)
    {
        const int nfaceNode = edge.faceType;
        for (int k = 0; k < nfaceNode; ++k)
        {
            const int ngid = edge.faceMap[k];
            if (ngid >= 0 &&
                ngid < (int)m_nodeVisitStamp.size() &&
                m_nodeVisitStamp[(std::size_t)ngid] == nodeEpoch)
                edge.faceMap[k] = m_nodeLocalMap[(std::size_t)ngid];
        }
    }

    package.nodeXyz.resize(3u * package.nodeGids.size());
    for (std::size_t ln = 0; ln < package.nodeGids.size(); ++ln)
    {
        const int gid = package.nodeGids[ln];
        if (gid < 0 || 3 * gid + 2 >= (int)m_mesh->localPointXY.size()) return false;
        package.nodeXyz[3u * ln + 0u] = m_mesh->localPointXY[(std::size_t)(3 * gid + 0)];
        package.nodeXyz[3u * ln + 1u] = m_mesh->localPointXY[(std::size_t)(3 * gid + 1)];
        package.nodeXyz[3u * ln + 2u] = m_mesh->localPointXY[(std::size_t)(3 * gid + 2)];
    }

    return true;
}

void MeshPartitionTransfer3D::reportHaloAdjacencyMisses(int calRank,
                                                        const PartitionState3D &state,
                                                        int haloRings,
                                                        const HaloBuildResult &halo) const
{
    if (m_mesh == nullptr) return;

    std::unordered_map<int, int> localLayer;
    localLayer.reserve(halo.allCellGids.size() * 2u + 1u);
    for (std::size_t idx = 0; idx < halo.allCellGids.size(); ++idx)
        localLayer[halo.allCellGids[idx]] = halo.allCellLayers[idx];

    int missCount = 0;
    int firstCell = -1;
    int firstLayer = -1;
    int firstFace = -1;
    int firstNeighbor = -1;
    int firstNeighborRank = -1;
    int firstFaceL = -1;
    int firstFaceR = -1;
    int firstFaceTag = -1;

    for (std::size_t idx = 0; idx < halo.allCellGids.size(); ++idx)
    {
        const int gid = halo.allCellGids[idx];
        const int layer = halo.allCellLayers[idx];
        if (layer < 0 || layer >= haloRings) continue;
        if (gid < 0 || gid >= (int)m_mesh->Dsmccells.size()) continue;

        const DsmcCell &cell = m_mesh->Dsmccells[(std::size_t)gid];
        for (int j = 0; j < cell.num && j < NN; ++j)
        {
            const int fid = cell.cell2face[j];
            if (fid < 0) break;
            if (fid >= (int)m_mesh->Dsmcedges.size()) continue;

            const DsmcEdge &face = m_mesh->Dsmcedges[(std::size_t)fid];
            const int g0 = face.faceMap[4];
            const int g1 = face.faceMap[5];
            if (g0 < 0 || g1 < 0 ||
                g0 >= (int)m_mesh->Dsmccells.size() ||
                g1 >= (int)m_mesh->Dsmccells.size())
                continue;

            int neighbor = -1;
            if (gid == g0) neighbor = g1;
            else if (gid == g1) neighbor = g0;
            else continue;

            if (localLayer.find(neighbor) != localLayer.end()) continue;

            ++missCount;
            if (firstCell < 0)
            {
                firstCell = gid;
                firstLayer = layer;
                firstFace = fid;
                firstNeighbor = neighbor;
                firstNeighborRank = state.ownerOf(neighbor);
                firstFaceL = g0;
                firstFaceR = g1;
                firstFaceTag = face.faceTag;
            }
        }
    }

    if (missCount > 0)
    {
        std::cout << "HALO_ADJ_MISS"
                  << " rank=" << calRank
                  << " haloRings=" << haloRings
                  << " missing=" << missCount
                  << " firstCell=" << firstCell
                  << " firstLayer=" << firstLayer
                  << " firstFace=" << firstFace
                  << " firstFaceTag=" << firstFaceTag
                  << " firstFaceL=" << firstFaceL
                  << " firstFaceR=" << firstFaceR
                  << " firstNeighbor=" << firstNeighbor
                  << " firstNeighborRank=" << firstNeighborRank
                  << std::endl;
    }
}

bool MeshPartitionTransfer3D::scatterCounts(const RootScatterBuffers &buffers,
                                            int &myNcell,
                                            int &myOwnedNcell,
                                            int &myNface,
                                            int &myNxyz) const
{
    if (MPI_Scatter(partition3dDataOrNull(buffers.cellCounts), 1, MPI_INT,
                    &myNcell, 1, MPI_INT, 0, m_mpi->comm) != MPI_SUCCESS)
        return false;
    if (MPI_Scatter(partition3dDataOrNull(buffers.ownedCounts), 1, MPI_INT,
                    &myOwnedNcell, 1, MPI_INT, 0, m_mpi->comm) != MPI_SUCCESS)
        return false;
    if (MPI_Scatter(partition3dDataOrNull(buffers.faceCounts), 1, MPI_INT,
                    &myNface, 1, MPI_INT, 0, m_mpi->comm) != MPI_SUCCESS)
        return false;
    if (MPI_Scatter(partition3dDataOrNull(buffers.xyzCounts), 1, MPI_INT,
                    &myNxyz, 1, MPI_INT, 0, m_mpi->comm) != MPI_SUCCESS)
        return false;
    return true;
}

bool MeshPartitionTransfer3D::scatterGeometry(const RootScatterBuffers &buffers,
                                              MPI_Datatype mpiCell,
                                              MPI_Datatype mpiEdge,
                                              int myNcell,
                                              int myNface,
                                              int myNxyz,
                                              std::vector<int> &localCells,
                                              std::vector<DsmcCell> &cells,
                                              std::vector<int> &faceGids,
                                              std::vector<DsmcEdge> &edges,
                                              std::vector<unsigned char> &faceSplitTags,
                                              std::vector<double> &localPointXyz) const
{
    if (isRoot())
    {
        if (MPI_Scatterv(partition3dDataOrNull(buffers.cellGids), partition3dDataOrNull(buffers.cellCounts),
                         partition3dDataOrNull(buffers.cellDispls), MPI_INT,
                         MPI_IN_PLACE, myNcell, MPI_INT, 0, m_mpi->comm) != MPI_SUCCESS)
            return false;
        if (MPI_Scatterv(partition3dDataOrNull(buffers.cells), partition3dDataOrNull(buffers.cellCounts),
                         partition3dDataOrNull(buffers.cellDispls), mpiCell,
                         MPI_IN_PLACE, myNcell, mpiCell, 0, m_mpi->comm) != MPI_SUCCESS)
            return false;
        if (MPI_Scatterv(partition3dDataOrNull(buffers.faceGids), partition3dDataOrNull(buffers.faceCounts),
                         partition3dDataOrNull(buffers.faceDispls), MPI_INT,
                         MPI_IN_PLACE, myNface, MPI_INT, 0, m_mpi->comm) != MPI_SUCCESS)
            return false;
        if (MPI_Scatterv(partition3dDataOrNull(buffers.edges), partition3dDataOrNull(buffers.faceCounts),
                         partition3dDataOrNull(buffers.faceDispls), mpiEdge,
                         MPI_IN_PLACE, myNface, mpiEdge, 0, m_mpi->comm) != MPI_SUCCESS)
            return false;
        if (MPI_Scatterv(partition3dDataOrNull(buffers.faceSplitTags), partition3dDataOrNull(buffers.faceCounts),
                         partition3dDataOrNull(buffers.faceDispls), MPI_UNSIGNED_CHAR,
                         MPI_IN_PLACE, myNface, MPI_UNSIGNED_CHAR, 0, m_mpi->comm) != MPI_SUCCESS)
            return false;
        if (MPI_Scatterv(partition3dDataOrNull(buffers.nodeXyz), partition3dDataOrNull(buffers.xyzCounts),
                         partition3dDataOrNull(buffers.xyzDispls), MPI_DOUBLE,
                         MPI_IN_PLACE, myNxyz, MPI_DOUBLE, 0, m_mpi->comm) != MPI_SUCCESS)
            return false;
    }
    else
    {
        if (MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT,
                         partition3dDataOrNull(localCells), myNcell, MPI_INT, 0, m_mpi->comm) != MPI_SUCCESS)
            return false;
        if (MPI_Scatterv(nullptr, nullptr, nullptr, mpiCell,
                         partition3dDataOrNull(cells), myNcell, mpiCell, 0, m_mpi->comm) != MPI_SUCCESS)
            return false;
        if (MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT,
                         partition3dDataOrNull(faceGids), myNface, MPI_INT, 0, m_mpi->comm) != MPI_SUCCESS)
            return false;
        if (MPI_Scatterv(nullptr, nullptr, nullptr, mpiEdge,
                         partition3dDataOrNull(edges), myNface, mpiEdge, 0, m_mpi->comm) != MPI_SUCCESS)
            return false;
        if (MPI_Scatterv(nullptr, nullptr, nullptr, MPI_UNSIGNED_CHAR,
                         partition3dDataOrNull(faceSplitTags), myNface, MPI_UNSIGNED_CHAR, 0, m_mpi->comm) != MPI_SUCCESS)
            return false;
        if (MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DOUBLE,
                         partition3dDataOrNull(localPointXyz), myNxyz, MPI_DOUBLE, 0, m_mpi->comm) != MPI_SUCCESS)
            return false;
    }
    return true;
}

void MeshPartitionTransfer3D::rebuildInitialMaps() const
{
    m_partinit->gid2local.clear();
    m_partinit->gid2local.reserve(m_partinit->local_cells.size());
    for (int lc = 0; lc < (int)m_partinit->local_cells.size(); ++lc)
    {
        const int gid = m_partinit->local_cells[(std::size_t)lc];
        if (gid >= 0 && gid < m_partinit->mess.Ncell)
            m_partinit->gid2local[gid] = lc;
    }

    m_partinit->face_gid2local.clear();
    m_partinit->face_gid2local.reserve(m_partinit->face_gids.size());
    for (int lf = 0; lf < (int)m_partinit->face_gids.size(); ++lf)
        m_partinit->face_gid2local[m_partinit->face_gids[(std::size_t)lf]] = lf;
}
