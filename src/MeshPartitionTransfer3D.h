/*
 * Partition state and local geometry transfer interfaces.
 */

#pragma once

#include "MessagePassing.h"
#include "MpiContext.h"
#include "meshImport.h"
#include <cstddef>
#include <mpi.h>
#include <unordered_map>
#include <vector>

class MeshparticalInitial;
class ProcessDSMC;

// PartitionState3D stores state used by this module.
struct PartitionState3D
{
    // ownerByCell[globalCell] gives the compute rank that owns that cell.
    std::vector<int> ownerByCell;
    // Reverse map used when building per-rank geometry packages.
    std::vector<std::vector<int>> cellsByRank;
    // Epoch increments whenever the ownership map changes.
    int epoch = 0;
// clear: prepares derived solver state.
    void clear()
    {
        ownerByCell.clear();
        cellsByRank.clear();
        epoch = 0;
    }
// assign (owners, rankCount, nextEpoch): performs one solver support operation.
    void assign(const std::vector<int> &owners, int rankCount, int nextEpoch = -1)
    {
        ownerByCell = owners;
        rebuildCellsByRank(rankCount);
        if (nextEpoch >= 0)
            epoch = nextEpoch;
        else
            ++epoch;
    }
// rebuildCellsByRank (rankCount): works with mesh topology or geometric intersections.
    void rebuildCellsByRank(int rankCount)
    {
        cellsByRank.assign((rankCount > 0) ? (std::size_t)rankCount : 0u, std::vector<int>());
        for (int gid = 0; gid < (int)ownerByCell.size(); ++gid)
        {
            const int owner = ownerByCell[(std::size_t)gid];
            if (owner >= 0 && owner < (int)cellsByRank.size())
                cellsByRank[(std::size_t)owner].push_back(gid);
        }
    }
// ownerOf (globalCell): updates partition ownership or load data.
    int ownerOf(int globalCell) const
    {
        if (globalCell < 0 || globalCell >= (int)ownerByCell.size()) return -1;
        return ownerByCell[(std::size_t)globalCell];
    }
};

// LocalGeometry3D stores state used by this module.
struct LocalGeometry3D
{
    // Local cells include owned cells followed by halo cells.
    std::vector<DsmcCell> cells;
    std::vector<DsmcEdge> edges;
    // Global ids preserve the mapping back to the imported mesh.
    std::vector<int> local_cells;
    std::vector<int> face_gids;
    std::unordered_map<int, int> gid2local;
    std::unordered_map<int, int> face_gid2local;
    std::vector<double> localPointXY;
    std::vector<unsigned char> faceSplitTag;
    // Number of entries at the front of cells/local_cells that are owned.
    int ownedCount = 0;
};

// MeshPartitionTransfer3D stores state used by this module.
class MeshPartitionTransfer3D
{
public:
// MeshPartitionTransfer3D (mesh, partinit, mpass, mpiCtx): updates partition ownership or load data.
    MeshPartitionTransfer3D(meshImport *mesh,
                            MeshparticalInitial *partinit,
                            MessagePassing *mpass,
                            const MpiContext &mpiCtx);
// broadcastMeshMessage (mess): moves structured data through MPI.
    bool broadcastMeshMessage(meshMessage &mess) const;
// initialPartitionAndDistribute (haloRings): updates partition ownership or load data.
    bool initialPartitionAndDistribute(int haloRings);
// distributeGeometryByOwners (state, haloRings): updates partition ownership or load data.
    bool distributeGeometryByOwners(const PartitionState3D &state, int haloRings);
// takeLocalGeometry: works with mesh topology or geometric intersections.
    LocalGeometry3D takeLocalGeometry();
// installLocalGeometryTo (process): works with mesh topology or geometric intersections.
    bool installLocalGeometryTo(ProcessDSMC &process);
// broadcastInitialParticleCounts (npcByCell): moves structured data through MPI.
    bool broadcastInitialParticleCounts(std::vector<int> &npcByCell) const;
private:
// LocalGeometryCounts3D stores state used by this module.
    struct LocalGeometryCounts3D
    {
        int ownedCellCount = 0;
        int ncell = 0;
        int nface = 0;
        int nxyz = 0;
    };
// MeshPartitionPackage3D stores state used by this module.
    struct MeshPartitionPackage3D
    {
        // Serialized geometry package sent from the root rank to one worker.
        int ownedCellCount = 0;
        std::vector<int> cellGids;
        std::vector<DsmcCell> cells;
        std::vector<int> faceGids;
        std::vector<DsmcEdge> edges;
        std::vector<unsigned char> faceSplitTags;
        std::vector<int> nodeGids;
        std::vector<double> nodeXyz;
    };
// RootScatterBuffers stores state used by this module.
    struct RootScatterBuffers
    {
        std::vector<int> cellCounts;
        std::vector<int> ownedCounts;
        std::vector<int> faceCounts;
        std::vector<int> xyzCounts;
        std::vector<int> cellDispls;
        std::vector<int> faceDispls;
        std::vector<int> xyzDispls;
        std::vector<int> cellGids;
        std::vector<DsmcCell> cells;
        std::vector<int> faceGids;
        std::vector<DsmcEdge> edges;
        std::vector<unsigned char> faceSplitTags;
        std::vector<double> nodeXyz;
    };
// HaloBuildResult stores state used by this module.
    struct HaloBuildResult
    {
        // Halo layers are kept so diagnostics can report missing adjacency depth.
        std::vector<int> ownedCellGids;
        std::vector<int> haloCellGids;
        std::vector<int> ownedCellLayers;
        std::vector<int> haloCellLayers;
        std::vector<int> allCellGids;
        std::vector<int> allCellLayers;
        int ownedCount = 0;
    };
// isRoot: performs one solver support operation.
    bool isRoot() const;
// worldFromCal (calRank): performs one solver support operation.
    int worldFromCal(int calRank) const;
// initialPartitionRoot (rankCellAll): updates partition ownership or load data.
    bool initialPartitionRoot(std::vector<int> &rankCellAll) const;
// syncDsmcCellOwners (rankCellAll): updates partition ownership or load data.
    void syncDsmcCellOwners(const std::vector<int> &rankCellAll) const;
// restoreOriginalMesh: performs one solver support operation.
    void restoreOriginalMesh() const;
// initializeCounts (buffers): prepares derived solver state.
    void initializeCounts(RootScatterBuffers &buffers) const;
// appendPackageToRootBuffers (worldRank, package, buffers): moves structured data through MPI.
    void appendPackageToRootBuffers(int worldRank,
                                    const MeshPartitionPackage3D &package,
                                    RootScatterBuffers &buffers) const;
// resizeLocalGeometryStorage (myNcell, myOwnedNcell, myNface, myNxyz): works with mesh topology or geometric intersections.
    void resizeLocalGeometryStorage(int myNcell,
                                    int myOwnedNcell,
                                    int myNface,
                                    int myNxyz) const;
// installPackageLocal (package): moves structured data through MPI.
    bool installPackageLocal(const MeshPartitionPackage3D &package) const;
// sendPackageToWorld (worldRank, package, mpiCell, mpiEdge): moves structured data through MPI.
    bool sendPackageToWorld(int worldRank,
                            const MeshPartitionPackage3D &package,
                            MPI_Datatype mpiCell,
                            MPI_Datatype mpiEdge) const;
// recvPackageFromRoot (mpiCell, mpiEdge): moves structured data through MPI.
    bool recvPackageFromRoot(MPI_Datatype mpiCell,
                             MPI_Datatype mpiEdge) const;
// buildRootBuffers (state, haloRings, buffers): performs one solver support operation.
    bool buildRootBuffers(const PartitionState3D &state,
                          int haloRings,
                          RootScatterBuffers &buffers) const;
// buildOwnedHaloCells (calRank, state, haloRings): works with mesh topology or geometric intersections.
    HaloBuildResult buildOwnedHaloCells(int calRank,
                                        const PartitionState3D &state,
                                        int haloRings) const;
// buildPackageForRank (calRank, state, haloRings, package): moves structured data through MPI.
    bool buildPackageForRank(int calRank,
                             const PartitionState3D &state,
                             int haloRings,
                             MeshPartitionPackage3D &package) const;
// fillPackageCells (halo, package): moves structured data through MPI.
    bool fillPackageCells(const HaloBuildResult &halo,
                          MeshPartitionPackage3D &package) const;
// fillPackageFaces (package): moves structured data through MPI.
    bool fillPackageFaces(MeshPartitionPackage3D &package) const;
// fillPackageNodes (package): moves structured data through MPI.
    bool fillPackageNodes(MeshPartitionPackage3D &package) const;
// reportHaloAdjacencyMisses (calRank, state, haloRings, halo): updates partition ownership or load data.
    void reportHaloAdjacencyMisses(int calRank,
                                   const PartitionState3D &state,
                                   int haloRings,
                                   const HaloBuildResult &halo) const;
// scatterCounts (buffers, myNcell, myOwnedNcell, myNface, myNxyz): moves structured data through MPI.
    bool scatterCounts(const RootScatterBuffers &buffers,
                       int &myNcell,
                       int &myOwnedNcell,
                       int &myNface,
                       int &myNxyz) const;
// scatterGeometry (buffers, mpiCell, mpiEdge, myNcell, myNface, myNxyz, localCells, cells, faceGids, edges, faceSplitTags, localPointXyz): moves structured data through MPI.
    bool scatterGeometry(const RootScatterBuffers &buffers,
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
                         std::vector<double> &localPointXyz) const;
// rebuildInitialMaps: prepares derived solver state.
    void rebuildInitialMaps() const;
    meshImport *m_mesh = nullptr;
    MeshparticalInitial *m_partinit = nullptr;
    MessagePassing *m_mpass = nullptr;
    const MpiContext *m_mpi = nullptr;
    mutable std::vector<int> m_cellVisitStamp;
    mutable int m_cellVisitEpoch = 1;
    mutable std::vector<int> m_faceVisitStamp;
    mutable std::vector<int> m_faceLocalMap;
    mutable int m_faceVisitEpoch = 1;
    mutable std::vector<int> m_nodeVisitStamp;
    mutable std::vector<int> m_nodeLocalMap;
    mutable int m_nodeVisitEpoch = 1;
};
