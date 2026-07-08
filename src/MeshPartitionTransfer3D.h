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

struct PartitionState3D
{
    std::vector<int> ownerByCell;
    std::vector<std::vector<int>> cellsByRank;
    int epoch = 0;

    void clear()
    {
        ownerByCell.clear();
        cellsByRank.clear();
        epoch = 0;
    }

    void assign(const std::vector<int> &owners, int rankCount, int nextEpoch = -1)
    {
        ownerByCell = owners;
        rebuildCellsByRank(rankCount);
        if (nextEpoch >= 0)
            epoch = nextEpoch;
        else
            ++epoch;
    }

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

    int ownerOf(int globalCell) const
    {
        if (globalCell < 0 || globalCell >= (int)ownerByCell.size()) return -1;
        return ownerByCell[(std::size_t)globalCell];
    }
};

struct LocalGeometry3D
{
    std::vector<DsmcCell> cells;
    std::vector<DsmcEdge> edges;
    std::vector<int> local_cells;
    std::vector<int> face_gids;
    std::unordered_map<int, int> gid2local;
    std::unordered_map<int, int> face_gid2local;
    std::vector<double> localPointXY;
    std::vector<unsigned char> faceSplitTag;
    int ownedCount = 0;
};

class MeshPartitionTransfer3D
{
public:
    MeshPartitionTransfer3D(meshImport *mesh,
                            MeshparticalInitial *partinit,
                            MessagePassing *mpass,
                            const MpiContext &mpiCtx);

    bool broadcastMeshMessage(meshMessage &mess) const;
    bool initialPartitionAndDistribute(int haloRings);
    bool distributeGeometryByOwners(const PartitionState3D &state, int haloRings);
    LocalGeometry3D takeLocalGeometry();
    bool installLocalGeometryTo(ProcessDSMC &process);
    bool broadcastInitialParticleCounts(std::vector<int> &npcByCell) const;

private:
    struct LocalGeometryCounts3D
    {
        int ownedCellCount = 0;
        int ncell = 0;
        int nface = 0;
        int nxyz = 0;
    };

    struct MeshPartitionPackage3D
    {
        int ownedCellCount = 0;
        std::vector<int> cellGids;
        std::vector<DsmcCell> cells;
        std::vector<int> faceGids;
        std::vector<DsmcEdge> edges;
        std::vector<unsigned char> faceSplitTags;
        std::vector<int> nodeGids;
        std::vector<double> nodeXyz;
    };

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

    struct HaloBuildResult
    {
        std::vector<int> ownedCellGids;
        std::vector<int> haloCellGids;
        std::vector<int> ownedCellLayers;
        std::vector<int> haloCellLayers;
        std::vector<int> allCellGids;
        std::vector<int> allCellLayers;
        int ownedCount = 0;
    };

    bool isRoot() const;
    int worldFromCal(int calRank) const;

    bool initialPartitionRoot(std::vector<int> &rankCellAll) const;
    void syncDsmcCellOwners(const std::vector<int> &rankCellAll) const;
    void restoreOriginalMesh() const;

    void initializeCounts(RootScatterBuffers &buffers) const;
    void appendPackageToRootBuffers(int worldRank,
                                    const MeshPartitionPackage3D &package,
                                    RootScatterBuffers &buffers) const;
    void resizeLocalGeometryStorage(int myNcell,
                                    int myOwnedNcell,
                                    int myNface,
                                    int myNxyz) const;
    bool installPackageLocal(const MeshPartitionPackage3D &package) const;
    bool sendPackageToWorld(int worldRank,
                            const MeshPartitionPackage3D &package,
                            MPI_Datatype mpiCell,
                            MPI_Datatype mpiEdge) const;
    bool recvPackageFromRoot(MPI_Datatype mpiCell,
                             MPI_Datatype mpiEdge) const;
    bool buildRootBuffers(const PartitionState3D &state,
                          int haloRings,
                          RootScatterBuffers &buffers) const;

    HaloBuildResult buildOwnedHaloCells(int calRank,
                                        const PartitionState3D &state,
                                        int haloRings) const;
    bool buildPackageForRank(int calRank,
                             const PartitionState3D &state,
                             int haloRings,
                             MeshPartitionPackage3D &package) const;
    bool fillPackageCells(const HaloBuildResult &halo,
                          MeshPartitionPackage3D &package) const;
    bool fillPackageFaces(MeshPartitionPackage3D &package) const;
    bool fillPackageNodes(MeshPartitionPackage3D &package) const;
    void reportHaloAdjacencyMisses(int calRank,
                                   const PartitionState3D &state,
                                   int haloRings,
                                   const HaloBuildResult &halo) const;

    bool scatterCounts(const RootScatterBuffers &buffers,
                       int &myNcell,
                       int &myOwnedNcell,
                       int &myNface,
                       int &myNxyz) const;

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
