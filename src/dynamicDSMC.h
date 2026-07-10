/*
 * Dynamic DSMC repartitioning and migration interface.
 */

#pragma once
#include "meshImport.h"
#include "MessagePassing.h"
#include "ProcessDSMC.h"
using namespace std;

// dynamicDSMC stores state used by this module.
class dynamicDSMC
{
private:
    std::vector<particle> repartitionParticles;
// planRepartitionByLoad (nParts, cellLoadGlobal): updates partition ownership or load data.
    bool planRepartitionByLoad(idx_t nParts,
                               const std::vector<int>& cellLoadGlobal);
// syncDsmcCellOwners: updates partition ownership or load data.
    void syncDsmcCellOwners();
    void buildLineGraph(std::vector<idx_t>& xadj,
                        std::vector<idx_t>& adjncy,
                        std::vector<int>* edgeCell = nullptr,
                        std::vector<int>* edgeSlot = nullptr) const;
// matchPartitions (graph): updates partition ownership or load data.
    std::vector<std::vector<int>> matchPartitions(
        const std::vector<std::vector<int>>& graph) const;
// findpath (x, nx, ny): performs one solver support operation.
    int findpath(int x, int nx, int ny);
// KM_match (graph): updates partition ownership or load data.
    std::vector<std::vector<int>> KM_match(const std::vector<std::vector<int>>& graph);
    bool migrateParticlesByOwner();
// migrateCellFieldsPacket: moves structured data through MPI.
    bool migrateCellFieldsPacket();
// rebuildParticleChainsAfterRepartition (particles): updates partition ownership or load data.
    bool rebuildParticleChainsAfterRepartition(const std::vector<particle>& particles);
    void rebuildDsmcAfterRepartition();
    mutable bool lineGraphCacheReady = false;
    mutable std::vector<idx_t> cachedLineGraphXadj;
    mutable std::vector<idx_t> cachedLineGraphAdjncy;
    mutable std::vector<int> cachedLineGraphEdgeCell;
    mutable std::vector<int> cachedLineGraphEdgeSlot;
public:
    meshImport *mesh = NULL;
    meshMessage mess;
    ProcessDSMC *DSMCprocess = NULL;
    MeshparticalInitial *partinit = NULL;
    int rank, size, c_rank,c_size;
    std::vector<int> lx, ly, match, slack, fa;
    std::vector<bool> visx, visy;
    std::vector<std::vector<int>> Graph;
    MPI_Comm comm;
    MPI_Comm calGroup;
    MessagePassing *mpass = NULL;
    const MpiContext* mpi = nullptr;
// dynamicDSMC (mesh, mpass, mess, partinit, DSMCprocess, mpiCtx): performs one solver support operation.
    dynamicDSMC(meshImport *mesh, MessagePassing *mpass, meshMessage mess , MeshparticalInitial *partinit, ProcessDSMC *DSMCprocess, const MpiContext& mpiCtx);
// ~dynamicDSMC: releases owned buffers and MPI helper state.
    ~dynamicDSMC();
// dynamic_rankload_distribute (nParts): updates partition ownership or load data.
    void dynamic_rankload_distribute(idx_t nParts);
    void compute_cell_load_local(vector<int>& cell_load);
// average_load_calculate: updates partition ownership or load data.
    void average_load_calculate();
};
