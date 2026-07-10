/*
 * Load calculation, repartition planning, particle migration, and DSMC rebuild.
 */

#include "dynamicDSMC.h"

#include "MeshPartitionTransfer3D.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

using namespace std;

namespace
{
    const int kFieldWidth = 77;
    constexpr int kMigrationHaloRings = 3;
    constexpr double kDynamicRepartitionImbalanceThreshold = 1.10;
/*
 * legacyCellLoadWeight: updates partition ownership or load data.
 * Params: process, globalCell; returns: index, count, owner, or status value.
 * Flow:
 *   - gather load/owner data.
 *   - build the new mapping.
 *   - refresh dependent local state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
    int legacyCellLoadWeight(const ProcessDSMC* process, int globalCell)
    {
        if (process == nullptr || globalCell < 0)
            return 0;
        int pnum = process->particleCount(globalCell);
        if (pnum == 0)
            pnum = 1;
        return pnum;
    }
    struct ConstFieldDesc
    {
        const std::vector<double>* values;
        int stride;
    };
    struct FieldDesc
    {
        std::vector<double>* values;
        int stride;
    };
/*
 * packetWidth: moves structured data through MPI.
 * Params: fields; returns: index, count, owner, or status value.
 * Flow:
 *   - prepare counts or datatypes.
 *   - perform MPI transfer.
 *   - store received data.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
    int packetWidth(const std::vector<ConstFieldDesc>& fields)
    {
        int width = 0;
        for (const auto& f : fields)
            width += std::max(0, f.stride);
        return width;
    }
/*
 * packetWidth: moves structured data through MPI.
 * Params: fields; returns: index, count, owner, or status value.
 * Flow:
 *   - prepare counts or datatypes.
 *   - perform MPI transfer.
 *   - store received data.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
    int packetWidth(const std::vector<FieldDesc>& fields)
    {
        int width = 0;
        for (const auto& f : fields)
            width += std::max(0, f.stride);
        return width;
    }
/*
 * readScalar: parses mesh or configuration input.
 * Params: v, i; returns: computed scalar.
 * Flow:
 *   - decode the input record.
 *   - fill mesh arrays.
 *   - report parse status.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
    inline double readScalar(const std::vector<double> &v, int i)
    {
        return (i >= 0 && i < (int)v.size()) ? v[(std::size_t)i] : 0.0;
    }
/*
 * readStride: parses mesh or configuration input.
 * Params: v, i, stride, k; returns: computed scalar.
 * Flow:
 *   - decode the input record.
 *   - fill mesh arrays.
 *   - report parse status.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
    inline double readStride(const std::vector<double> &v, int i, int stride, int k)
    {
        const std::size_t pos = (std::size_t)i * (std::size_t)stride + (std::size_t)k;
        return (pos < v.size()) ? v[pos] : 0.0;
    }
/*
 * writeScalar: writes solver fields or diagnostics.
 * Params: v, i, value; returns: none.
 * Flow:
 *   - select fields.
 *   - format by mesh order.
 *   - flush output.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
    inline void writeScalar(std::vector<double> &v, int i, double value)
    {
        if (i >= 0 && i < (int)v.size()) v[(std::size_t)i] = value;
    }
/*
 * writeStride: writes solver fields or diagnostics.
 * Params: v, i, stride, k, value; returns: none.
 * Flow:
 *   - select fields.
 *   - format by mesh order.
 *   - flush output.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
    inline void writeStride(std::vector<double> &v, int i, int stride, int k, double value)
    {
        const std::size_t pos = (std::size_t)i * (std::size_t)stride + (std::size_t)k;
        if (pos < v.size()) v[pos] = value;
    }
/*
 * packScalar: moves structured data through MPI.
 * Params: src, i, dst, base, pos; returns: none.
 * Flow:
 *   - prepare counts or datatypes.
 *   - perform MPI transfer.
 *   - store received data.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
    inline void packScalar(const std::vector<double> &src, int i, std::vector<double> &dst, std::size_t base, int &pos)
    {
        dst[base + (std::size_t)pos++] = readScalar(src, i);
    }
/*
 * packStride: moves structured data through MPI.
 * Params: src, i, stride, dst, base, pos; returns: none.
 * Flow:
 *   - prepare counts or datatypes.
 *   - perform MPI transfer.
 *   - store received data.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
    inline void packStride(const std::vector<double> &src, int i, int stride, std::vector<double> &dst, std::size_t base, int &pos)
    {
        for (int k = 0; k < stride; ++k)
            dst[base + (std::size_t)pos++] = readStride(src, i, stride, k);
    }
/*
 * unpackScalar: moves structured data through MPI.
 * Params: dst, i, src, base, pos; returns: none.
 * Flow:
 *   - prepare counts or datatypes.
 *   - perform MPI transfer.
 *   - store received data.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
    inline void unpackScalar(std::vector<double> &dst, int i, const std::vector<double> &src, std::size_t base, int &pos)
    {
        writeScalar(dst, i, src[base + (std::size_t)pos++]);
    }
/*
 * unpackStride: moves structured data through MPI.
 * Params: dst, i, stride, src, base, pos; returns: none.
 * Flow:
 *   - prepare counts or datatypes.
 *   - perform MPI transfer.
 *   - store received data.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
    inline void unpackStride(std::vector<double> &dst, int i, int stride, const std::vector<double> &src, std::size_t base, int &pos)
    {
        for (int k = 0; k < stride; ++k)
            writeStride(dst, i, stride, k, src[base + (std::size_t)pos++]);
    }
/*
 * packFields: moves structured data through MPI.
 * Params: fields, srcCell, dst, base, pos; returns: none.
 * Flow:
 *   - prepare counts or datatypes.
 *   - perform MPI transfer.
 *   - store received data.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
    void packFields(const std::vector<ConstFieldDesc>& fields,
                    int srcCell,
                    std::vector<double>& dst,
                    std::size_t base,
                    int& pos)
    {
        for (const auto& f : fields)
        {
            if (f.values == nullptr || f.stride <= 0) continue;
            if (f.stride == 1)
                packScalar(*f.values, srcCell, dst, base, pos);
            else
                packStride(*f.values, srcCell, f.stride, dst, base, pos);
        }
    }
/*
 * unpackFields: moves structured data through MPI.
 * Params: fields, dstCell, src, base, pos; returns: none.
 * Flow:
 *   - prepare counts or datatypes.
 *   - perform MPI transfer.
 *   - store received data.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
    void unpackFields(const std::vector<FieldDesc>& fields,
                      int dstCell,
                      const std::vector<double>& src,
                      std::size_t base,
                      int& pos)
    {
        for (const auto& f : fields)
        {
            if (f.values == nullptr || f.stride <= 0) continue;
            if (f.stride == 1)
                unpackScalar(*f.values, dstCell, src, base, pos);
            else
                unpackStride(*f.values, dstCell, f.stride, src, base, pos);
        }
    }
/*
 * copyFieldValues: performs one solver support operation.
 * Params: srcFields, srcCell, dstFields, dstCell; returns: none.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
    void copyFieldValues(const std::vector<ConstFieldDesc>& srcFields,
                         int srcCell,
                         const std::vector<FieldDesc>& dstFields,
                         int dstCell)
    {
        const std::size_t n = std::min(srcFields.size(), dstFields.size());
        for (std::size_t idx = 0; idx < n; ++idx)
        {
            const ConstFieldDesc& src = srcFields[idx];
            const FieldDesc& dst = dstFields[idx];
            if (src.values == nullptr || dst.values == nullptr) continue;
            if (src.stride <= 0 || dst.stride <= 0) continue;
            const int stride = std::min(src.stride, dst.stride);
            if (stride == 1)
            {
                writeScalar(*dst.values, dstCell, readScalar(*src.values, srcCell));
            }
            else
            {
                for (int k = 0; k < stride; ++k)
                    writeStride(*dst.values, dstCell, dst.stride, k,
                                readStride(*src.values, srcCell, src.stride, k));
            }
        }
    }
}

/*
 * dynamicDSMC: initializes dynamicDSMC state.
 * Params: mesh, mpass, mess, partinit, DSMCprocess, mpiCtx; returns: none.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
dynamicDSMC::dynamicDSMC(meshImport *mesh,
                         MessagePassing *mpass,
                         meshMessage mess,
                         MeshparticalInitial *partinit,
                         ProcessDSMC *DSMCprocess,
                         const MpiContext& mpiCtx)
{
    this->mpi = &mpiCtx;
    this->mesh = mesh;
    this->mess = mess;
    this->partinit = partinit;
    this->DSMCprocess = DSMCprocess;
    this->mpass = mpass;
    this->comm = mpiCtx.comm;
    this->calGroup = mpiCtx.calGroup;
    this->rank = mpiCtx.rank;
    this->size = mpiCtx.size;
    this->c_rank = mpiCtx.c_rank;
    this->c_size = mpiCtx.c_size;
}

/*
 * ~dynamicDSMC: releases owned buffers and MPI helper state.
 * Params: none; returns: none.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
dynamicDSMC::~dynamicDSMC()
{
}

/*
 * dynamic_rankload_distribute: updates partition ownership or load data.
 * Params: nParts; returns: none.
 * Flow:
 *   - gather load/owner data.
 *   - build the new mapping.
 *   - refresh dependent local state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
void dynamicDSMC::dynamic_rankload_distribute(idx_t nParts)
{
    const auto nowSeconds = [this]() -> double
    {
        if (this->comm != MPI_COMM_NULL)
            return MPI_Wtime();
        return 0.0;
    };
    double stagePlan = 0.0;
    double stageParticleMigrate = 0.0;
    double stageGeometryDistribute = 0.0;
    double stageFieldMigrate = 0.0;
    double stageRebuild = 0.0;
    const double totalBegin = nowSeconds();
    const std::vector<int> previousOwners =
        (this->DSMCprocess != nullptr) ? this->DSMCprocess->rank_cell_all : std::vector<int>();
    const auto reportStageTimes = [&](const char* status, int partitionChangedFlag)
    {
        double stageLocal[6] = {
            stagePlan,
            stageParticleMigrate,
            stageGeometryDistribute,
            stageFieldMigrate,
            stageRebuild,
            nowSeconds() - totalBegin
        };
        double stageMax[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double stageSum[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        if (this->comm != MPI_COMM_NULL)
        {
            MPI_Reduce(stageLocal, stageMax, 6, MPI_DOUBLE, MPI_MAX, 0, this->comm);
            MPI_Reduce(stageLocal, stageSum, 6, MPI_DOUBLE, MPI_SUM, 0, this->comm);
        }
        else
        {
            for (int i = 0; i < 6; ++i)
            {
                stageMax[i] = stageLocal[i];
                stageSum[i] = stageLocal[i];
            }
        }
        const bool isRoot = (this->mpi == nullptr) ? true : this->mpi->root();
        if (!isRoot) return;
        const double denom = (this->size > 0) ? (double)this->size : 1.0;
        cout << "DYNAMIC_REPARTITION_TIMING_MAX"
             << " status=" << status
             << " changed=" << partitionChangedFlag
             << " plan=" << stageMax[0]
             << " particle_migrate=" << stageMax[1]
             << " geometry_distribute=" << stageMax[2]
             << " field_migrate=" << stageMax[3]
             << " rebuild=" << stageMax[4]
             << " total=" << stageMax[5]
             << endl;
        cout << "DYNAMIC_REPARTITION_TIMING_AVG"
             << " status=" << status
             << " changed=" << partitionChangedFlag
             << " plan=" << (stageSum[0] / denom)
             << " particle_migrate=" << (stageSum[1] / denom)
             << " geometry_distribute=" << (stageSum[2] / denom)
             << " field_migrate=" << (stageSum[3] / denom)
             << " rebuild=" << (stageSum[4] / denom)
             << " total=" << (stageSum[5] / denom)
             << endl;
    };
    const double planBegin = nowSeconds();
    vector<int> cellLoad(this->mess.Ncell, 0);
    compute_cell_load_local(cellLoad);
    int skipBalanced = 0;
    if (this->mpi != nullptr && this->mpi->root())
    {
        std::vector<long long> rankLoad((std::size_t)std::max(1, this->c_size), 0);
        const std::vector<int>* owners =
            (this->DSMCprocess != nullptr) ? &this->DSMCprocess->rank_cell_all : nullptr;
        const int ncell = (owners != nullptr)
            ? std::min((int)cellLoad.size(), (int)owners->size())
            : 0;
        for (int c = 0; c < ncell; ++c)
        {
            const int owner = (*owners)[(std::size_t)c];
            if (owner < 0 || owner >= this->c_size) continue;
            rankLoad[(std::size_t)owner] += std::max(0, cellLoad[(std::size_t)c]);
        }
        long long totalLoad = 0;
        long long maxLoad = 0;
        for (long long load : rankLoad)
        {
            totalLoad += load;
            if (load > maxLoad) maxLoad = load;
        }
        const double avgLoad = (this->c_size > 0) ? (double)totalLoad / (double)this->c_size : 0.0;
        const double imbalance = (avgLoad > 0.0) ? (double)maxLoad / avgLoad : 1.0;
        if (imbalance <= kDynamicRepartitionImbalanceThreshold)
            skipBalanced = 1;
    }
    if (this->comm != MPI_COMM_NULL)
        MPI_Bcast(&skipBalanced, 1, MPI_INT, 0, this->comm);
    if (skipBalanced != 0)
    {
        stagePlan = nowSeconds() - planBegin;
        reportStageTimes("skip_balanced", 0);
        return;
    }
    if (!planRepartitionByLoad(nParts, cellLoad))
    {
        stagePlan = nowSeconds() - planBegin;
        reportStageTimes("plan_fail", 0);
        if (this->mpi != nullptr && this->mpi->root())
        {
            cout << "MESH_PARTITION_DYNAMIC_PLAN_FAIL"
                 << " rank=" << this->rank << endl;
        }
        return;
    }
    stagePlan = nowSeconds() - planBegin;
    int partitionChanged = 1;
    if (this->DSMCprocess != nullptr)
    {
        const std::vector<int>& currentOwners = this->DSMCprocess->rank_cell_all;
        if (previousOwners.size() == currentOwners.size())
        {
            partitionChanged = 0;
            for (std::size_t i = 0; i < previousOwners.size(); ++i)
            {
                if (previousOwners[i] != currentOwners[i])
                {
                    partitionChanged = 1;
                    break;
                }
            }
        }
    }
    if (partitionChanged == 0)
    {
        reportStageTimes("skip_unchanged", 0);
        return;
    }
    const double particleBegin = nowSeconds();
    if (this->mpi->active())
    {
        const int oldOwned = std::min(this->DSMCprocess->iNcell,
                                      (int)this->DSMCprocess->local_cells.size());
        this->DSMCprocess->old_local_cell.clear();
        this->DSMCprocess->old_local_cell.reserve((std::size_t)oldOwned);
        for (int lc = 0; lc < oldOwned; ++lc)
        {
            const int gid = this->DSMCprocess->globalOfLocalCell(lc);
            if (gid >= 0)
                this->DSMCprocess->old_local_cell.push_back(gid);
        }
    }
    if (!migrateParticlesByOwner())
    {
        stageParticleMigrate = nowSeconds() - particleBegin;
        reportStageTimes("particle_migrate_fail", partitionChanged);
        cout << "PARTICLE_MIGRATION_DYNAMIC_FAIL"
             << " rank=" << this->rank << endl;
        return;
    }
    stageParticleMigrate = nowSeconds() - particleBegin;
    const double geometryBegin = nowSeconds();
    MeshPartitionTransfer3D transfer(this->mesh, this->partinit, this->mpass, *this->mpi);
    if (!transfer.distributeGeometryByOwners(this->DSMCprocess->partitionState,
                                             kMigrationHaloRings))
    {
        stageGeometryDistribute = nowSeconds() - geometryBegin;
        reportStageTimes("geometry_distribute_fail", partitionChanged);
        cout << "MESH_PARTITION_DYNAMIC_DISTRIBUTE_FAIL"
             << " rank=" << this->rank << endl;
        return;
    }
    if (this->DSMCprocess == nullptr || !transfer.installLocalGeometryTo(*this->DSMCprocess))
    {
        stageGeometryDistribute = nowSeconds() - geometryBegin;
        reportStageTimes("geometry_install_fail", partitionChanged);
        cout << "MESH_PARTITION_DYNAMIC_INSTALL_FAIL"
             << " rank=" << this->rank << endl;
        return;
    }
    stageGeometryDistribute = nowSeconds() - geometryBegin;
    const double fieldBegin = nowSeconds();
    if (!migrateCellFieldsPacket())
    {
        stageFieldMigrate = nowSeconds() - fieldBegin;
        reportStageTimes("field_migrate_fail", partitionChanged);
        cout << "CELL_FIELD_MIGRATION_FAIL"
             << " rank=" << this->rank << endl;
        return;
    }
    stageFieldMigrate = nowSeconds() - fieldBegin;
    const double rebuildBegin = nowSeconds();
    if (!rebuildParticleChainsAfterRepartition(this->repartitionParticles))
    {
        stageRebuild = nowSeconds() - rebuildBegin;
        reportStageTimes("rebuild_particles_fail", partitionChanged);
        cout << "PARTICLE_REBUILD_DYNAMIC_FAIL"
             << " rank=" << this->rank << endl;
        return;
    }
    rebuildDsmcAfterRepartition();
    average_load_calculate();
    stageRebuild = nowSeconds() - rebuildBegin;
    reportStageTimes("ok", partitionChanged);
}

/*
 * planRepartitionByLoad: updates partition ownership or load data.
 * Params: nParts, cellLoadGlobal; returns: success or decision flag.
 * Flow:
 *   - gather load/owner data.
 *   - build the new mapping.
 *   - refresh dependent local state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
bool dynamicDSMC::planRepartitionByLoad(idx_t nParts,
                                        const std::vector<int>& cellLoadGlobal)
{
    if (this->mpi == nullptr || this->DSMCprocess == nullptr) return false;
    if ((int)this->DSMCprocess->rank_cell_all.size() != this->DSMCprocess->mess.Ncell)
        this->DSMCprocess->rank_cell_all.assign((std::size_t)this->DSMCprocess->mess.Ncell, 0);
    PartitionState3D previousState;
    previousState.assign(this->DSMCprocess->rank_cell_all,
                         this->mpi->c_size,
                         this->DSMCprocess->partitionState.epoch);
    const int ncell = this->DSMCprocess->mess.Ncell;
    const std::size_t edgeFluxSlots = (std::size_t)ncell * (std::size_t)NN;
    int localHasFluxWeights = 0;
    if (this->DSMCprocess->enable_partition_flux_weights &&
        !this->DSMCprocess->partition_edge_flux.empty())
    {
        for (const auto& item : this->DSMCprocess->partition_edge_flux)
        {
            if (item.second > 0)
            {
                localHasFluxWeights = 1;
                break;
            }
        }
    }
    int globalHasFluxWeights = 0;
    MPI_Allreduce(&localHasFluxWeights,
                  &globalHasFluxWeights,
                  1,
                  MPI_INT,
                  MPI_LOR,
                  this->mpi->comm);
    std::vector<int> edgeFluxGlobal;
    if (globalHasFluxWeights != 0 && edgeFluxSlots <= (std::size_t)INT_MAX)
    {
        std::vector<int> edgeFluxLocal(edgeFluxSlots, 0);
        if (this->DSMCprocess->enable_partition_flux_weights &&
            !this->DSMCprocess->partition_edge_flux.empty())
        {
            for (const auto& item : this->DSMCprocess->partition_edge_flux)
            {
                if (item.first < edgeFluxLocal.size())
                    edgeFluxLocal[item.first] = item.second;
            }
        }
        if (this->mpi->root())
        {
            edgeFluxGlobal.assign(edgeFluxSlots, 0);
            MPI_Reduce(edgeFluxLocal.data(),
                       edgeFluxGlobal.data(),
                       (int)edgeFluxSlots,
                       MPI_INT,
                       MPI_SUM,
                       0,
                       this->mpi->comm);
        }
        else
        {
            MPI_Reduce(edgeFluxLocal.data(),
                       nullptr,
                       (int)edgeFluxSlots,
                       MPI_INT,
                       MPI_SUM,
                       0,
                       this->mpi->comm);
        }
    }
    bool ok = true;
    if (this->mpi->root())
    {
        ok = (this->mesh != nullptr);
        if (ok)
        {
            if (nParts <= 1)
            {
                this->DSMCprocess->rank_cell_all = previousState.ownerByCell;
                if ((int)this->DSMCprocess->rank_cell_all.size() != this->DSMCprocess->mess.Ncell)
                    this->DSMCprocess->rank_cell_all.assign((std::size_t)this->DSMCprocess->mess.Ncell, 0);
                syncDsmcCellOwners();
            }
            else
            {
                std::vector<idx_t> xadj;
                std::vector<idx_t> adjncy;
                std::vector<int> edgeCell;
                std::vector<int> edgeSlot;
                buildLineGraph(xadj, adjncy, &edgeCell, &edgeSlot);
                idx_t nvtxs = this->DSMCprocess->mess.Ncell;
                idx_t ncon = 1;
                std::vector<idx_t> weights((std::size_t)this->DSMCprocess->mess.Ncell, 1);
                for (int c = 0; c < this->DSMCprocess->mess.Ncell; ++c)
                {
                    int w = (c < (int)cellLoadGlobal.size()) ? cellLoadGlobal[(std::size_t)c] : 1;
                    if (w <= 0) w = 1;
                    weights[(std::size_t)c] = (idx_t)w;
                }
                std::vector<idx_t> adjwgt;
                bool useFluxWeights = (!edgeFluxGlobal.empty() &&
                                       edgeCell.size() == adjncy.size() &&
                                       edgeSlot.size() == adjncy.size());
                if (useFluxWeights)
                {
                    adjwgt.assign(adjncy.size(), 1);
                    for (std::size_t pos = 0; pos < adjncy.size(); ++pos)
                    {
                        const int oldCell = edgeCell[pos];
                        const int slot = edgeSlot[pos];
                        const int adjOld = (int)adjncy[pos];
                        int w1 = 0;
                        const std::size_t idx1 = (std::size_t)oldCell * (std::size_t)NN + (std::size_t)slot;
                        if (oldCell >= 0 && slot >= 0 && idx1 < edgeFluxGlobal.size())
                            w1 = edgeFluxGlobal[idx1];
                        int w2 = 0;
                        if (adjOld >= 0 &&
                            adjOld < this->DSMCprocess->mess.Ncell &&
                            adjOld < (int)this->mesh->Dsmccells.size())
                        {
                            const DsmcCell& nbCell = this->mesh->Dsmccells[(std::size_t)adjOld];
                            const int nbTag = nbCell.num;
                            for (int k = 0; k < nbTag && k < NN; ++k)
                            {
                                if (nbCell.cell2cell[k] == oldCell)
                                {
                                    const std::size_t idx2 = (std::size_t)adjOld * (std::size_t)NN + (std::size_t)k;
                                    if (idx2 < edgeFluxGlobal.size())
                                        w2 = edgeFluxGlobal[idx2];
                                    break;
                                }
                            }
                        }
                        long long wsum = (long long)w1 + (long long)w2;
                        if (wsum <= 0) wsum = 1;
                        const long long wmax = (long long)std::numeric_limits<idx_t>::max();
                        if (wsum > wmax) wsum = wmax;
                        adjwgt[pos] = (idx_t)wsum;
                    }
                }
                std::vector<idx_t> part((std::size_t)this->DSMCprocess->mess.Ncell, 0);
                idx_t objval = 0;
                idx_t options[METIS_NOPTIONS];
                METIS_SetDefaultOptions(options);
                options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
                options[METIS_OPTION_CTYPE] = METIS_CTYPE_SHEM;
                options[METIS_OPTION_UFACTOR] = 105;
                options[METIS_OPTION_CONTIG] = 1;
                options[METIS_OPTION_CCORDER] = 1;
                options[METIS_OPTION_SEED] = 1;
                const int ret = METIS_PartGraphRecursive(&nvtxs, &ncon,
                                                         xadj.data(), adjncy.data(),
                                                         weights.data(), NULL,
                                                         useFluxWeights ? adjwgt.data() : NULL,
                                                         &nParts, NULL, NULL,
                                                         options, &objval, part.data());
                if (ret != METIS_OK)
                {
                    cout << "MESH_PARTITION_METIS_FAIL ret=" << ret << endl;
                    ok = false;
                }
                else
                {
                    std::vector<std::vector<int>> matchData((std::size_t)nParts,
                                                            std::vector<int>((std::size_t)nParts, 0));
                    for (int c = 0; c < this->DSMCprocess->mess.Ncell; ++c)
                    {
                        const int newRank = (int)part[(std::size_t)c];
                        const int oldRank = previousState.ownerOf(c);
                        if (newRank >= 0 && newRank < (int)nParts &&
                            oldRank >= 0 && oldRank < (int)nParts)
                            matchData[(std::size_t)newRank][(std::size_t)oldRank] += 1;
                    }
                    const std::vector<std::vector<int>> assignments = matchPartitions(matchData);
                    std::vector<int> remapNewToOld((std::size_t)nParts, 0);
                    for (int r = 0; r < (int)nParts; ++r)
                        remapNewToOld[(std::size_t)r] = r;
                    for (std::size_t i = 0; i < assignments.size(); ++i)
                    {
                        if (assignments[i].size() < 2) continue;
                        const int newRank = assignments[i][0];
                        const int oldRank = assignments[i][1];
                        if (newRank >= 0 && newRank < (int)nParts &&
                            oldRank >= 0 && oldRank < (int)nParts)
                            remapNewToOld[(std::size_t)newRank] = oldRank;
                    }
                    for (int c = 0; c < this->DSMCprocess->mess.Ncell; ++c)
                    {
                        const int newRank = (int)part[(std::size_t)c];
                        if (newRank >= 0 && newRank < (int)nParts)
                            part[(std::size_t)c] = (idx_t)remapNewToOld[(std::size_t)newRank];
                    }
                    this->DSMCprocess->rank_cell_all.assign((std::size_t)this->DSMCprocess->mess.Ncell, 0);
                    for (int i = 0; i < this->DSMCprocess->mess.Ncell; ++i)
                        this->DSMCprocess->rank_cell_all[(std::size_t)i] = (int)part[(std::size_t)i];
                    syncDsmcCellOwners();
                }
            }
        }
    }
    int okInt = ok ? 1 : 0;
    MPI_Bcast(&okInt, 1, MPI_INT, 0, this->mpi->comm);
    if (!okInt) return false;
    MPI_Bcast(this->DSMCprocess->rank_cell_all.data(),
              this->DSMCprocess->mess.Ncell,
              MPI_INT,
              0,
              this->mpi->comm);
    this->DSMCprocess->reset_partition_flux_counts();
    this->DSMCprocess->partitionState.assign(this->DSMCprocess->rank_cell_all,
                                             this->mpi->c_size,
                                             previousState.epoch + 1);
    this->partinit->partitionState = this->DSMCprocess->partitionState;
    this->partinit->rank_cell_all = this->DSMCprocess->rank_cell_all;
    return true;
}

/*
 * syncDsmcCellOwners: updates partition ownership or load data.
 * Params: none; returns: none.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
void dynamicDSMC::syncDsmcCellOwners()
{
    if (this->mesh == nullptr || this->DSMCprocess == nullptr) return;
    const int n = std::min((int)this->mesh->Dsmccells.size(),
                           (int)this->DSMCprocess->rank_cell_all.size());
    for (int i = 0; i < n; ++i)
        this->mesh->Dsmccells[(std::size_t)i].no = this->DSMCprocess->rank_cell_all[(std::size_t)i];
}

/*
 * buildLineGraph: performs one solver support operation.
 * Params: xadj, adjncy, edgeCell, edgeSlot; returns: none.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
void dynamicDSMC::buildLineGraph(std::vector<idx_t>& xadj,
                                 std::vector<idx_t>& adjncy,
                                 std::vector<int>* edgeCell,
                                 std::vector<int>* edgeSlot) const
{
    if (lineGraphCacheReady)
    {
        xadj = cachedLineGraphXadj;
        adjncy = cachedLineGraphAdjncy;
        if (edgeCell != nullptr) *edgeCell = cachedLineGraphEdgeCell;
        if (edgeSlot != nullptr) *edgeSlot = cachedLineGraphEdgeSlot;
        return;
    }
    int edgeCount = 0;
    for (int i = 0; i < this->DSMCprocess->mess.Ncell; ++i)
    {
        const int faceCount = (i < (int)this->mesh->Dsmccells.size())
            ? this->mesh->Dsmccells[(std::size_t)i].num
            : 0;
        for (int j = 0; j < faceCount && j < NN; ++j)
        {
            const int neighbor = this->mesh->Dsmccells[(std::size_t)i].cell2cell[j];
            if (neighbor < 0 || neighbor >= this->DSMCprocess->mess.Ncell) continue;
            ++edgeCount;
        }
    }
    xadj.assign((std::size_t)this->DSMCprocess->mess.Ncell + 1u, 0);
    adjncy.assign((std::size_t)edgeCount, 0);
    std::vector<int> builtEdgeCell((std::size_t)edgeCount, -1);
    std::vector<int> builtEdgeSlot((std::size_t)edgeCount, -1);
    int cursor = 0;
    for (int i = 0; i < this->DSMCprocess->mess.Ncell; ++i)
    {
        const int faceCount = (i < (int)this->mesh->Dsmccells.size())
            ? this->mesh->Dsmccells[(std::size_t)i].num
            : 0;
        for (int j = 0; j < faceCount && j < NN; ++j)
        {
            const int neighbor = this->mesh->Dsmccells[(std::size_t)i].cell2cell[j];
            if (neighbor < 0 || neighbor >= this->DSMCprocess->mess.Ncell) continue;
            builtEdgeCell[(std::size_t)cursor] = i;
            builtEdgeSlot[(std::size_t)cursor] = j;
            adjncy[(std::size_t)cursor++] = neighbor;
        }
        xadj[(std::size_t)i + 1u] = cursor;
    }
    cachedLineGraphXadj = xadj;
    cachedLineGraphAdjncy = adjncy;
    cachedLineGraphEdgeCell = builtEdgeCell;
    cachedLineGraphEdgeSlot = builtEdgeSlot;
    if (edgeCell != nullptr) *edgeCell = builtEdgeCell;
    if (edgeSlot != nullptr) *edgeSlot = builtEdgeSlot;
    lineGraphCacheReady = true;
}

/*
 * matchPartitions: updates partition ownership or load data.
 * Params: graph; returns: computed container.
 * Flow:
 *   - gather load/owner data.
 *   - build the new mapping.
 *   - refresh dependent local state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
std::vector<std::vector<int>> dynamicDSMC::matchPartitions(
    const std::vector<std::vector<int>>& graph) const
{
    return const_cast<dynamicDSMC*>(this)->KM_match(graph);
}

/*
 * findpath: performs one solver support operation.
 * Params: x, nx, ny; returns: index, count, owner, or status value.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
int dynamicDSMC::findpath(int x, int nx, int ny)
{
    visx[(std::size_t)x] = true;
    for (int y = 0; y < ny; ++y)
    {
        if (visy[(std::size_t)y]) continue;
        const int tempDelta = this->lx[(std::size_t)x] + this->ly[(std::size_t)y] - this->Graph[(std::size_t)x][(std::size_t)y];
        if (tempDelta == 0)
        {
            this->visy[(std::size_t)y] = true;
            this->fa[(std::size_t)(y + nx)] = x;
            if (this->match[(std::size_t)y] == -1)
                return y + nx;
            this->fa[(std::size_t)this->match[(std::size_t)y]] = y + nx;
            const int res = findpath(this->match[(std::size_t)y], nx, ny);
            if (res > 0) return res;
        }
        else if (this->slack[(std::size_t)x] > tempDelta)
            this->slack[(std::size_t)x] = tempDelta;
    }
    return -1;
}

/*
 * KM_match: updates partition ownership or load data.
 * Params: graph; returns: computed container.
 * Flow:
 *   - gather load/owner data.
 *   - build the new mapping.
 *   - refresh dependent local state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
std::vector<std::vector<int>> dynamicDSMC::KM_match(const std::vector<std::vector<int>>& graph)
{
    const int n = (int)graph.size();
    const int nx = n;
    const int ny = n;
    this->Graph = graph;
    this->lx.assign((std::size_t)n, 0);
    this->ly.assign((std::size_t)n, 0);
    this->match.assign((std::size_t)n, -1);
    this->slack.assign((std::size_t)n, INT_MAX);
    this->fa.assign((std::size_t)(nx + ny), -1);
    this->visx.assign((std::size_t)n, false);
    this->visy.assign((std::size_t)n, false);
    for (int x = 0; x < n; ++x)
        this->lx[(std::size_t)x] = graph[(std::size_t)x].empty() ? 0 : *std::max_element(graph[(std::size_t)x].begin(), graph[(std::size_t)x].end());
    for (int x = 0; x < nx; ++x)
    {
        std::fill(this->slack.begin(), this->slack.end(), INT_MAX);
        std::fill(this->fa.begin(), this->fa.end(), -1);
        std::fill(this->visx.begin(), this->visx.end(), false);
        std::fill(this->visy.begin(), this->visy.end(), false);
        int first = 1;
        int leaf = -1;
        int totalTimes = 0;
        while (true)
        {
            ++totalTimes;
            if (first == 1)
            {
                leaf = findpath(x, nx, ny);
                first = 0;
            }
            else
            {
                for (int i = 0; i < nx; ++i)
                {
                    if (this->slack[(std::size_t)i] == 0)
                    {
                        this->slack[(std::size_t)i] = INT_MAX;
                        leaf = findpath(i, nx, ny);
                        if (leaf > 0) break;
                    }
                }
            }
            if (leaf > 0)
            {
                int p = leaf;
                while (p > 0)
                {
                    this->match[(std::size_t)(p - nx)] = this->fa[(std::size_t)p];
                    p = this->fa[(std::size_t)this->fa[(std::size_t)p]];
                }
                break;
            }
            int delta = INT_MAX;
            for (int i = 0; i < nx; ++i)
            {
                if (this->visx[(std::size_t)i] && delta > this->slack[(std::size_t)i])
                    delta = this->slack[(std::size_t)i];
            }
            for (int i = 0; i < nx; ++i)
            {
                if (this->visx[(std::size_t)i])
                {
                    this->lx[(std::size_t)i] -= delta;
                    this->slack[(std::size_t)i] -= delta;
                }
            }
            for (int j = 0; j < ny; ++j)
            {
                if (this->visy[(std::size_t)j])
                    this->ly[(std::size_t)j] += delta;
            }
            if (totalTimes > 10000)
            {
                cout << "KM is error" << endl;
                break;
            }
        }
    }
    std::vector<std::vector<int>> result;
    for (int y = 0; y < n; ++y)
    {
        if (this->match[(std::size_t)y] != -1)
            result.push_back({this->match[(std::size_t)y], y});
    }
    this->lx.clear();
    this->ly.clear();
    this->match.clear();
    this->slack.clear();
    this->fa.clear();
    this->visx.clear();
    this->visy.clear();
    this->Graph.clear();
    return result;
}

/*
 * migrateParticlesByOwner: updates partition ownership or load data.
 * Params: none; returns: success or decision flag.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
bool dynamicDSMC::migrateParticlesByOwner()
{
    this->repartitionParticles.clear();
    if (this->mpi == nullptr || !this->mpi->active()) return true;
    if (this->mpass == nullptr || this->partinit == nullptr || this->DSMCprocess == nullptr) return false;
    const int nrank = this->mpi->c_size;
    std::vector<std::vector<particle>> sendBins((std::size_t)nrank);
    std::vector<std::vector<particle>> recvBins;
    std::vector<int> sendParticleCounts((std::size_t)nrank, 0);
    int retainedParticleCount = 0;
    for (int gid : this->DSMCprocess->old_local_cell)
    {
        if (gid < 0 || gid >= this->mess.Ncell) continue;
        const int dest = this->DSMCprocess->ownerOfGlobalCell(gid);
        if (dest < 0 || dest >= nrank)
        {
            cout << "PARTICLE_MIGRATION_BAD_OWNER rank=" << this->mpi->c_rank
                 << " gid=" << gid
                 << " owner=" << dest << endl;
            return false;
        }
        const int bucketSize = (int)this->DSMCprocess->currParticles(gid).size();
        if (dest == this->mpi->c_rank)
            retainedParticleCount += bucketSize;
        else
            sendParticleCounts[(std::size_t)dest] += bucketSize;
    }
    if (retainedParticleCount > 0)
        this->repartitionParticles.reserve((std::size_t)retainedParticleCount);
    for (int r = 0; r < nrank; ++r)
        if (sendParticleCounts[(std::size_t)r] > 0)
            sendBins[(std::size_t)r].reserve((std::size_t)sendParticleCounts[(std::size_t)r]);
    for (int gid : this->DSMCprocess->old_local_cell)
    {
        if (gid < 0 || gid >= this->mess.Ncell) continue;
        const int dest = this->DSMCprocess->ownerOfGlobalCell(gid);
        if (dest < 0 || dest >= nrank)
            return false;
        ParticleBucketSoA& bucket = this->DSMCprocess->currParticles(gid);
        const std::size_t bucketSize = bucket.size();
        for (std::size_t i = 0; i < bucketSize; ++i)
        {
            particle p = bucket.get(i);
            p.p_rank_serial = dest;
            p.p_mesh_serial = gid;
            if (dest == this->mpi->c_rank)
                this->repartitionParticles.push_back(p);
            else
                sendBins[(std::size_t)dest].push_back(p);
        }
        if (gid >= 0 && gid < (int)this->partinit->cell_particle_reserve_hint.size())
        {
            this->partinit->cell_particle_reserve_hint[(std::size_t)gid] =
                std::max(this->partinit->cell_particle_reserve_hint[(std::size_t)gid],
                         (int)bucketSize);
        }
        bucket.clear();
    }
    if (!this->mpass->exchangeParticleVectors(sendBins, recvBins, *this->mpi, 3101))
        return false;
    std::size_t recvTotal = 0;
    for (int r = 0; r < nrank; ++r)
        recvTotal += recvBins[(std::size_t)r].size();
    this->repartitionParticles.reserve(this->repartitionParticles.size() + recvTotal);
    for (int r = 0; r < nrank; ++r)
        this->repartitionParticles.insert(this->repartitionParticles.end(),
                                          recvBins[(std::size_t)r].begin(),
                                          recvBins[(std::size_t)r].end());
    return true;
}

/*
 * migrateCellFieldsPacket: moves structured data through MPI.
 * Params: none; returns: success or decision flag.
 * Flow:
 *   - prepare counts or datatypes.
 *   - perform MPI transfer.
 *   - store received data.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
bool dynamicDSMC::migrateCellFieldsPacket()
{
    if (this->mpi == nullptr || !this->mpi->active()) return true;
    if (this->partinit == nullptr || this->DSMCprocess == nullptr) return false;
    const int nrank = this->mpi->c_size;
    const std::vector<int> &oldGids = this->DSMCprocess->old_local_cell;
    std::vector<double> sparseStateField((std::size_t)oldGids.size(), 0.0);
    std::vector<double> sparseAccumStepsField((std::size_t)oldGids.size(), 0.0);
    for (int oldLocal = 0; oldLocal < (int)oldGids.size(); ++oldLocal)
    {
        if (oldLocal < (int)this->DSMCprocess->dsmc2ns_sparse_state.size())
            sparseStateField[(std::size_t)oldLocal] =
                (double)this->DSMCprocess->dsmc2ns_sparse_state[(std::size_t)oldLocal];
        if (oldLocal < (int)this->DSMCprocess->dsmc2ns_sparse_accum_steps.size())
            sparseAccumStepsField[(std::size_t)oldLocal] =
                (double)this->DSMCprocess->dsmc2ns_sparse_accum_steps[(std::size_t)oldLocal];
    }
    const std::vector<ConstFieldDesc> sendFields = {
        {&this->partinit->crmax, 1},
        {&this->partinit->remainderincoll, 1},
        {&this->partinit->remainderinpre, 1},
        {&this->DSMCprocess->steady_rho, 1},
        {&this->DSMCprocess->step_rho, 1},
        {&this->DSMCprocess->stepinter_rho, 1},
        {&this->DSMCprocess->stepsum_rho, 1},
        {&this->DSMCprocess->steady_T, 1},
        {&this->DSMCprocess->step_T, 1},
        {&this->DSMCprocess->stepinter_T, 1},
        {&this->DSMCprocess->stepsum_T, 1},
        {&this->DSMCprocess->steady_U, 3},
        {&this->DSMCprocess->step_U, 3},
        {&this->DSMCprocess->stepinter_U, 3},
        {&this->DSMCprocess->stepsum_U, 3},
        {&this->DSMCprocess->steady_q, 3},
        {&this->DSMCprocess->step_q, 3},
        {&this->DSMCprocess->stepinter_q, 3},
        {&this->DSMCprocess->stepsum_q, 3},
        {&this->DSMCprocess->steady_sigma, 6},
        {&this->DSMCprocess->step_sigma, 6},
        {&this->DSMCprocess->stepinter_sigma, 6},
        {&this->DSMCprocess->stepsum_sigma, 6},
        {&this->DSMCprocess->steady_qr, 4},
        {&this->DSMCprocess->step_qr, 4},
        {&this->DSMCprocess->stepinter_qr, 4},
        {&this->DSMCprocess->stepsum_qr, 4},
        {&sparseStateField, 1},
        {&sparseAccumStepsField, 1}
    };
    const int fieldWidth = packetWidth(sendFields);
    if (fieldWidth != kFieldWidth)
    {
        cout << "CELL_FIELD_WIDTH_MISMATCH width=" << fieldWidth
             << " expected=" << kFieldWidth << endl;
        return false;
    }
    const int nnew = this->DSMCprocess->iNcell;
    std::vector<double> crmaxNew((std::size_t)nnew, 0.0);
    std::vector<double> remColNew((std::size_t)nnew, 0.0);
    std::vector<double> remPreNew((std::size_t)nnew, 0.0);
    std::vector<double> steadyRhoNew((std::size_t)nnew, 0.0), stepRhoNew((std::size_t)nnew, 0.0);
    std::vector<double> stepInterRhoNew((std::size_t)nnew, 0.0), stepSumRhoNew((std::size_t)nnew, 0.0);
    std::vector<double> steadyTNew((std::size_t)nnew, 0.0), stepTNew((std::size_t)nnew, 0.0);
    std::vector<double> stepInterTNew((std::size_t)nnew, 0.0), stepSumTNew((std::size_t)nnew, 0.0);
    std::vector<double> steadyUNew((std::size_t)nnew * 3u, 0.0), stepUNew((std::size_t)nnew * 3u, 0.0);
    std::vector<double> stepInterUNew((std::size_t)nnew * 3u, 0.0), stepSumUNew((std::size_t)nnew * 3u, 0.0);
    std::vector<double> steadyQNew((std::size_t)nnew * 3u, 0.0), stepQNew((std::size_t)nnew * 3u, 0.0);
    std::vector<double> stepInterQNew((std::size_t)nnew * 3u, 0.0), stepSumQNew((std::size_t)nnew * 3u, 0.0);
    std::vector<double> steadySigmaNew((std::size_t)nnew * 6u, 0.0), stepSigmaNew((std::size_t)nnew * 6u, 0.0);
    std::vector<double> stepInterSigmaNew((std::size_t)nnew * 6u, 0.0), stepSumSigmaNew((std::size_t)nnew * 6u, 0.0);
    std::vector<double> steadyQrNew((std::size_t)nnew * 4u, 0.0), stepQrNew((std::size_t)nnew * 4u, 0.0);
    std::vector<double> stepInterQrNew((std::size_t)nnew * 4u, 0.0), stepSumQrNew((std::size_t)nnew * 4u, 0.0);
    std::vector<double> sparseStateNewField((std::size_t)nnew, 0.0);
    std::vector<double> sparseAccumStepsNewField((std::size_t)nnew, 0.0);
    const std::vector<FieldDesc> recvFields = {
        {&crmaxNew, 1},
        {&remColNew, 1},
        {&remPreNew, 1},
        {&steadyRhoNew, 1},
        {&stepRhoNew, 1},
        {&stepInterRhoNew, 1},
        {&stepSumRhoNew, 1},
        {&steadyTNew, 1},
        {&stepTNew, 1},
        {&stepInterTNew, 1},
        {&stepSumTNew, 1},
        {&steadyUNew, 3},
        {&stepUNew, 3},
        {&stepInterUNew, 3},
        {&stepSumUNew, 3},
        {&steadyQNew, 3},
        {&stepQNew, 3},
        {&stepInterQNew, 3},
        {&stepSumQNew, 3},
        {&steadySigmaNew, 6},
        {&stepSigmaNew, 6},
        {&stepInterSigmaNew, 6},
        {&stepSumSigmaNew, 6},
        {&steadyQrNew, 4},
        {&stepQrNew, 4},
        {&stepInterQrNew, 4},
        {&stepSumQrNew, 4},
        {&sparseStateNewField, 1},
        {&sparseAccumStepsNewField, 1}
    };
    if (packetWidth(recvFields) != fieldWidth)
        return false;
    std::vector<int> sendCounts((std::size_t)nrank, 0);
    for (int oldLocal = 0; oldLocal < (int)oldGids.size(); ++oldLocal)
    {
        const int gid = oldGids[(std::size_t)oldLocal];
        const int dst = this->DSMCprocess->ownerOfGlobalCell(gid);
        if (dst < 0 || dst >= nrank)
        {
            cout << "CELL_FIELD_MIGRATION_BAD_OWNER rank=" << this->mpi->c_rank
                 << " gid=" << gid
                 << " owner=" << dst << endl;
            return false;
        }
        if (dst == this->mpi->c_rank)
        {
            const int lc = this->DSMCprocess->localOfGlobalCell(gid);
            if (lc < 0 || lc >= nnew)
            {
                cout << "CELL_FIELD_MIGRATION_BAD_LOCAL rank=" << this->mpi->c_rank
                     << " gid=" << gid
                     << " local=" << lc << endl;
                return false;
            }
            copyFieldValues(sendFields, oldLocal, recvFields, lc);
        }
        else
        {
            ++sendCounts[(std::size_t)dst];
        }
    }
    std::vector<int> sdispls;
    int sendTotal = 0;
    sdispls.assign((std::size_t)nrank, 0);
    for (int r = 0; r < nrank; ++r)
    {
        sdispls[(std::size_t)r] = sendTotal;
        sendTotal += sendCounts[(std::size_t)r];
    }
    std::vector<int> sendGids((std::size_t)sendTotal, 0);
    std::vector<double> sendValues((std::size_t)sendTotal * (std::size_t)fieldWidth, 0.0);
    std::vector<int> cursor = sdispls;
    for (int oldLocal = 0; oldLocal < (int)oldGids.size(); ++oldLocal)
    {
        const int gid = oldGids[(std::size_t)oldLocal];
        const int dst = this->DSMCprocess->ownerOfGlobalCell(gid);
        if (dst == this->mpi->c_rank)
            continue;
        if (dst < 0 || dst >= nrank)
            return false;
        const int slot = cursor[(std::size_t)dst]++;
        sendGids[(std::size_t)slot] = gid;
        const std::size_t base = (std::size_t)slot * (std::size_t)fieldWidth;
        int pos = 0;
        packFields(sendFields, oldLocal, sendValues, base, pos);
    }
    std::vector<int> recvGids;
    std::vector<double> recvValues;
    if (this->mpass == nullptr ||
        !this->mpass->exchangeFixedWidthDoublePackets(sendCounts,
                                                      sendGids,
                                                      sendValues,
                                                      fieldWidth,
                                                      recvGids,
                                                      recvValues,
                                                      *this->mpi))
        return false;
    const int recvTotal = (int)recvGids.size();
    for (int j = 0; j < recvTotal; ++j)
    {
        const int gid = recvGids[(std::size_t)j];
        const int lc = this->DSMCprocess->localOfGlobalCell(gid);
        if (lc < 0 || lc >= nnew) continue;
        const std::size_t base = (std::size_t)j * (std::size_t)fieldWidth;
        int pos = 0;
        unpackFields(recvFields, lc, recvValues, base, pos);
    }
    this->partinit->crmax.swap(crmaxNew);
    this->partinit->remainderincoll.swap(remColNew);
    this->partinit->remainderinpre.swap(remPreNew);
    this->DSMCprocess->steady_rho.swap(steadyRhoNew);
    this->DSMCprocess->step_rho.swap(stepRhoNew);
    this->DSMCprocess->stepinter_rho.swap(stepInterRhoNew);
    this->DSMCprocess->stepsum_rho.swap(stepSumRhoNew);
    this->DSMCprocess->steady_T.swap(steadyTNew);
    this->DSMCprocess->step_T.swap(stepTNew);
    this->DSMCprocess->stepinter_T.swap(stepInterTNew);
    this->DSMCprocess->stepsum_T.swap(stepSumTNew);
    this->DSMCprocess->steady_U.swap(steadyUNew);
    this->DSMCprocess->step_U.swap(stepUNew);
    this->DSMCprocess->stepinter_U.swap(stepInterUNew);
    this->DSMCprocess->stepsum_U.swap(stepSumUNew);
    this->DSMCprocess->steady_q.swap(steadyQNew);
    this->DSMCprocess->step_q.swap(stepQNew);
    this->DSMCprocess->stepinter_q.swap(stepInterQNew);
    this->DSMCprocess->stepsum_q.swap(stepSumQNew);
    this->DSMCprocess->steady_sigma.swap(steadySigmaNew);
    this->DSMCprocess->step_sigma.swap(stepSigmaNew);
    this->DSMCprocess->stepinter_sigma.swap(stepInterSigmaNew);
    this->DSMCprocess->stepsum_sigma.swap(stepSumSigmaNew);
    this->DSMCprocess->steady_qr.swap(steadyQrNew);
    this->DSMCprocess->step_qr.swap(stepQrNew);
    this->DSMCprocess->stepinter_qr.swap(stepInterQrNew);
    this->DSMCprocess->stepsum_qr.swap(stepSumQrNew);
    std::vector<unsigned char> sparseStateNew(
        (std::size_t)nnew, ProcessDSMC::DSMC2NS_SPARSE_NORMAL);
    std::vector<int> sparseAccumStepsNew((std::size_t)nnew, 0);
    for (int i = 0; i < nnew; ++i)
    {
        int state = (int)std::lround(sparseStateNewField[(std::size_t)i]);
        if (state < ProcessDSMC::DSMC2NS_SPARSE_NORMAL ||
            state > ProcessDSMC::DSMC2NS_SPARSE_RELEASED)
            state = ProcessDSMC::DSMC2NS_SPARSE_NORMAL;
        int steps = (int)std::lround(sparseAccumStepsNewField[(std::size_t)i]);
        if (steps < 0) steps = 0;
        sparseStateNew[(std::size_t)i] = (unsigned char)state;
        sparseAccumStepsNew[(std::size_t)i] = steps;
    }
    this->DSMCprocess->dsmc2ns_sparse_state.swap(sparseStateNew);
    this->DSMCprocess->dsmc2ns_sparse_accum_steps.swap(sparseAccumStepsNew);
    return true;
}

/*
 * rebuildParticleChainsAfterRepartition: updates partition ownership or load data.
 * Params: particles; returns: success or decision flag.
 * Flow:
 *   - gather load/owner data.
 *   - build the new mapping.
 *   - refresh dependent local state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
bool dynamicDSMC::rebuildParticleChainsAfterRepartition(const std::vector<particle>& particles)
{
    if (this->mpi == nullptr || !this->mpi->active()) return true;
    if (this->partinit == nullptr || this->DSMCprocess == nullptr) return false;
    const int nnew = this->DSMCprocess->iNcell;
    this->partinit->cell_particles_curr.clear();
    this->partinit->cell_particles_curr.resize((std::size_t)nnew);
    std::vector<int> counts((std::size_t)nnew, 0);
    for (std::size_t i = 0; i < particles.size(); ++i)
    {
        const int gid = particles[i].p_mesh_serial;
        const int local = this->DSMCprocess->localOfGlobalCell(gid);
        if (local < 0 || local >= nnew)
        {
            cout << "PARTICLE_REBUILD_BAD_CELL rank=" << this->mpi->c_rank
                 << " gid=" << gid
                 << " local=" << local << endl;
            return false;
        }
        ++counts[(std::size_t)local];
    }
    this->DSMCprocess->clearNextParticleBuffers();
    for (int lc = 0; lc < nnew; ++lc)
    {
        ParticleBucketSoA& bucket =
            this->partinit->cell_particles_curr[(std::size_t)lc];
        const int want = counts[(std::size_t)lc];
        if (want > 0)
            bucket.reserve((std::size_t)want);
    }
    for (std::size_t i = 0; i < particles.size(); ++i)
    {
        particle p = particles[i];
        const int gid = p.p_mesh_serial;
        const int local = this->DSMCprocess->localOfGlobalCell(gid);
        if (local < 0 || local >= nnew)
            return false;
        ParticleBucketSoA& bucket =
            this->partinit->cell_particles_curr[(std::size_t)local];
        p.p_serial = (int)bucket.size();
        p.p_rank_serial = this->mpi->c_rank;
        p.p_mesh_serial = local;
        p.dt_left = 0.0;
        bucket.push_back(p);
    }
    for (int lc = 0; lc < nnew; ++lc)
    {
        const int gid = this->DSMCprocess->globalOfLocalCell(lc);
        if (gid < 0 || gid >= this->mess.Ncell) continue;
        ParticleBucketSoA& bucket =
            this->partinit->cell_particles_curr[(std::size_t)lc];
        const std::size_t bucketSize = bucket.size();
        for (std::size_t pi = 0; pi < bucketSize; ++pi)
        {
            particle part = bucket.get(pi);
            part.p_serial = (int)pi;
            part.p_rank_serial = this->mpi->c_rank;
            part.p_mesh_serial = lc;
            part.dt_left = 0.0;
            bucket.set(pi, part);
        }
        if (gid >= 0 && gid < (int)this->partinit->cell_particle_reserve_hint.size())
        {
            this->partinit->cell_particle_reserve_hint[(std::size_t)gid] =
                std::max(this->partinit->cell_particle_reserve_hint[(std::size_t)gid],
                         (int)bucketSize);
        }
    }
#ifdef CHECK_PARTICLE_BUCKETS
    this->DSMCprocess->checkParticleBucketConsistency("dynamic_repartition_rebuild");
#endif
    return true;
}

/*
 * compute_cell_load_local: updates partition ownership or load data.
 * Params: cell_load; returns: none.
 * Flow:
 *   - gather load/owner data.
 *   - build the new mapping.
 *   - refresh dependent local state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
void dynamicDSMC::compute_cell_load_local(vector<int>& cell_load)
{
    if (this->mpi == nullptr || this->DSMCprocess == nullptr || this->mess.Ncell <= 0) return;
    vector<int> particle_load((std::size_t)this->mess.Ncell, 0);
    std::vector<int> localCellIds;
    std::vector<int> localParticleLoads;
    if (this->mpi->active())
    {
        localCellIds.reserve((std::size_t)this->DSMCprocess->iNcell);
        localParticleLoads.reserve((std::size_t)this->DSMCprocess->iNcell);
        for (int i = 0; i < this->DSMCprocess->iNcell; i++)
        {
            int meshindex = this->DSMCprocess->globalOfLocalCell(i);
            if (meshindex < 0 || meshindex >= this->mess.Ncell) continue;
            localCellIds.push_back(meshindex);
            localParticleLoads.push_back(legacyCellLoadWeight(this->DSMCprocess, meshindex));
        }
    }
    const int localEntryCount = (int)localCellIds.size();
    std::vector<int> recvCounts;
    if (this->mpi->root())
        recvCounts.assign((std::size_t)this->size, 0);
    MPI_Gather(&localEntryCount,
               1,
               MPI_INT,
               this->mpi->root() ? recvCounts.data() : nullptr,
               1,
               MPI_INT,
               0,
               this->comm);
    std::vector<int> displs;
    int totalEntries = 0;
    if (this->mpi->root())
    {
        displs.assign((std::size_t)this->size, 0);
        for (int r = 0; r < this->size; ++r)
        {
            displs[(std::size_t)r] = totalEntries;
            totalEntries += recvCounts[(std::size_t)r];
        }
    }
    std::vector<int> allCellIds;
    std::vector<int> allParticleLoads;
    if (this->mpi->root())
    {
        allCellIds.assign((std::size_t)totalEntries, 0);
        allParticleLoads.assign((std::size_t)totalEntries, 0);
    }
    MPI_Gatherv(localEntryCount > 0 ? localCellIds.data() : nullptr,
                localEntryCount,
                MPI_INT,
                this->mpi->root() ? allCellIds.data() : nullptr,
                this->mpi->root() ? recvCounts.data() : nullptr,
                this->mpi->root() ? displs.data() : nullptr,
                MPI_INT,
                0,
                this->comm);
    MPI_Gatherv(localEntryCount > 0 ? localParticleLoads.data() : nullptr,
                localEntryCount,
                MPI_INT,
                this->mpi->root() ? allParticleLoads.data() : nullptr,
                this->mpi->root() ? recvCounts.data() : nullptr,
                this->mpi->root() ? displs.data() : nullptr,
                MPI_INT,
                0,
                this->comm);
    if (this->mpi->root())
    {
        for (int i = 0; i < totalEntries; ++i)
        {
            const int cell = allCellIds[(std::size_t)i];
            if (cell >= 0 && cell < this->mess.Ncell)
                particle_load[(std::size_t)cell] += allParticleLoads[(std::size_t)i];
        }
        cell_load = particle_load;
    }
    if (!this->DSMCprocess->enable_partition_time_weights)
        return;
    std::vector<int> localTimeCellIds;
    std::vector<double> localTimeValues;
    if (this->mpi->active())
    {
        this->DSMCprocess->ensure_partition_time_storage();
        if (!this->DSMCprocess->cell_time_weight_accum.empty())
        {
            localTimeCellIds.reserve((std::size_t)this->DSMCprocess->iNcell);
            localTimeValues.reserve((std::size_t)this->DSMCprocess->iNcell);
            for (const auto& item : this->DSMCprocess->cell_time_weight_accum)
            {
                const int cell = item.first;
                if (cell < 0 || cell >= this->mess.Ncell) continue;
                const double value = item.second;
                if (!(value > 0.0) || !std::isfinite(value)) continue;
                localTimeCellIds.push_back(cell);
                localTimeValues.push_back(value);
            }
        }
    }
    else if (this->mpi->root())
    {
        this->DSMCprocess->ensure_partition_time_storage();
    }
    const int localTimeCount = (int)localTimeCellIds.size();
    std::vector<int> timeRecvCounts;
    if (this->mpi->root())
        timeRecvCounts.assign((std::size_t)this->size, 0);
    MPI_Gather(&localTimeCount,
               1,
               MPI_INT,
               this->mpi->root() ? timeRecvCounts.data() : nullptr,
               1,
               MPI_INT,
               0,
               this->comm);
    std::vector<int> timeDispls;
    int totalTimeEntries = 0;
    if (this->mpi->root())
    {
        timeDispls.assign((std::size_t)this->size, 0);
        for (int r = 0; r < this->size; ++r)
        {
            timeDispls[(std::size_t)r] = totalTimeEntries;
            totalTimeEntries += timeRecvCounts[(std::size_t)r];
        }
    }
    std::vector<int> allTimeCellIds;
    std::vector<double> allTimeValues;
    if (this->mpi->root())
    {
        allTimeCellIds.assign((std::size_t)totalTimeEntries, 0);
        allTimeValues.assign((std::size_t)totalTimeEntries, 0.0);
    }
    MPI_Gatherv(localTimeCount > 0 ? localTimeCellIds.data() : nullptr,
                localTimeCount,
                MPI_INT,
                this->mpi->root() ? allTimeCellIds.data() : nullptr,
                this->mpi->root() ? timeRecvCounts.data() : nullptr,
                this->mpi->root() ? timeDispls.data() : nullptr,
                MPI_INT,
                0,
                this->comm);
    MPI_Gatherv(localTimeCount > 0 ? localTimeValues.data() : nullptr,
                localTimeCount,
                MPI_DOUBLE,
                this->mpi->root() ? allTimeValues.data() : nullptr,
                this->mpi->root() ? timeRecvCounts.data() : nullptr,
                this->mpi->root() ? timeDispls.data() : nullptr,
                MPI_DOUBLE,
                0,
                this->comm);
    if (this->mpi->root())
    {
        std::vector<double> global_time((std::size_t)this->mess.Ncell, 0.0);
        for (int i = 0; i < totalTimeEntries; ++i)
        {
            const int cell = allTimeCellIds[(std::size_t)i];
            if (cell >= 0 && cell < this->mess.Ncell)
                global_time[(std::size_t)cell] += allTimeValues[(std::size_t)i];
        }
        if (totalTimeEntries > 0)
        {
            this->DSMCprocess->ensure_partition_time_storage();
            for (int c = 0; c < this->mess.Ncell; ++c)
            {
                const double fallback = (double)std::max(1, particle_load[(std::size_t)c]);
                const double sample = (global_time[(std::size_t)c] > 0.0 && std::isfinite(global_time[(std::size_t)c]))
                    ? global_time[(std::size_t)c]
                    : fallback;
                double& ema = this->DSMCprocess->cell_time_weight_ema[(std::size_t)c];
                if (ema > 0.0 && std::isfinite(ema))
                    ema = this->DSMCprocess->cell_time_weight_ema_alpha * sample +
                          (1.0 - this->DSMCprocess->cell_time_weight_ema_alpha) * ema;
                else
                    ema = sample;
            }
            double time_avg = 0.0;
            int time_count = 0;
            for (double t : this->DSMCprocess->cell_time_weight_ema)
            {
                if (!(t > 0.0) || !std::isfinite(t)) continue;
                time_avg += t;
                ++time_count;
            }
            if (time_count > 0) time_avg /= (double)time_count;
            if (!(time_avg > 0.0) || !std::isfinite(time_avg)) time_avg = 1.0;
            for (int c = 0; c < this->mess.Ncell; ++c)
            {
                const double fallback = (double)std::max(1, particle_load[(std::size_t)c]);
                const double tw = this->DSMCprocess->cell_time_weight_ema[(std::size_t)c];
                const double effective = (tw > 0.0 && std::isfinite(tw)) ? tw : fallback;
                long long scaled = (long long)std::llround(100.0 * effective / time_avg);
                if (scaled <= 0) scaled = 1;
                if (scaled > (long long)INT_MAX) scaled = (long long)INT_MAX;
                cell_load[(std::size_t)c] = (int)scaled;
            }
        }
    }
    this->DSMCprocess->reset_partition_time_weights();
}

/*
 * rebuildDsmcAfterRepartition: updates partition ownership or load data.
 * Params: none; returns: none.
 * Flow:
 *   - gather load/owner data.
 *   - build the new mapping.
 *   - refresh dependent local state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
void dynamicDSMC::rebuildDsmcAfterRepartition()
{
    if (!this->mpi->active()) return;
    this->DSMCprocess->rebuildBoundaryDerivedState();
    this->DSMCprocess->record.assign((std::size_t)this->DSMCprocess->iNcell * (std::size_t)this->DSMCprocess->Madata, 0.0);
    this->DSMCprocess->final_record.assign((std::size_t)this->DSMCprocess->iNcell * (std::size_t)this->DSMCprocess->Madata, 0.0);
    this->DSMCprocess->local.assign((std::size_t)this->DSMCprocess->iNcell * (std::size_t)this->DSMCprocess->Madata, 0.0);
    this->DSMCprocess->dsmc2ns_window_samples.assign((std::size_t)this->DSMCprocess->iNcell, 0.0);
    this->DSMCprocess->dsmc2ns_window_valid.assign((std::size_t)this->DSMCprocess->iNcell, 0);
}
/*
 * average_load_calculate: updates partition ownership or load data.
 * Params: none; returns: none.
 * Flow:
 *   - gather load/owner data.
 *   - build the new mapping.
 *   - refresh dependent local state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
void dynamicDSMC::average_load_calculate()
{
    if (!this->mpi->active()) return;
    vector<int> cell_load(this->mess.Ncell, 0);
    int rank_load = 0;
    for (int i = 0; i < this->DSMCprocess->iNcell; i++)
    {
        int meshindex = this->DSMCprocess->globalOfLocalCell(i);
        if (meshindex < 0) continue;
        const int pnum = legacyCellLoadWeight(this->DSMCprocess, meshindex);
        rank_load += pnum;
        cell_load[meshindex] = pnum;
    }
    std::vector<int> cell_load_global;
    if (this->c_rank == 0)
        cell_load_global.assign((std::size_t)this->mess.Ncell, 0);
    MPI_Reduce(cell_load.data(),
               this->c_rank == 0 ? cell_load_global.data() : nullptr,
               this->mess.Ncell,
               MPI_INT,
               MPI_SUM,
               0,
               calGroup);
    vector<int> rank_load_all(this->c_size, 0);
    MPI_Gather(&rank_load, 1, MPI_INT, rank_load_all.data(), 1, MPI_INT, 0, calGroup);
    if (this->c_rank != 0) return;
    double avrg_rank_load = 0.0;
    for (int i = 0; i < this->c_size; i++)
        avrg_rank_load += rank_load_all[i];
    avrg_rank_load /= this->c_size;
}
