/*
 * MPI datatype builders and packet exchange helpers.
 */

#pragma once
#include "meshImport.h"
#include "MpiContext.h"
#include <vector>

struct particle;
struct DtleftPacket;

// MessagePassing stores state used by this module.
class MessagePassing
{
private:
    MPI_Datatype particleDatatype = MPI_DATATYPE_NULL;
    MPI_Datatype dtleftPacketDatatype = MPI_DATATYPE_NULL;
// ensureParticleDatatype: writes solver fields or diagnostics.
    bool ensureParticleDatatype();
// ensureDtleftPacketDatatype: writes solver fields or diagnostics.
    bool ensureDtleftPacketDatatype();
public:
    MessagePassing();
// ~MessagePassing: releases owned buffers and MPI helper state.
    ~MessagePassing();
// commitMyDsmcCell (datatype): moves structured data through MPI.
    bool commitMyDsmcCell(MPI_Datatype &datatype);
// commitMyDsmcEdge (datatype): moves structured data through MPI.
    bool commitMyDsmcEdge(MPI_Datatype &datatype);
    bool commitMyMesssge(MPI_Datatype &datatype);
// commitMyEdge (datatype): moves structured data through MPI.
    bool commitMyEdge(MPI_Datatype &datatype);
// commitMyCell (datatype): moves structured data through MPI.
    bool commitMyCell(MPI_Datatype &datatype);
// commitMyvis (datatype): moves structured data through MPI.
    bool commitMyvis(MPI_Datatype &datatype);
// commitParticle (datatype): moves structured data through MPI.
    bool commitParticle(MPI_Datatype &datatype);
    bool commitDtleftPacket(MPI_Datatype &datatype);
// exchangeParticleVectors (sendCache, recvCache, mpi, tag): moves structured data through MPI.
    bool exchangeParticleVectors(std::vector<std::vector<particle>> &sendCache,
                                 std::vector<std::vector<particle>> &recvCache,
                                 const MpiContext &mpi,
                                 int tag);
// exchangeParticleVectorsOnPeers (sendCache, recvCache, mpi, peerRanks, tag): moves structured data through MPI.
    bool exchangeParticleVectorsOnPeers(std::vector<std::vector<particle>> &sendCache,
                                        std::vector<std::vector<particle>> &recvCache,
                                        const MpiContext &mpi,
                                        const std::vector<int> &peerRanks,
                                        int tag);
// exchangeDtleftPacketVectors (sendCache, recvCache, mpi, tag, hadTraffic): moves structured data through MPI.
    bool exchangeDtleftPacketVectors(std::vector<std::vector<DtleftPacket>> &sendCache,
                                     std::vector<std::vector<DtleftPacket>> &recvCache,
                                     const MpiContext &mpi,
                                     int tag,
                                     bool *hadTraffic = nullptr);
    bool exchangeDtleftPacketVectorsOnPeers(std::vector<std::vector<DtleftPacket>> &sendCache,
                                            std::vector<std::vector<DtleftPacket>> &recvCache,
                                            const MpiContext &mpi,
                                            const std::vector<int> &peerRanks,
                                            int tag,
                                            bool *hadTraffic = nullptr);
// exchangeFixedWidthDoublePackets (sendCounts, sendGids, sendValues, valueWidth, recvGids, recvValues, mpi): moves structured data through MPI.
    bool exchangeFixedWidthDoublePackets(const std::vector<int> &sendCounts,
                                         const std::vector<int> &sendGids,
                                         const std::vector<double> &sendValues,
                                         int valueWidth,
                                         std::vector<int> &recvGids,
                                         std::vector<double> &recvValues,
                                         const MpiContext &mpi) const;
};
