#pragma once
#include "meshImport.h"
#include "MpiContext.h"
#include <vector>

struct particle;
struct DtleftPacket;

class MessagePassing
{
private:
    MPI_Datatype particleDatatype = MPI_DATATYPE_NULL;
    MPI_Datatype dtleftPacketDatatype = MPI_DATATYPE_NULL;
    bool ensureParticleDatatype();
    bool ensureDtleftPacketDatatype();
public:
    MessagePassing();
    ~MessagePassing();
    bool commitMyDsmcCell(MPI_Datatype &datatype);
    bool commitMyDsmcEdge(MPI_Datatype &datatype);
    bool commitMyMesssge(MPI_Datatype &datatype);
    bool commitMyEdge(MPI_Datatype &datatype);
    bool commitMyCell(MPI_Datatype &datatype);
    bool commitMyvis(MPI_Datatype &datatype);
    bool commitParticle(MPI_Datatype &datatype);
    bool commitDtleftPacket(MPI_Datatype &datatype);

    bool exchangeParticleVectors(std::vector<std::vector<particle>> &sendCache,
                                 std::vector<std::vector<particle>> &recvCache,
                                 const MpiContext &mpi,
                                 int tag);
    bool exchangeParticleVectorsOnPeers(std::vector<std::vector<particle>> &sendCache,
                                        std::vector<std::vector<particle>> &recvCache,
                                        const MpiContext &mpi,
                                        const std::vector<int> &peerRanks,
                                        int tag);

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

    bool exchangeFixedWidthDoublePackets(const std::vector<int> &sendCounts,
                                         const std::vector<int> &sendGids,
                                         const std::vector<double> &sendValues,
                                         int valueWidth,
                                         std::vector<int> &recvGids,
                                         std::vector<double> &recvValues,
                                         const MpiContext &mpi) const;
};
