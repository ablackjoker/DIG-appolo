#include "MessagePassing.h"
#include "MeshparticalInitial.h"

#include <algorithm>
#include <cstddef>

namespace
{
void prefixCounts(const std::vector<int> &counts, std::vector<int> &displs, int &total)
{
    displs.assign(counts.size(), 0);
    total = 0;
    for (std::size_t i = 0; i < counts.size(); ++i)
    {
        displs[i] = total;
        total += counts[i];
    }
}

bool exchangeFlatPayload(const void *sendData,
                         const std::vector<int> &sendCounts,
                         const std::vector<int> &sendDispls,
                         MPI_Datatype datatype,
                         void *recvData,
                         const std::vector<int> &recvCounts,
                         const std::vector<int> &recvDispls,
                         const MpiContext &mpi)
{
    return MPI_Alltoallv(const_cast<void*>(sendData),
                         const_cast<int*>(sendCounts.data()),
                         const_cast<int*>(sendDispls.data()),
                         datatype,
                         recvData,
                         const_cast<int*>(recvCounts.data()),
                         const_cast<int*>(recvDispls.data()),
                         datatype,
                         mpi.calGroup) == MPI_SUCCESS;
}
}

MessagePassing::MessagePassing()
{
}

MessagePassing::~MessagePassing()
{
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) return;
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized && dtleftPacketDatatype != MPI_DATATYPE_NULL)
    {
        MPI_Type_free(&dtleftPacketDatatype);
        dtleftPacketDatatype = MPI_DATATYPE_NULL;
    }
    if (!finalized && particleDatatype != MPI_DATATYPE_NULL)
    {
        MPI_Type_free(&particleDatatype);
        particleDatatype = MPI_DATATYPE_NULL;
    }
}

bool MessagePassing::commitMyDsmcCell(MPI_Datatype &datatype)
{
    const int Len = 8;
    DsmcCell c;

    MPI_Aint offsets[Len];
    int blockcounts[Len] = {
        1,          
        1,          
        NN,         
        NN,         
        NN,         
        1,          
        1,          
        DIM         
    };

    MPI_Datatype types[Len] = {
        MPI_INT,
        MPI_INT,
        MPI_INT,
        MPI_INT,
        MPI_INT,
        MPI_DOUBLE,
        MPI_INT,
        MPI_DOUBLE
    };

    MPI_Get_address(&c.num,              &offsets[0]);
    MPI_Get_address(&c.fluentCellType,   &offsets[1]);
    MPI_Get_address(&c.cell2face[0],     &offsets[2]);
    MPI_Get_address(&c.cell2cell[0],     &offsets[3]);
    MPI_Get_address(&c.cell2face_sgn[0], &offsets[4]);
    MPI_Get_address(&c.area,             &offsets[5]);
    MPI_Get_address(&c.no,               &offsets[6]);
    MPI_Get_address(&c.cellXY[0],        &offsets[7]);

    for (int i = 1; i < Len; ++i) offsets[i] -= offsets[0];
    offsets[0] = 0;

    MPI_Type_create_struct(Len, blockcounts, offsets, types, &datatype);
    MPI_Type_commit(&datatype);
    return true;
}


bool MessagePassing::commitMyDsmcEdge(MPI_Datatype &datatype)
{
    const int Len = 6;
    DsmcEdge e;

    MPI_Aint offsets[Len];
    int blockcounts[Len] = {
        1,      
        1,      
        1,      
        NN,     
        DIM,     
        DIM    
    };

    MPI_Datatype types[Len] = {
        MPI_INT,
        MPI_INT,
        MPI_DOUBLE,
        MPI_INT,
        MPI_DOUBLE,
        MPI_DOUBLE
    };

    MPI_Get_address(&e.faceTag,        &offsets[0]);
    MPI_Get_address(&e.faceType,       &offsets[1]);
    MPI_Get_address(&e.length,         &offsets[2]);
    MPI_Get_address(&e.faceMap[0],     &offsets[3]);
    MPI_Get_address(&e.edgeNormal[0],  &offsets[4]);
    MPI_Get_address(&e.edgeCenter[0],  &offsets[5]);
    for (int i = 1; i < Len; ++i) offsets[i] -= offsets[0];
    offsets[0] = 0;

    MPI_Type_create_struct(Len, blockcounts, offsets, types, &datatype);
    MPI_Type_commit(&datatype);
    return true;
}

bool MessagePassing::commitMyCell(MPI_Datatype &datatype)
{
    const int Len = 14;
    cell c;
    MPI_Aint offsets[Len];
    int blockcounts[] = {1, 1, NV, NN, NN, NN, 1, 1, DIM * DIM, NN * DIM, 1, 1, 1, DIM};
    MPI_Datatype datatypes[] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_INT, MPI_INT, MPI_INT, MPI_DOUBLE};
    MPI_Get_address(&c.num, &offsets[0]); MPI_Get_address(&c.dim, &offsets[1]); MPI_Get_address(&c.cell2node[0], &offsets[2]);
    MPI_Get_address(&c.cell2face[0], &offsets[3]); MPI_Get_address(&c.cell2cell[0], &offsets[4]); MPI_Get_address(&c.cell2face_sgn[0], &offsets[5]);
    MPI_Get_address(&c.area, &offsets[6]); MPI_Get_address(&c.cellLengthEff, &offsets[7]); MPI_Get_address(&c.Ainv[0][0], &offsets[8]);
    MPI_Get_address(&c.dxyz[0][0], &offsets[9]); MPI_Get_address(&c.cellType, &offsets[10]); MPI_Get_address(&c.rawCellType, &offsets[11]); MPI_Get_address(&c.no, &offsets[12]);MPI_Get_address(&c.cellXY[0], &offsets[13]);

    for(int i=1; i<Len; i++){
        offsets[i] -= offsets[0];
    }
    offsets[0] = 0;

    MPI_Type_create_struct(Len, blockcounts, offsets, datatypes, &datatype);
    MPI_Type_commit(&datatype);

    return true;
}

bool MessagePassing::commitMyEdge(MPI_Datatype &datatype)
{
    const int Len = 13;
    edge e;
    MPI_Aint offsets[Len];
    int blockcounts[] = {1, 1, 1, 1, 1, 1, NN, DIM, DIM, 1, DIM, DIM, DIM};
    MPI_Datatype datatypes[] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_DOUBLE, MPI_INT, MPI_DOUBLE, MPI_DOUBLE, \
    MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
    MPI_Get_address(&e.no, &offsets[0]); MPI_Get_address(&e.faceTag, &offsets[1]); MPI_Get_address(&e.dim, &offsets[2]);
    MPI_Get_address(&e.faceType, &offsets[3]); MPI_Get_address(&e.bcType, &offsets[4]);
    MPI_Get_address(&e.length, &offsets[5]); MPI_Get_address(&e.faceMap[0], &offsets[6]); MPI_Get_address(&e.edgeCenter[0], &offsets[7]);
    MPI_Get_address(&e.edgeNormal[0], &offsets[8]);  MPI_Get_address(&e.edgeDist, &offsets[9]); 
    MPI_Get_address(&e.edgerij[0], &offsets[10]);
    MPI_Get_address(&e.edgerL[0], &offsets[11]); MPI_Get_address(&e.edgerR[0], &offsets[12]);

    for(int i=1; i<Len; i++){
        offsets[i] -= offsets[0];
    }
    offsets[0] = 0;
    
    
    
    
    MPI_Type_create_struct(Len, blockcounts, offsets, datatypes, &datatype);
    MPI_Type_commit(&datatype);

    return true;
}

bool MessagePassing::commitMyMesssge(MPI_Datatype &datatype)
{
    meshMessage mess;
    MPI_Aint offsets[47];
    int blockcounts[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    MPI_Datatype datatypes[] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_DOUBLE, MPI_DOUBLE, \
    MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_INT, \
    MPI_INT, MPI_DOUBLE, MPI_DOUBLE, \
    MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_INT,\
    MPI_DOUBLE, MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,\
    MPI_DOUBLE, MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE, MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE};
    MPI_Get_address(&mess.Ncell, &offsets[0]); MPI_Get_address(&mess.Nface, &offsets[1]); MPI_Get_address(&mess.Npoint, &offsets[2]);
    MPI_Get_address(&mess.Nk, &offsets[3]); MPI_Get_address(&mess.bcnumber, &offsets[4]); MPI_Get_address(&mess.Nghost, &offsets[5]);
    MPI_Get_address(&mess.Area, &offsets[6]); MPI_Get_address(&mess.Ma, &offsets[7]); MPI_Get_address(&mess.Kn, &offsets[8]);
    MPI_Get_address(&mess.cfl, &offsets[9]); MPI_Get_address(&mess.gamma, &offsets[10]);
    MPI_Get_address(&mess.dr, &offsets[11]); MPI_Get_address(&mess.dv, &offsets[12]);
    MPI_Get_address(&mess.zr, &offsets[13]); MPI_Get_address(&mess.zv, &offsets[14]); MPI_Get_address(&mess.omega, &offsets[15]);
    MPI_Get_address(&mess.var, &offsets[16]); MPI_Get_address(&mess.qn, &offsets[17]); 
    MPI_Get_address(&mess.f_tra, &offsets[18]); MPI_Get_address(&mess.f_rot, &offsets[19]);
    MPI_Get_address(&mess.delta_rp, &offsets[20]); MPI_Get_address(&mess.delta_sm, &offsets[21]); MPI_Get_address(&mess.omeag0, &offsets[22]);
    MPI_Get_address(&mess.omeag1, &offsets[23]); MPI_Get_address(&mess.cfl_ns, &offsets[24]); MPI_Get_address(&mess.pr, &offsets[25]);
    MPI_Get_address(&mess.cp, &offsets[26]); MPI_Get_address(&mess.cv, &offsets[27]); MPI_Get_address(&mess.eNface, &offsets[28]);
    MPI_Get_address(&mess.kB, &offsets[29]);MPI_Get_address(&mess.p_mass, &offsets[30]);
    MPI_Get_address(&mess.p_mass_r, &offsets[31]);MPI_Get_address(&mess.T_ref, &offsets[32]);MPI_Get_address(&mess.n_ref, &offsets[33]);
    MPI_Get_address(&mess.miu_ref, &offsets[34]);MPI_Get_address(&mess.Twall_ref, &offsets[35]);MPI_Get_address(&mess.p_ref, &offsets[36]);
    MPI_Get_address(&mess.d_ref, &offsets[37]);MPI_Get_address(&mess.alpha, &offsets[38]);MPI_Get_address(&mess.v_rms, &offsets[39]);
    MPI_Get_address(&mess.eta, &offsets[40]);MPI_Get_address(&mess.P_relax, &offsets[41]);MPI_Get_address(&mess.dt_ref, &offsets[42]);
    MPI_Get_address(&mess.T_in, &offsets[43]);MPI_Get_address(&mess.v_in, &offsets[44]);MPI_Get_address(&mess.dtime, &offsets[45]);
    MPI_Get_address(&mess.Neff, &offsets[46]);

    for(int i=1; i<47; i++){
        offsets[i] -= offsets[0];
    }
    offsets[0] = 0; 
    
    
    
    
    
    
    
    
    
    
    MPI_Type_create_struct(47, blockcounts, offsets, datatypes, &datatype);
    MPI_Type_commit(&datatype);

    return true;
}

bool MessagePassing::commitMyvis(MPI_Datatype &datatype)
{
    fNode fo;
    MPI_Aint offsets[2];
    int blockcounts[] = {1, 1};
    MPI_Datatype datatypes[] = {MPI_INT, MPI_INT};
    MPI_Get_address(&fo.fid, &offsets[0]); MPI_Get_address(&fo.no, &offsets[1]);  
    for(int i=1; i<2; i++){
        offsets[i] -= offsets[0];
    }
    offsets[0] = 0; 
    MPI_Type_create_struct(2, blockcounts, offsets, datatypes, &datatype);
    MPI_Type_commit(&datatype);

    return true;
}

bool MessagePassing::commitParticle(MPI_Datatype &datatype)
{
    const int nitems = 7;
    int blocklengths[7] = {1, 1, 1, DIM, DIM, 1, 1};
    MPI_Datatype types[7] = {
        MPI_INT,
        MPI_INT,
        MPI_INT,
        MPI_DOUBLE,
        MPI_DOUBLE,
        MPI_DOUBLE,
        MPI_DOUBLE
    };
    MPI_Aint offsets[7];

    offsets[0] = offsetof(particle, p_serial);
    offsets[1] = offsetof(particle, p_rank_serial);
    offsets[2] = offsetof(particle, p_mesh_serial);
    offsets[3] = offsetof(particle, p_velocity);
    offsets[4] = offsetof(particle, p_location);
    offsets[5] = offsetof(particle, p_Ir);
    offsets[6] = offsetof(particle, dt_left);

    MPI_Datatype rawDatatype = MPI_DATATYPE_NULL;
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &rawDatatype);
    MPI_Type_create_resized(rawDatatype, 0, (MPI_Aint)sizeof(particle), &datatype);
    MPI_Type_commit(&datatype);
    MPI_Type_free(&rawDatatype);
    return true;
}

bool MessagePassing::commitDtleftPacket(MPI_Datatype &datatype)
{
    if (!ensureParticleDatatype()) return false;

    const int nitems = 4;
    int blocklengths[4] = {1, 1, 1, 1};
    MPI_Datatype types[4] = {
        particleDatatype,
        MPI_INT,
        MPI_INT,
        MPI_INT
    };
    MPI_Aint offsets[4];

    offsets[0] = offsetof(DtleftPacket, p);
    offsets[1] = offsetof(DtleftPacket, gface);
    offsets[2] = offsetof(DtleftPacket, gcell);
    offsets[3] = offsetof(DtleftPacket, tri);

    MPI_Datatype rawDatatype = MPI_DATATYPE_NULL;
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &rawDatatype);
    MPI_Type_create_resized(rawDatatype, 0, (MPI_Aint)sizeof(DtleftPacket), &datatype);
    MPI_Type_commit(&datatype);
    MPI_Type_free(&rawDatatype);
    return true;
}

bool MessagePassing::ensureParticleDatatype()
{
    if (particleDatatype != MPI_DATATYPE_NULL) return true;
    return commitParticle(particleDatatype);
}

bool MessagePassing::ensureDtleftPacketDatatype()
{
    if (dtleftPacketDatatype != MPI_DATATYPE_NULL) return true;
    return commitDtleftPacket(dtleftPacketDatatype);
}

bool MessagePassing::exchangeParticleVectors(std::vector<std::vector<particle>> &sendCache,
                                             std::vector<std::vector<particle>> &recvCache,
                                             const MpiContext &mpi,
                                             int tag)
{
    if (!mpi.active()) return true;

    const int nrank = mpi.c_size;
    if ((int)sendCache.size() < nrank)
        sendCache.resize((std::size_t)nrank);
    if ((int)recvCache.size() < nrank)
        recvCache.resize((std::size_t)nrank);
    for (int r = 0; r < nrank; ++r)
        recvCache[(std::size_t)r].clear();

    std::vector<int> sendCounts((std::size_t)nrank, 0);
    std::vector<int> recvCounts((std::size_t)nrank, 0);
    for (int r = 0; r < nrank; ++r)
        sendCounts[(std::size_t)r] = (int)sendCache[(std::size_t)r].size();

    if (MPI_Alltoall(sendCounts.data(), 1, MPI_INT,
                     recvCounts.data(), 1, MPI_INT,
                     mpi.calGroup) != MPI_SUCCESS)
        return false;

    if (!ensureParticleDatatype()) return false;

    std::vector<MPI_Request> requests;
    requests.reserve((std::size_t)nrank * 2u);

    if (mpi.c_rank >= 0 && mpi.c_rank < nrank && recvCounts[(std::size_t)mpi.c_rank] > 0)
        recvCache[(std::size_t)mpi.c_rank] = sendCache[(std::size_t)mpi.c_rank];

    for (int r = 0; r < nrank; ++r)
    {
        if (r == mpi.c_rank) continue;
        if (recvCounts[(std::size_t)r] > 0)
        {
            if (recvCache[(std::size_t)r].capacity() < (std::size_t)recvCounts[(std::size_t)r])
                recvCache[(std::size_t)r].reserve((std::size_t)recvCounts[(std::size_t)r]);
            recvCache[(std::size_t)r].resize((std::size_t)recvCounts[(std::size_t)r]);
            MPI_Request req = MPI_REQUEST_NULL;
            if (MPI_Irecv(recvCache[(std::size_t)r].data(), recvCounts[(std::size_t)r], particleDatatype,
                          r, tag, mpi.calGroup, &req) != MPI_SUCCESS)
            {
                return false;
            }
            requests.push_back(req);
        }
    }

    for (int r = 0; r < nrank; ++r)
    {
        if (r == mpi.c_rank) continue;
        if (sendCounts[(std::size_t)r] > 0)
        {
            MPI_Request req = MPI_REQUEST_NULL;
            if (MPI_Isend(sendCache[(std::size_t)r].data(), sendCounts[(std::size_t)r], particleDatatype,
                          r, tag, mpi.calGroup, &req) != MPI_SUCCESS)
            {
                return false;
            }
            requests.push_back(req);
        }
    }

    if (!requests.empty() &&
        MPI_Waitall((int)requests.size(), requests.data(), MPI_STATUSES_IGNORE) != MPI_SUCCESS)
    {
        return false;
    }

    return true;
}

bool MessagePassing::exchangeParticleVectorsOnPeers(std::vector<std::vector<particle>> &sendCache,
                                                    std::vector<std::vector<particle>> &recvCache,
                                                    const MpiContext &mpi,
                                                    const std::vector<int> &peerRanks,
                                                    int tag)
{
    if (!mpi.active()) return true;

    const int nrank = mpi.c_size;
    if ((int)sendCache.size() < nrank)
        sendCache.resize((std::size_t)nrank);
    if ((int)recvCache.size() < nrank)
        recvCache.resize((std::size_t)nrank);
    for (int r = 0; r < nrank; ++r)
        recvCache[(std::size_t)r].clear();

    std::vector<char> peerMask((std::size_t)nrank, 0);
    for (int peer : peerRanks)
    {
        if (peer >= 0 && peer < nrank && peer != mpi.c_rank)
            peerMask[(std::size_t)peer] = 1;
    }

    for (int r = 0; r < nrank; ++r)
    {
        if (r == mpi.c_rank) continue;
        if (!peerMask[(std::size_t)r] && !sendCache[(std::size_t)r].empty())
            return false;
    }

    if (!ensureParticleDatatype()) return false;

    std::vector<int> sendCounts((std::size_t)nrank, 0);
    std::vector<int> recvCounts((std::size_t)nrank, 0);
    for (int peer : peerRanks)
    {
        if (peer < 0 || peer >= nrank || peer == mpi.c_rank) continue;
        sendCounts[(std::size_t)peer] = (int)sendCache[(std::size_t)peer].size();
    }

    const int countTag = tag + 10000;
    std::vector<MPI_Request> countRequests;
    countRequests.reserve(peerRanks.size() * 2u);
    for (int peer : peerRanks)
    {
        if (peer < 0 || peer >= nrank || peer == mpi.c_rank) continue;
        MPI_Request req = MPI_REQUEST_NULL;
        if (MPI_Irecv(&recvCounts[(std::size_t)peer], 1, MPI_INT,
                      peer, countTag, mpi.calGroup, &req) != MPI_SUCCESS)
            return false;
        countRequests.push_back(req);
    }
    for (int peer : peerRanks)
    {
        if (peer < 0 || peer >= nrank || peer == mpi.c_rank) continue;
        MPI_Request req = MPI_REQUEST_NULL;
        if (MPI_Isend(&sendCounts[(std::size_t)peer], 1, MPI_INT,
                      peer, countTag, mpi.calGroup, &req) != MPI_SUCCESS)
            return false;
        countRequests.push_back(req);
    }
    if (!countRequests.empty() &&
        MPI_Waitall((int)countRequests.size(), countRequests.data(), MPI_STATUSES_IGNORE) != MPI_SUCCESS)
        return false;

    if (mpi.c_rank >= 0 && mpi.c_rank < nrank && !sendCache[(std::size_t)mpi.c_rank].empty())
        recvCache[(std::size_t)mpi.c_rank] = sendCache[(std::size_t)mpi.c_rank];

    std::vector<MPI_Request> requests;
    requests.reserve(peerRanks.size() * 2u);
    for (int peer : peerRanks)
    {
        if (peer < 0 || peer >= nrank || peer == mpi.c_rank) continue;
        const int recvCount = recvCounts[(std::size_t)peer];
        if (recvCount <= 0) continue;
        if (recvCache[(std::size_t)peer].capacity() < (std::size_t)recvCount)
            recvCache[(std::size_t)peer].reserve((std::size_t)recvCount);
        recvCache[(std::size_t)peer].resize((std::size_t)recvCount);
        MPI_Request req = MPI_REQUEST_NULL;
        if (MPI_Irecv(recvCache[(std::size_t)peer].data(), recvCount, particleDatatype,
                      peer, tag, mpi.calGroup, &req) != MPI_SUCCESS)
            return false;
        requests.push_back(req);
    }
    for (int peer : peerRanks)
    {
        if (peer < 0 || peer >= nrank || peer == mpi.c_rank) continue;
        const int sendCount = sendCounts[(std::size_t)peer];
        if (sendCount <= 0) continue;
        MPI_Request req = MPI_REQUEST_NULL;
        if (MPI_Isend(sendCache[(std::size_t)peer].data(), sendCount, particleDatatype,
                      peer, tag, mpi.calGroup, &req) != MPI_SUCCESS)
            return false;
        requests.push_back(req);
    }
    if (!requests.empty() &&
        MPI_Waitall((int)requests.size(), requests.data(), MPI_STATUSES_IGNORE) != MPI_SUCCESS)
        return false;

    return true;
}

bool MessagePassing::exchangeDtleftPacketVectors(std::vector<std::vector<DtleftPacket>> &sendCache,
                                                 std::vector<std::vector<DtleftPacket>> &recvCache,
                                                 const MpiContext &mpi,
                                                 int tag,
                                                 bool *hadTraffic)
{
    if (hadTraffic != nullptr) *hadTraffic = false;
    if (!mpi.active()) return true;

    const int nrank = mpi.c_size;
    if ((int)sendCache.size() < nrank)
        sendCache.resize((std::size_t)nrank);
    if ((int)recvCache.size() < nrank)
        recvCache.resize((std::size_t)nrank);
    for (int r = 0; r < nrank; ++r)
        recvCache[(std::size_t)r].clear();

    std::vector<int> sendCounts((std::size_t)nrank, 0);
    std::vector<int> recvCounts((std::size_t)nrank, 0);
    for (int r = 0; r < nrank; ++r)
        sendCounts[(std::size_t)r] = (int)sendCache[(std::size_t)r].size();

    if (MPI_Alltoall(sendCounts.data(), 1, MPI_INT,
                     recvCounts.data(), 1, MPI_INT,
                     mpi.calGroup) != MPI_SUCCESS)
        return false;

    if (hadTraffic != nullptr)
    {
        for (int r = 0; r < nrank; ++r)
        {
            if (sendCounts[(std::size_t)r] > 0 || recvCounts[(std::size_t)r] > 0)
            {
                *hadTraffic = true;
                break;
            }
        }
    }

    if (!ensureDtleftPacketDatatype()) return false;

    std::vector<MPI_Request> requests;
    requests.reserve((std::size_t)nrank * 2u);

    if (mpi.c_rank >= 0 && mpi.c_rank < nrank && recvCounts[(std::size_t)mpi.c_rank] > 0)
        recvCache[(std::size_t)mpi.c_rank] = sendCache[(std::size_t)mpi.c_rank];

    for (int r = 0; r < nrank; ++r)
    {
        if (r == mpi.c_rank) continue;
        if (recvCounts[(std::size_t)r] > 0)
        {
            if (recvCache[(std::size_t)r].capacity() < (std::size_t)recvCounts[(std::size_t)r])
                recvCache[(std::size_t)r].reserve((std::size_t)recvCounts[(std::size_t)r]);
            recvCache[(std::size_t)r].resize((std::size_t)recvCounts[(std::size_t)r]);
            MPI_Request req = MPI_REQUEST_NULL;
            if (MPI_Irecv(recvCache[(std::size_t)r].data(), recvCounts[(std::size_t)r], dtleftPacketDatatype,
                          r, tag, mpi.calGroup, &req) != MPI_SUCCESS)
            {
                return false;
            }
            requests.push_back(req);
        }
    }

    for (int r = 0; r < nrank; ++r)
    {
        if (r == mpi.c_rank) continue;
        if (sendCounts[(std::size_t)r] > 0)
        {
            MPI_Request req = MPI_REQUEST_NULL;
            if (MPI_Isend(sendCache[(std::size_t)r].data(), sendCounts[(std::size_t)r], dtleftPacketDatatype,
                          r, tag, mpi.calGroup, &req) != MPI_SUCCESS)
            {
                return false;
            }
            requests.push_back(req);
        }
    }

    if (!requests.empty() &&
        MPI_Waitall((int)requests.size(), requests.data(), MPI_STATUSES_IGNORE) != MPI_SUCCESS)
    {
        return false;
    }

    return true;
}

bool MessagePassing::exchangeDtleftPacketVectorsOnPeers(std::vector<std::vector<DtleftPacket>> &sendCache,
                                                        std::vector<std::vector<DtleftPacket>> &recvCache,
                                                        const MpiContext &mpi,
                                                        const std::vector<int> &peerRanks,
                                                        int tag,
                                                        bool *hadTraffic)
{
    if (hadTraffic != nullptr) *hadTraffic = false;
    if (!mpi.active()) return true;

    const int nrank = mpi.c_size;
    if ((int)sendCache.size() < nrank)
        sendCache.resize((std::size_t)nrank);
    if ((int)recvCache.size() < nrank)
        recvCache.resize((std::size_t)nrank);
    for (int r = 0; r < nrank; ++r)
        recvCache[(std::size_t)r].clear();

    std::vector<char> peerMask((std::size_t)nrank, 0);
    for (int peer : peerRanks)
    {
        if (peer >= 0 && peer < nrank && peer != mpi.c_rank)
            peerMask[(std::size_t)peer] = 1;
    }

    for (int r = 0; r < nrank; ++r)
    {
        if (r == mpi.c_rank) continue;
        if (!peerMask[(std::size_t)r] && !sendCache[(std::size_t)r].empty())
            return false;
    }

    if (!ensureDtleftPacketDatatype()) return false;

    std::vector<int> sendCounts((std::size_t)nrank, 0);
    std::vector<int> recvCounts((std::size_t)nrank, 0);
    for (int peer : peerRanks)
    {
        if (peer < 0 || peer >= nrank || peer == mpi.c_rank) continue;
        sendCounts[(std::size_t)peer] = (int)sendCache[(std::size_t)peer].size();
    }

    const int countTag = tag + 10000;
    std::vector<MPI_Request> countRequests;
    countRequests.reserve(peerRanks.size() * 2u);
    for (int peer : peerRanks)
    {
        if (peer < 0 || peer >= nrank || peer == mpi.c_rank) continue;
        MPI_Request req = MPI_REQUEST_NULL;
        if (MPI_Irecv(&recvCounts[(std::size_t)peer], 1, MPI_INT,
                      peer, countTag, mpi.calGroup, &req) != MPI_SUCCESS)
            return false;
        countRequests.push_back(req);
    }
    for (int peer : peerRanks)
    {
        if (peer < 0 || peer >= nrank || peer == mpi.c_rank) continue;
        MPI_Request req = MPI_REQUEST_NULL;
        if (MPI_Isend(&sendCounts[(std::size_t)peer], 1, MPI_INT,
                      peer, countTag, mpi.calGroup, &req) != MPI_SUCCESS)
            return false;
        countRequests.push_back(req);
    }
    if (!countRequests.empty() &&
        MPI_Waitall((int)countRequests.size(), countRequests.data(), MPI_STATUSES_IGNORE) != MPI_SUCCESS)
        return false;

    if (hadTraffic != nullptr)
    {
        if (!sendCache[(std::size_t)mpi.c_rank].empty())
            *hadTraffic = true;
        for (int peer : peerRanks)
        {
            if (peer < 0 || peer >= nrank || peer == mpi.c_rank) continue;
            if (sendCounts[(std::size_t)peer] > 0 || recvCounts[(std::size_t)peer] > 0)
            {
                *hadTraffic = true;
                break;
            }
        }
    }

    if (mpi.c_rank >= 0 && mpi.c_rank < nrank && !sendCache[(std::size_t)mpi.c_rank].empty())
        recvCache[(std::size_t)mpi.c_rank] = sendCache[(std::size_t)mpi.c_rank];

    std::vector<MPI_Request> requests;
    requests.reserve(peerRanks.size() * 2u);
    for (int peer : peerRanks)
    {
        if (peer < 0 || peer >= nrank || peer == mpi.c_rank) continue;
        const int recvCount = recvCounts[(std::size_t)peer];
        if (recvCount <= 0) continue;
        if (recvCache[(std::size_t)peer].capacity() < (std::size_t)recvCount)
            recvCache[(std::size_t)peer].reserve((std::size_t)recvCount);
        recvCache[(std::size_t)peer].resize((std::size_t)recvCount);
        MPI_Request req = MPI_REQUEST_NULL;
        if (MPI_Irecv(recvCache[(std::size_t)peer].data(), recvCount, dtleftPacketDatatype,
                      peer, tag, mpi.calGroup, &req) != MPI_SUCCESS)
            return false;
        requests.push_back(req);
    }
    for (int peer : peerRanks)
    {
        if (peer < 0 || peer >= nrank || peer == mpi.c_rank) continue;
        const int sendCount = sendCounts[(std::size_t)peer];
        if (sendCount <= 0) continue;
        MPI_Request req = MPI_REQUEST_NULL;
        if (MPI_Isend(sendCache[(std::size_t)peer].data(), sendCount, dtleftPacketDatatype,
                      peer, tag, mpi.calGroup, &req) != MPI_SUCCESS)
            return false;
        requests.push_back(req);
    }
    if (!requests.empty() &&
        MPI_Waitall((int)requests.size(), requests.data(), MPI_STATUSES_IGNORE) != MPI_SUCCESS)
        return false;

    return true;
}
bool MessagePassing::exchangeFixedWidthDoublePackets(const std::vector<int> &sendCounts,
                                                     const std::vector<int> &sendGids,
                                                     const std::vector<double> &sendValues,
                                                     int valueWidth,
                                                     std::vector<int> &recvGids,
                                                     std::vector<double> &recvValues,
                                                     const MpiContext &mpi) const
{
    if (!mpi.active())
    {
        recvGids.clear();
        recvValues.clear();
        return true;
    }

    const int nrank = mpi.c_size;
    if (nrank <= 0 || valueWidth <= 0 || (int)sendCounts.size() != nrank)
        return false;

    int expectedSend = 0;
    bool localOk = true;
    for (int c : sendCounts)
    {
        if (c < 0) localOk = false;
        expectedSend += c;
    }
    if (expectedSend < 0 ||
        expectedSend != (int)sendGids.size() ||
        expectedSend * valueWidth != (int)sendValues.size())
        localOk = false;

    int localOkInt = localOk ? 1 : 0;
    std::vector<int> allOk((std::size_t)nrank, 0);
    MPI_Allgather(&localOkInt, 1, MPI_INT, allOk.data(), 1, MPI_INT, mpi.calGroup);
    if (std::find(allOk.begin(), allOk.end(), 0) != allOk.end())
        return false;

    std::vector<int> recvCounts((std::size_t)nrank, 0);
    if (MPI_Alltoall(const_cast<int*>(sendCounts.data()), 1, MPI_INT,
                     recvCounts.data(), 1, MPI_INT,
                     mpi.calGroup) != MPI_SUCCESS)
        return false;

    std::vector<int> sdispls;
    std::vector<int> rdispls;
    int sendTotal = 0;
    int recvTotal = 0;
    prefixCounts(sendCounts, sdispls, sendTotal);
    prefixCounts(recvCounts, rdispls, recvTotal);

    recvGids.assign((std::size_t)recvTotal, 0);
    recvValues.assign((std::size_t)recvTotal * (std::size_t)valueWidth, 0.0);

    if (!exchangeFlatPayload(sendGids.empty() ? nullptr : sendGids.data(),
                             sendCounts,
                             sdispls,
                             MPI_INT,
                             recvGids.empty() ? nullptr : recvGids.data(),
                             recvCounts,
                             rdispls,
                             mpi))
        return false;

    std::vector<int> sendCountsV((std::size_t)nrank, 0);
    std::vector<int> recvCountsV((std::size_t)nrank, 0);
    std::vector<int> sdisplsV((std::size_t)nrank, 0);
    std::vector<int> rdisplsV((std::size_t)nrank, 0);
    for (int r = 0; r < nrank; ++r)
    {
        sendCountsV[(std::size_t)r] = sendCounts[(std::size_t)r] * valueWidth;
        recvCountsV[(std::size_t)r] = recvCounts[(std::size_t)r] * valueWidth;
        sdisplsV[(std::size_t)r] = sdispls[(std::size_t)r] * valueWidth;
        rdisplsV[(std::size_t)r] = rdispls[(std::size_t)r] * valueWidth;
    }

    if (!exchangeFlatPayload(sendValues.empty() ? nullptr : sendValues.data(),
                             sendCountsV,
                             sdisplsV,
                             MPI_DOUBLE,
                             recvValues.empty() ? nullptr : recvValues.data(),
                             recvCountsV,
                             rdisplsV,
                             mpi))
        return false;

    return true;
}
