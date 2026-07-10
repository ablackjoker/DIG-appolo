#pragma once
#include <mpi.h>

class MessagePassing;

struct MpiContext
{
    MPI_Comm comm     = MPI_COMM_NULL;
    MPI_Comm calGroup = MPI_COMM_NULL;
    int rank = 0;
    int size = 1;
    int c_rank = -1;
    int c_size = 0;
    MessagePassing* mpass = nullptr;
    MpiContext() = default;
    static inline bool activeFromWorldRank(int worldRank) { return worldRank != 0; }
    MpiContext(MPI_Comm comm_, MPI_Comm calGroup_, MessagePassing* mp = nullptr)
        : comm(comm_), calGroup(calGroup_), mpass(mp)
    {
        if (comm != MPI_COMM_NULL) {
            MPI_Comm_rank(comm, &rank);
            MPI_Comm_size(comm, &size);
        } else {
            rank = 0;
            size = 1;
        }
        if (calGroup != MPI_COMM_NULL) {
            MPI_Comm_rank(calGroup, &c_rank);
            MPI_Comm_size(calGroup, &c_size);
        } else {
            c_rank = -1;
            c_size = (size > 0) ? (size - 1) : 0;  
        }
    }
    inline bool active() const { return calGroup != MPI_COMM_NULL; }
    inline bool root() const { return !active(); }
    inline bool activeLeader() const { return active() && c_rank == 0; }
};
