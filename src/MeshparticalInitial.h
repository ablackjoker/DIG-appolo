/*
 * Particle records, SoA buckets, and particle initialization API.
 */

#pragma once
# include "meshImport.h"
# include "MessagePassing.h"
# include "MpiContext.h"
# include "MeshPartitionTransfer3D.h"
# include <random>
# include <unordered_map>
using namespace std;

// particle stores state used by this module.
struct particle
{
    // Serial ids track global particle identity and the owning rank.
    int p_serial;
    int p_rank_serial;
    int p_mesh_serial;
    // Velocity and position are stored in normalized DSMC units.
    double p_velocity[DIM];
    double p_location[DIM];
    // Rotational internal energy and remaining sub-step time.
    double p_Ir;
    double dt_left;
};

// DtleftPacket stores state used by this module.
struct DtleftPacket
{
    // Particle plus crossing metadata for unfinished remote tracing.
    particle p;
    int gface = -1;
    int gcell = -1;
    int tri = -1;
};

// ParticleBucketSoA stores state used by this module.
struct ParticleBucketSoA
{
    // Structure-of-arrays storage keeps hot particle fields cache friendly.
    std::vector<int> p_serial;
    std::vector<int> p_rank_serial;
    std::vector<int> p_mesh_serial;
    std::vector<double> vx;
    std::vector<double> vy;
    std::vector<double> vz;
    std::vector<double> px;
    std::vector<double> py;
    std::vector<double> pz;
    std::vector<double> p_Ir;
    std::vector<double> dt_left;
// size: returns the current particle count.
    size_t size() const { return p_serial.size(); }
// empty: checks whether the bucket contains particles.
    bool empty() const { return p_serial.empty(); }
// capacity: exposes allocated particle capacity.
    size_t capacity() const { return p_serial.capacity(); }
// clear: prepares derived solver state.
    void clear()
    {
        p_serial.clear();
        p_rank_serial.clear();
        p_mesh_serial.clear();
        vx.clear();
        vy.clear();
        vz.clear();
        px.clear();
        py.clear();
        pz.clear();
        p_Ir.clear();
        dt_left.clear();
    }
// reserve (n): reserves every SoA column together.
    void reserve(size_t n)
    {
        p_serial.reserve(n);
        p_rank_serial.reserve(n);
        p_mesh_serial.reserve(n);
        vx.reserve(n);
        vy.reserve(n);
        vz.reserve(n);
        px.reserve(n);
        py.reserve(n);
        pz.reserve(n);
        p_Ir.reserve(n);
        dt_left.reserve(n);
    }
// resize (n): resizes every SoA column together.
    void resize(size_t n)
    {
        p_serial.resize(n);
        p_rank_serial.resize(n);
        p_mesh_serial.resize(n);
        vx.resize(n);
        vy.resize(n);
        vz.resize(n);
        px.resize(n);
        py.resize(n);
        pz.resize(n);
        p_Ir.resize(n);
        dt_left.resize(n);
    }
// get (i): reconstructs an AoS particle from SoA columns.
    particle get(size_t i) const
    {
        particle part;
        part.p_serial = p_serial[i];
        part.p_rank_serial = p_rank_serial[i];
        part.p_mesh_serial = p_mesh_serial[i];
        part.p_velocity[0] = vx[i];
        part.p_velocity[1] = vy[i];
        part.p_velocity[2] = vz[i];
        part.p_location[0] = px[i];
        part.p_location[1] = py[i];
        part.p_location[2] = pz[i];
        part.p_Ir = p_Ir[i];
        part.dt_left = dt_left[i];
        return part;
    }
// set (i, part): writes an AoS particle into SoA columns.
    void set(size_t i, const particle &part)
    {
        p_serial[i] = part.p_serial;
        p_rank_serial[i] = part.p_rank_serial;
        p_mesh_serial[i] = part.p_mesh_serial;
        vx[i] = part.p_velocity[0];
        vy[i] = part.p_velocity[1];
        vz[i] = part.p_velocity[2];
        px[i] = part.p_location[0];
        py[i] = part.p_location[1];
        pz[i] = part.p_location[2];
        p_Ir[i] = part.p_Ir;
        dt_left[i] = part.dt_left;
    }
// push_back (part): appends one particle to all SoA columns.
    void push_back(const particle &part)
    {
        p_serial.push_back(part.p_serial);
        p_rank_serial.push_back(part.p_rank_serial);
        p_mesh_serial.push_back(part.p_mesh_serial);
        vx.push_back(part.p_velocity[0]);
        vy.push_back(part.p_velocity[1]);
        vz.push_back(part.p_velocity[2]);
        px.push_back(part.p_location[0]);
        py.push_back(part.p_location[1]);
        pz.push_back(part.p_location[2]);
        p_Ir.push_back(part.p_Ir);
        dt_left.push_back(part.dt_left);
    }
// pop_back: removes the last particle from every SoA column.
    void pop_back()
    {
        p_serial.pop_back();
        p_rank_serial.pop_back();
        p_mesh_serial.pop_back();
        vx.pop_back();
        vy.pop_back();
        vz.pop_back();
        px.pop_back();
        py.pop_back();
        pz.pop_back();
        p_Ir.pop_back();
        dt_left.pop_back();
    }
// append_bucket (other): moves all particles from another bucket.
    void append_bucket(ParticleBucketSoA &other)
    {
        const size_t addSize = other.size();
        if (addSize == 0) return;
        reserve(size() + addSize);
        p_serial.insert(p_serial.end(), other.p_serial.begin(), other.p_serial.end());
        p_rank_serial.insert(p_rank_serial.end(), other.p_rank_serial.begin(), other.p_rank_serial.end());
        p_mesh_serial.insert(p_mesh_serial.end(), other.p_mesh_serial.begin(), other.p_mesh_serial.end());
        vx.insert(vx.end(), other.vx.begin(), other.vx.end());
        vy.insert(vy.end(), other.vy.begin(), other.vy.end());
        vz.insert(vz.end(), other.vz.begin(), other.vz.end());
        px.insert(px.end(), other.px.begin(), other.px.end());
        py.insert(py.end(), other.py.begin(), other.py.end());
        pz.insert(pz.end(), other.pz.begin(), other.pz.end());
        p_Ir.insert(p_Ir.end(), other.p_Ir.begin(), other.p_Ir.end());
        dt_left.insert(dt_left.end(), other.dt_left.begin(), other.dt_left.end());
        other.clear();
    }
};

// MeshparticalInitial stores state used by this module.
class MeshparticalInitial
{
private:
public:
// thread_rng: returns one random generator per host thread.
    inline mt19937& thread_rng()
    {
        static thread_local mt19937 gen{std::random_device{}()};
        return gen;
    }
// get_uniform: reuses the standard [0, 1] sampler.
    static std::uniform_real_distribution<double>& get_uniform() {
        static thread_local std::uniform_real_distribution<double> dis(0.0, 1.0);
        return dis;
    }
    // Local mesh package installed after partition transfer.
    meshImport *mesh = NULL;
    meshMessage mess; 
    // Compact cell, face, point, and split-tag data for this rank.
    vector<DsmcCell> cells;
    vector<DsmcEdge> edges;
    vector<double> localPointXY;
    vector<unsigned char> faceSplitTag;
    const MpiContext* mpi = nullptr;
    MessagePassing *mpass = NULL;
    // Rank ids and local ownership maps for particle initialization.
    int rank,size,c_rank,c_size;
    // Global-to-local maps are shared with DSMC transport after setup.
    vector<int> local_cells;vector<int> face_gids;
    unordered_map<int,int> gid2local;unordered_map<int,int> face_gid2local;
    vector<int> rank_cell_all;
    // Partition readiness tells the driver whether initialization can continue.
    PartitionState3D partitionState;
    bool partitionReady = false;
    int my_ncell, my_nface;
    int my_owned_ncell = 0; 
    // Per-cell particle buckets and collision remainders.
    MPI_Comm calGroup;
    MPI_Comm comm;
    // Current buckets hold active particles; next buckets receive advected ones.
    vector<ParticleBucketSoA> cell_particles_curr;
    unordered_map<int, ParticleBucketSoA> cell_particles_next;
    // Reserve hints reduce reallocations during migration and refill.
    vector<int> cell_particle_reserve_hint;
    vector<double> crmax;
    // Fractional remainders preserve stochastic collision/preprocess counts.
    vector<double> remainderincoll;
    vector<double> remainderinpre;
    // Object setup and per-rank storage allocation.
// MeshparticalInitial: prepares derived solver state.
    MeshparticalInitial();
// ~MeshparticalInitial: releases owned buffers and MPI helper state.
    ~MeshparticalInitial();
// MeshparticalInitial (mesh, mess, mpass, mpiCtx): prepares derived solver state.
    MeshparticalInitial(meshImport *mesh, meshMessage mess, MessagePassing *mpass, const MpiContext& mpiCtx);
// module_variables: performs one solver support operation.
    void module_variables();
// allocateOwnedParticleBuckets: updates particles or particle-derived state.
    void allocateOwnedParticleBuckets();
    // Particle-number correction keeps integer counts consistent with targets.
// Iround (x): performs one solver support operation.
    int Iround(double x);
// adjust_particles (regions, size, delta): updates particles or particle-derived state.
    void adjust_particles(vector<double>& regions, int size, int delta);
// Np_tri_inititial (i, Np_tri, Np_exa): works with mesh topology or geometric intersections.
    void Np_tri_inititial(int i, vector<double>& Np_tri, int Np_exa);
// meshfillpar (i, Np_tri): performs one solver support operation.
    void meshfillpar(int i,vector<double>& Np_tri);
    // Collision constants are initialized before particles enter the DSMC loop.
    // Particle creation and velocity/location sampling routines.
// Collision_constant_initial: updates particles or particle-derived state.
    void Collision_constant_initial();
// CreateParticlesnode (j, i, index0, index1, index2, center): works with mesh topology or geometric intersections.
    void CreateParticlesnode(int j,int i, int index0, int index1, int index2, const double center[DIM]);
// CreateParticlesnode (j, i, index0, index1, index2, index3): works with mesh topology or geometric intersections.
    void CreateParticlesnode(int j,int i, int index0, int index1, int index2, int index3);
// velocityLocationGenerate (i, index0, index1, index2, center): updates particles or particle-derived state.
    particle velocityLocationGenerate(int i, int index0, int index1, int index2, const double center[DIM]);
// velocityLocationGenerate (i, index0, index1, index2, index3): updates particles or particle-derived state.
    particle velocityLocationGenerate(int i, int index0, int index1, int index2, int index3);
    // Basic vector math helpers used by tetrahedral sampling.
// crossProduct (vecA, vecB, result): performs one solver support operation.
    void crossProduct(const double* vecA, const double* vecB, double* result);
// dotProduct (vecA, vecB, size): performs one solver support operation.
    double dotProduct(const double* vecA, const double* vecB, int size);
};
