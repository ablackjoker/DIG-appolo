/*
 * Mesh constants, geometry records, and meshImport public API.
 */

#pragma once
#include <string.h>
#include <vector>
#include <array>
#include <algorithm>
#include <cstdio>
#include <cstring>

#include <unordered_map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <queue>
#include <metis.h>
#include <cfloat>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

// Basic scalar aliases and mesh dimension constants.
#define mint int
#define mdouble double
#define BASE 16
#define DIM 3
#define NK 6
#define NN 6 
#define NV 8 

// Global Apollo case limits and DSMC run length.
const int MAXSIZE = 4e5; 
const int VAR = 6;       
const int VAR2 = 12;     
const int VAR3 = 6;      
const int NCELL = 372500;    
const int NTOTAL = 10000; 
const int Npinitial = 100;
const int NSS = 5000;
const int NDSIPLAY = 50;
// Optional boundary stress/heat output cadence.
const bool EnableBoundaryStressHeatStatistic = false;
const int NBoundaryStressHeatOutputEvery = 1000;
// GSIS coupling cadence and activation controls.
const int Nevery = 1;
const int Nrepeat = 100;
const int NGSIS = Nevery*Nrepeat;
const int ifvary = 1;
const int NSCHEME = 2000;
const int ifgsis = 1;
// High-order DSMC-to-NS data is ramped during the transition window.
const bool EnableHighOrderRamp = true;
const int NHIGH_ORDER_RAMP_START = NSCHEME;
const int NHIGH_ORDER_RAMP_END = NSS;
const double HIGH_ORDER_RAMP_MIN = 0.0;
// Final DSMC records can be used after the startup delay.
const bool EnableDsmc2NsFinalRecord = true;
const int NFINAL_RECORD_DELAY = NSCHEME;

// DsmcReservoirBoundaryState stores state used by this module.
enum DsmcReservoirBoundaryState
{
    DSMC_RESERVOIR_INLET = 0,
    DSMC_RESERVOIR_OUTLET = 1
};

// DsmcReservoirBoundaryConfig stores state used by this module.
struct DsmcReservoirBoundaryConfig
{
    bool enabled;
    bool injectParticles;
    bool useFreestreamUx;
    double freestreamUxScale;
    double rho;
    double ux;
    double uy;
    double uz;
    bool useInletTemperature;
    double temperature;
};

const DsmcReservoirBoundaryConfig DsmcInletReservoir = {
    // Enable inflow particle creation from the freestream state.
    true,  
    true,  
    true,  
    1.0,   
    1.0,   
    0.0,   
    0.0,   
    0.0,   
    true,  
    1.0    
};

const DsmcReservoirBoundaryConfig DsmcOutletReservoir = {
    // Outlet reservoir defaults are passive unless explicitly enabled.
    false, 
    false, 
    true,  
    1.0,   
    1.0,   
    0.0,   
    0.0,   
    0.0,   
    true,  
    1.0    
};

// ZONE stores state used by this module.
struct ZONE
{ 
    int id, firstidx, lastidx;
// setvalue (x1, x2, x3): performs one solver support operation.
    void setvalue(int x1, int x2, int x3){
        id = x1;
        firstidx = x2;
        lastidx = x3;
    }
};

// fNode stores state used by this module.
struct fNode
{
    int fid, no;
};

// cellNode stores state used by this module.
struct cellNode
{
    int id = -1, no = 100000;
};

// cell stores state used by this module.
struct cell
{
    int num, dim; 
    int cell2node[NV];
    int cell2face[NN] = {-1, -1, -1, -1, -1, -1};
    int cell2cell[NN] = {-1, -1, -1, -1, -1, -1};
    int cell2face_sgn[NN] = {-1, -1, -1, -1, -1, -1};
    double area, cellLengthEff;
    double Ainv[DIM][DIM]; 
    double dxyz[NN][DIM];
    int cellType, rawCellType = 0, no;
    double cellXY[DIM];
};

// edge stores state used by this module.
struct edge
{
    int no, faceTag, dim, faceType, bcType;
    double length;
    int faceMap[NN];
    double edgeCenter[DIM], edgeNormal[DIM], edgeDist, edgerij[DIM], edgerL[DIM], edgerR[DIM];
};

// DsmcCell stores state used by this module.
struct DsmcCell
{
    int num = 0;
    int fluentCellType = 0;
    int cell2face[NN] = {-1,-1,-1,-1,-1,-1};
    int cell2cell[NN] = {-1,-1,-1,-1,-1,-1};
    int cell2face_sgn[NN] = {-1,-1,-1,-1,-1,-1};
    double area = 0.0;
    int no = -1;
    double cellXY[DIM] = {0.0,0.0,0.0};
};

// DsmcEdge stores state used by this module.
struct DsmcEdge
{
    int faceTag = 0;
    int faceType = 0;
    double length = 0.0;
    int faceMap[NN] = {-1,-1,-1,-1,-1,-1};  
    double edgeNormal[DIM] = {0.0,0.0,0.0};
    double edgeCenter[DIM];
};

// meshMessage stores state used by this module.
struct meshMessage
{
    int Ncell, Nface, Npoint, Nk, bcnumber = 0, Nghost = 0;
    double Area = 0;
    double Ma = 3, Kn = 0.05, cfl = 0.62; 
    double gamma, dr = 2, dv = 0, zr = 2.226, zv = 0, omega = 0.74;
    int var = 6, qn = 3; 
    double f_tra = 2.261, f_rot = 1.499;
    double delta_rp, delta_sm, omeag0, omeag1;
    double cfl_ns, pr, cp, cv;
    double kB,p_mass,p_mass_r,T_ref,n_ref,miu_ref,Twall_ref,p_ref,d_ref;
    double alpha,v_rms,eta,P_relax,dt_ref;
    double T_in, v_in;
    int eNface,iNface;
    double Neff;
    double dtime;
};

// meshImport stores state used by this module.
class meshImport
{
private:
public:
    // Case scaling and Fluent zone metadata.
    double ScaleFactor = 1.0;
    vector<ZONE> zonemap;
    // Global mesh sizes and boundary-face counts.
    int dim = DIM;
    int Ncell, Nface, Npoint, Nk, bcnumber = 0, Nghost = 0, eNface = 0, iNface = 0;
    int edge1stLocation = 0; 
    // Reference flow parameters used to build meshMessage.
    double Area = 0;
    double Ma = 3, Kn = 0.05, cfl = 0.62; 
    double gamma, dr = 2, dv = 0, zr = 2.226, zv = 0, omega = 0.74;
    int var = 6, qn = 2; 
    // Velocity-space and timestep controls.
    int xn = 42, yn = 36, zn = 36; 
    double maxv = 15, cfl_psu;
    double f_tra = 2.261, f_rot = 1.499;
    double delta_rp, delta_sm, omeag0, omeag1;
    double cfl_ns = 5.0, pr, cp, cv;
    // Raw Fluent arrays and compact DSMC geometry storage.
    int *TAG = NULL, *NTAG = NULL;
    double **pointXY = NULL, **cellXY = NULL, **cellXYGhost = NULL;
    vector<int> ghost2cell;
    cell *cells = NULL; edge *edges = NULL;
    vector<cell> originalCells;
    vector<edge> originalEdges;
    vector<double> localPointXY;
    vector<DsmcCell> Dsmccells;
    vector<DsmcEdge> Dsmcedges;
    vector<unsigned char> DsmcfaceSplitTag;
    // MPI context and partition index maps.
    MPI_Comm comm;
    MPI_Comm calGroup;
    meshMessage mess;
    int *reMeshIndex = NULL, *reMeshIndex2 = NULL;
    int *Npc_exa = NULL;
    int *startGrid = NULL, *endGrid = NULL;
    int rank, c_size;
    // Derived DSMC particle and timestep scales.
    double miniLength;
    int Nps;
    int Npc_initial;
    double Neff;
    double dtime;
    // Construction and lifetime management.
// meshImport: performs one solver support operation.
    meshImport();
// meshImport (filePath, nParts, sfactor): performs one solver support operation.
    meshImport(const char *filePath, idx_t nParts, double sfactor);
// meshImport (filePath, tag, nParts, sfactor): performs one solver support operation.
    meshImport(const char *filePath, bool tag, idx_t nParts, double sfactor);
// ~meshImport: releases owned buffers and MPI helper state.
    ~meshImport();
    // Fluent CAS parsing entry points.
// readCAS (filePath): parses mesh or configuration input.
    bool readCAS(const char *filePath);
// hex2dcm (str): parses mesh or configuration input.
    int hex2dcm(string str);  
// classification (str, sst, infile, commentLoc): parses mesh or configuration input.
    void classification(string str, stringstream &sst, ifstream &infile, stringstream &commentLoc);
// readFaces (sst, infile): parses mesh or configuration input.
    bool readFaces(stringstream &sst, ifstream &infile);
// readCells (sst, infile): parses mesh or configuration input.
    bool readCells(stringstream &sst, ifstream &infile);
// readNodes (sst, infile): parses mesh or configuration input.
    bool readNodes(stringstream &sst, ifstream &infile);
// readPeriodic (sst, infile): parses mesh or configuration input.
    bool readPeriodic(stringstream &sst, ifstream &infile);
// preprocess (nParts): performs one solver support operation.
    bool preprocess(idx_t nParts);   
// setFaceCellNew: works with mesh topology or geometric intersections.
    bool setFaceCellNew();
// getZoneid (comment): performs one solver support operation.
    bool getZoneid(stringstream &comment);
    // Tecplot and Fluent output helpers.
// out2dat (filePath, w, sq): writes solver fields or diagnostics.
    bool out2dat(const char *filePath, const double w[][NCELL], const double sq[][VAR2]);
// out2Fluent_ns2 (filePath, qf): writes solver fields or diagnostics.
    bool out2Fluent_ns2(const char *filePath, double *qf);
// out2Fluent_dsmc (filePath, w): writes solver fields or diagnostics.
    bool out2Fluent_dsmc(const char *filePath, double *w);
// out2Fluent_ns6 (filePath, w): writes solver fields or diagnostics.
    bool out2Fluent_ns6(const char *filePath, double *w);
    enum : unsigned char
    {
        FACE_SPLIT_INVALID = 255,
        FACE_SPLIT_02 = 0,   
        FACE_SPLIT_13 = 1    
    };
    // Quadrilateral faces are split consistently for DSMC triangle tracing.
// build_face_split_tags_from_faces: works with mesh topology or geometric intersections.
    void build_face_split_tags_from_faces();
// decode_quad_split_tag (tag, tri0, tri1): performs one solver support operation.
    static inline void decode_quad_split_tag(unsigned char tag, int tri0[3], int tri1[3])
    {
        if (tag == FACE_SPLIT_02)
        {
            tri0[0] = 0; tri0[1] = 1; tri0[2] = 2;
            tri1[0] = 0; tri1[1] = 2; tri1[2] = 3;
        }
        else if (tag == FACE_SPLIT_13)
        {
            tri0[0] = 0; tri0[1] = 1; tri0[2] = 3;
            tri1[0] = 1; tri1[1] = 2; tri1[2] = 3;
        }
        else 
        {
            tri0[0] = tri0[1] = tri0[2] = -1;
            tri1[0] = tri1[1] = tri1[2] = -1;
        }
    }
// mutualMapping (pl, pr, vis, len): performs one solver support operation.
    bool mutualMapping(int *pl, int *pr, vector<int> vis, int &len);
    // Physical scaling and compact message construction.
// setMa_Kn_CFL (ma, kn, cfl, cfl_psu, cfl_ns): performs one solver support operation.
    void setMa_Kn_CFL(double ma, double kn, double cfl, double cfl_psu, double cfl_ns);
// setMeshMessage: performs one solver support operation.
    void setMeshMessage();
// setParticleMessage: updates particles or particle-derived state.
    void setParticleMessage();
// calGridpar: performs one solver support operation.
    void calGridpar();
// norm: performs one solver support operation.
    void norm();
// cross: performs one solver support operation.
    void cross();
    // Small matrix utilities used by cell metric preprocessing.
// calcMatrixInversion (src, n, des): works with mesh topology or geometric intersections.
    bool calcMatrixInversion(double src[DIM][DIM], int n, double des[DIM][DIM]);
// calcDeterminant (arcs, n): prepares derived solver state.
    double calcDeterminant(double arcs[DIM][DIM],int n);
// calcCofactor (arcs, n, ans): prepares derived solver state.
    void  calcCofactor(double arcs[DIM][DIM],int n,double ans[DIM][DIM]);
    // Mesh conversion, partitioning, and node cleanup helpers.
// changeFluent (filePath, tag): performs one solver support operation.
    int changeFluent(const char *filePath, bool tag); 
// metisPartition (nParts): updates partition ownership or load data.
    void metisPartition(idx_t nParts);  
// reMesh: performs one solver support operation.
    void reMesh();
// out2VTKw (filePath, w): writes solver fields or diagnostics.
    bool out2VTKw(const char *filePath, const double *w);
// getNode: works with mesh topology or geometric intersections.
    void getNode();
// deleteRepeat (Index, temp, cellType): performs one solver support operation.
    void deleteRepeat(int Index, int *temp, int cellType);
// rootCaptureGlobalDsmcAndReleaseMesh: performs one solver support operation.
    void rootCaptureGlobalDsmcAndReleaseMesh();
private:
// ensureReMeshIndexIdentity (includeReverse): prepares derived solver state.
    void ensureReMeshIndexIdentity(bool includeReverse = false);
};
