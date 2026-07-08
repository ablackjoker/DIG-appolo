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

#define mint int
#define mdouble double
#define BASE 16
#define DIM 3
#define NK 6
#define NN 6 
#define NV 8 



const int MAXSIZE = 4e5; 
const int VAR = 6;       
const int VAR2 = 12;     
const int VAR3 = 6;      
const int NCELL = 372500;    
const int NTOTAL = 10000; 
const int Npinitial = 100;
const int NSS = 5000;
const int NDSIPLAY = 50;
const bool EnableBoundaryStressHeatStatistic = false;
const int NBoundaryStressHeatOutputEvery = 1000;
const int Nevery = 1;
const int Nrepeat = 100;
const int NGSIS = Nevery*Nrepeat;
const int ifvary = 1;
const int NSCHEME = 2000;
const int ifgsis = 1;
const bool EnableHighOrderRamp = true;
const int NHIGH_ORDER_RAMP_START = NSCHEME;
const int NHIGH_ORDER_RAMP_END = NSS;
const double HIGH_ORDER_RAMP_MIN = 0.0;
const bool EnableDsmc2NsFinalRecord = true;
const int NFINAL_RECORD_DELAY = NSCHEME;

enum DsmcReservoirBoundaryState
{
    DSMC_RESERVOIR_INLET = 0,
    DSMC_RESERVOIR_OUTLET = 1
};

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

struct ZONE
{ 
    int id, firstidx, lastidx;
    void setvalue(int x1, int x2, int x3){
        id = x1;
        firstidx = x2;
        lastidx = x3;
    }
};

struct fNode
{
    int fid, no;
};

struct cellNode
{
    int id = -1, no = 100000;
};


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

struct edge
{
    int no, faceTag, dim, faceType, bcType;
    double length;
    int faceMap[NN];
    
    double edgeCenter[DIM], edgeNormal[DIM], edgeDist, edgerij[DIM], edgerL[DIM], edgerR[DIM];
};


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

struct DsmcEdge
{
    int faceTag = 0;
    int faceType = 0;
    double length = 0.0;
    int faceMap[NN] = {-1,-1,-1,-1,-1,-1};  
    double edgeNormal[DIM] = {0.0,0.0,0.0};
    double edgeCenter[DIM];
};


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

class meshImport
{
private:
    
public:
    double ScaleFactor = 1.0;
    vector<ZONE> zonemap;
    int dim = DIM;
    int Ncell, Nface, Npoint, Nk, bcnumber = 0, Nghost = 0, eNface = 0, iNface = 0;
    int edge1stLocation = 0; 
    double Area = 0;
    double Ma = 3, Kn = 0.05, cfl = 0.62; 
    double gamma, dr = 2, dv = 0, zr = 2.226, zv = 0, omega = 0.74;
    int var = 6, qn = 2; 
    int xn = 42, yn = 36, zn = 36; 

    double maxv = 15, cfl_psu;
    double f_tra = 2.261, f_rot = 1.499;
    double delta_rp, delta_sm, omeag0, omeag1;
    double cfl_ns = 5.0, pr, cp, cv;

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

    MPI_Comm comm;
    MPI_Comm calGroup;

    meshMessage mess;

    int *reMeshIndex = NULL, *reMeshIndex2 = NULL;

    int *Npc_exa = NULL;
    int *startGrid = NULL, *endGrid = NULL;
    int rank, c_size;
    
    double miniLength;
    int Nps;
    int Npc_initial;
    double Neff;
    double dtime;
    
    meshImport();
    meshImport(const char *filePath, idx_t nParts, double sfactor);
    meshImport(const char *filePath, bool tag, idx_t nParts, double sfactor);
    ~meshImport();
    bool readCAS(const char *filePath);
    int hex2dcm(string str);  
    void classification(string str, stringstream &sst, ifstream &infile, stringstream &commentLoc);
    bool readFaces(stringstream &sst, ifstream &infile);
    bool readCells(stringstream &sst, ifstream &infile);
    bool readNodes(stringstream &sst, ifstream &infile);
    bool readPeriodic(stringstream &sst, ifstream &infile);
    bool preprocess(idx_t nParts);   
    bool setFaceCellNew();
    bool getZoneid(stringstream &comment);


    bool out2dat(const char *filePath, const double w[][NCELL], const double sq[][VAR2]);
 
 
    bool out2Fluent_ns2(const char *filePath, double *qf);
 
 
    bool out2Fluent_dsmc(const char *filePath, double *w);
    bool out2Fluent_ns6(const char *filePath, double *w);
  
    enum : unsigned char
    {
        FACE_SPLIT_INVALID = 255,
        FACE_SPLIT_02 = 0,   
        FACE_SPLIT_13 = 1    
    };

    void build_face_split_tags_from_faces();

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

    bool mutualMapping(int *pl, int *pr, vector<int> vis, int &len);
    void setMa_Kn_CFL(double ma, double kn, double cfl, double cfl_psu, double cfl_ns);
    void setMeshMessage();
    void setParticleMessage();
    void calGridpar();
    void norm();
    void cross();


    bool calcMatrixInversion(double src[DIM][DIM], int n, double des[DIM][DIM]);

    double calcDeterminant(double arcs[DIM][DIM],int n);

    void  calcCofactor(double arcs[DIM][DIM],int n,double ans[DIM][DIM]);

    int changeFluent(const char *filePath, bool tag); 
  


    void metisPartition(idx_t nParts);  


    void reMesh();

    bool out2VTKw(const char *filePath, const double *w);
    void getNode();
    void deleteRepeat(int Index, int *temp, int cellType);
    void rootCaptureGlobalDsmcAndReleaseMesh();
private:
    void ensureReMeshIndexIdentity(bool includeReverse = false);
};
