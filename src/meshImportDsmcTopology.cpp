#include "meshImport.h"

#include <cfloat>
#include <cmath>

namespace
{
static void point3_sub(const double* a, const double* b, double r[3])
{
    r[0] = a[0] - b[0];
    r[1] = a[1] - b[1];
    r[2] = a[2] - b[2];
}
static double point3_dot(const double a[3], const double b[3])
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}
static void point3_cross(const double a[3], const double b[3], double r[3])
{
    r[0] = a[1]*b[2] - a[2]*b[1];
    r[1] = a[2]*b[0] - a[0]*b[2];
    r[2] = a[0]*b[1] - a[1]*b[0];
}
static double point3_norm(const double a[3])
{
    return std::sqrt(point3_dot(a, a));
}
static bool load_quad_face_points_global(const DsmcEdge& face,
                                         const meshImport& mesh,
                                         double P[4][3])
{
    if (face.faceType != 4) return false;
    for (int i = 0; i < 4; ++i)
    {
        const int pid = face.faceMap[i];
        if (pid < 0 || (mesh.Npoint > 0 && pid >= mesh.Npoint)) return false;
        for (int d = 0; d < 3; ++d)
        {
            if (mesh.pointXY != NULL)
            {
                if (mesh.pointXY[pid] == NULL) return false;
                P[i][d] = mesh.pointXY[pid][d];
            }
            else
            {
                const size_t idx = (size_t)DIM * (size_t)pid + (size_t)d;
                if (idx >= mesh.localPointXY.size()) return false;
                P[i][d] = mesh.localPointXY[idx];
            }
            if (!std::isfinite(P[i][d])) return false;
        }
    }
    return true;
}
static bool triangle_normal_from_quad_points(const double P[4][3],
                                             const int tri[3],
                                             double normal[3],
                                             double& normalLen)
{
    double e1[3], e2[3];
    point3_sub(P[tri[1]], P[tri[0]], e1);
    point3_sub(P[tri[2]], P[tri[0]], e2);
    point3_cross(e1, e2, normal);
    normalLen = point3_norm(normal);
    return std::isfinite(normalLen) && normalLen > 1.0e-30;
}
static double quad_split_quality_score(const double P[4][3],
                                       const int tri0[3],
                                       const int tri1[3])
{
    double n0[3], n1[3];
    double len0 = 0.0, len1 = 0.0;
    if (!triangle_normal_from_quad_points(P, tri0, n0, len0)) return DBL_MAX;
    if (!triangle_normal_from_quad_points(P, tri1, n1, len1)) return DBL_MAX;
    double cosAngle = point3_dot(n0, n1) / (len0 * len1);
    if (cosAngle < -1.0) cosAngle = -1.0;
    if (cosAngle >  1.0) cosAngle =  1.0;
    const double normalPenalty = 1.0 - cosAngle;
    const double areaPenalty = std::fabs(len0 - len1) / (len0 + len1);
    return normalPenalty * 16.0 + areaPenalty;
}
static bool choose_face_level_quad_split_tag(const DsmcEdge& face,
                                             const meshImport& mesh,
                                             unsigned char& tag)
{
    tag = meshImport::FACE_SPLIT_02;
    double P[4][3];
    if (!load_quad_face_points_global(face, mesh, P)) return false;
    const int tri02a[3] = {0, 1, 2};
    const int tri02b[3] = {0, 2, 3};
    const int tri13a[3] = {0, 1, 3};
    const int tri13b[3] = {1, 2, 3};
    const double score02 = quad_split_quality_score(P, tri02a, tri02b);
    const double score13 = quad_split_quality_score(P, tri13a, tri13b);
    const bool valid02 = (score02 < DBL_MAX * 0.5);
    const bool valid13 = (score13 < DBL_MAX * 0.5);
    if (!valid02 && !valid13) return false;
    if (!valid02 || (valid13 && score13 + 1.0e-12 < score02))
        tag = meshImport::FACE_SPLIT_13;
    else
        tag = meshImport::FACE_SPLIT_02;
    return true;
}
} 

void meshImport::build_face_split_tags_from_faces()
{
    const int nFace = (int)Dsmcedges.size();
    DsmcfaceSplitTag.assign((size_t)nFace, FACE_SPLIT_INVALID);
    for (int fid = 0; fid < nFace; ++fid)
    {
        const DsmcEdge& f = Dsmcedges[(size_t)fid];
        if (f.faceType != 4) continue;
        unsigned char tag = FACE_SPLIT_02;
        if (!choose_face_level_quad_split_tag(f, *this, tag))
            tag = FACE_SPLIT_02;
        DsmcfaceSplitTag[(size_t)fid] = tag;
    }
}

void meshImport::rootCaptureGlobalDsmcAndReleaseMesh()
{
    Dsmccells.resize((size_t)this->Ncell);
    for (int gid = 0; gid < this->Ncell; ++gid)
    {
        const cell& c = this->cells[gid];
        DsmcCell dc;
        dc.num = c.num;
        dc.fluentCellType = c.rawCellType;
        dc.area = c.area;
        dc.no = c.no;
        for (int k = 0; k < NN; ++k)
        {
            dc.cell2face[k] = c.cell2face[k];
            dc.cell2cell[k] = c.cell2cell[k];
            dc.cell2face_sgn[k] = c.cell2face_sgn[k];
        }
        for (int d = 0; d < DIM; ++d) dc.cellXY[d] = c.cellXY[d];
        Dsmccells[(size_t)gid] = dc;
    }
    Dsmcedges.resize((size_t)this->Nface);
    for (int fid = 0; fid < this->Nface; ++fid)
    {
        const edge& e = this->edges[fid];
        DsmcEdge de;
        de.faceTag = e.faceTag;
        de.faceType = e.faceType;
        de.length = e.length;
        for (int k = 0; k < NN; ++k) de.faceMap[k] = e.faceMap[k];
        for (int d = 0; d < DIM; ++d)
        {
            de.edgeNormal[d] = e.edgeNormal[d];
            de.edgeCenter[d] = e.edgeCenter[d];
        }
        Dsmcedges[(size_t)fid] = de;
    }
    build_face_split_tags_from_faces();
    localPointXY.resize((size_t)DIM * (size_t)this->Npoint);
    for (int pid = 0; pid < this->Npoint; ++pid)
    {
        localPointXY[(size_t)DIM*pid + 0] = this->pointXY[pid][0];
        localPointXY[(size_t)DIM*pid + 1] = this->pointXY[pid][1];
        localPointXY[(size_t)DIM*pid + 2] = this->pointXY[pid][2];
    }
    if (this->cells) { delete[] this->cells; this->cells = NULL; }
    if (this->edges) { delete[] this->edges; this->edges = NULL; }
    if (this->pointXY)
    {
        for (int pid = 0; pid < this->Npoint; ++pid)
            delete[] this->pointXY[pid];
        delete[] this->pointXY;
        this->pointXY = NULL;
    }
}
