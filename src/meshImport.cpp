/*
 * Fluent CAS parsing, mesh preprocessing, partition data, and writers.
 */

#include "meshImport.h"

/*
 * meshImport: initializes meshImport state.
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
meshImport::meshImport()
{
}

/*
 * meshImport: initializes meshImport state.
 * Params: filePath, nParts, sfactor; returns: none.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
meshImport::meshImport(const char *filePath, idx_t nParts, double sfactor)
{
    this->ScaleFactor = sfactor;
    if (this->readCAS(filePath))
    {
        this->preprocess(nParts);
    }
}

/*
 * meshImport: initializes meshImport state.
 * Params: filePath, tag, nParts, sfactor; returns: none.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
meshImport::meshImport(const char *filePath, bool tag, idx_t nParts, double sfactor)
{
    this->iNface = this->changeFluent(filePath, tag);
    this->ScaleFactor = sfactor;
    if (this->readCAS(filePath))
    {
        this->preprocess(nParts);
    }
}

/*
 * ~meshImport: releases owned buffers and MPI helper state.
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
meshImport::~meshImport()
{
    if (this->TAG != NULL)
    {
        delete[] this->TAG;
    }
    if (this->pointXY != NULL)
    {
        for (int i = 0; i < Npoint; i++)
        {
            delete[] this->pointXY[i];
        }
        delete[] this->pointXY;
    }
    if (this->cellXY != NULL)
    {
        for (int i = 0; i < Ncell; i++)
        {
            delete[] this->cellXY[i];
        }
        delete[] this->cellXY;
    }
    if (this->cellXYGhost != NULL)
    {
        for (int i = 0; i < Nghost; i++)
        {
            delete[] this->cellXYGhost[i];
        }
        delete[] this->cellXYGhost;
    }
    if (this->cells != NULL)
    {
        delete[] this->cells;
    }
    if (this->edges != NULL)
    {
        delete[] this->edges;
    }
    if(this->reMeshIndex != NULL){
        delete[] this->reMeshIndex;
        delete[] this->reMeshIndex2;
    }
    if(this->Npc_exa != NULL){
        delete[] this->Npc_exa;
    }
    if(this->startGrid != NULL){
        delete[] this->startGrid;
        delete[] this->endGrid;
    }
}

/*
 * ensureReMeshIndexIdentity: prepares derived solver state.
 * Params: includeReverse; returns: none.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
void meshImport::ensureReMeshIndexIdentity(bool includeReverse)
{
    const int ncell = this->Ncell;
    bool initForward = false;
    bool initReverse = false;
    if (this->reMeshIndex == NULL)
    {
        this->reMeshIndex = new int[ncell];
        initForward = true;
    }
    if (includeReverse && this->reMeshIndex2 == NULL)
    {
        this->reMeshIndex2 = new int[ncell];
        initReverse = true;
    }
    for (int i = 0; i < ncell; i++)
    {
        if (initForward)
            this->reMeshIndex[i] = i;
        if (initReverse)
            this->reMeshIndex2[i] = i;
    }
}

/*
 * readCAS: parses mesh or configuration input.
 * Params: filePath; returns: success or decision flag.
 * Flow:
 *   - decode the input record.
 *   - fill mesh arrays.
 *   - report parse status.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
bool meshImport::readCAS(const char *filePath)
{
    ifstream infile;
    string s;
    infile.open(filePath, ios::in);
    if (!infile.is_open())
    {
        printf("%s, open failed !", filePath);
        return false;
    }
    stringstream sst;
    stringstream comment;
    while (getline(infile, s))
    {
        comment.clear();
        comment.str(s);  
        sst.clear();
        sst.str(s);  
        sst >> s;
        classification(s, sst, infile, comment);
    }
    infile.close();
    return true;
}

/*
 * classification: parses mesh or configuration input.
 * Params: str, sst, infile, commentLoc; returns: none.
 * Flow:
 *   - decode the input record.
 *   - fill mesh arrays.
 *   - report parse status.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
void meshImport::classification(string str, stringstream &sst, ifstream &infile, stringstream &commentLoc)
{
    string s = "";
    if (!str.compare("(2"))
    {
        sst >> s;
        sscanf(s.c_str(), "%d)", &this->dim);
        return;
    }
    else if (!str.compare("(10")) 
    {
        this->readNodes(sst, infile);
        return;
    }
    else if (!str.compare("(12")) 
    {
        this->getZoneid(commentLoc); 
        this->readCells(sst, infile);
        return;
    }
    else if (!str.compare("(13")) 
    {
        this->getZoneid(commentLoc); 
        this->readFaces(sst, infile); 
        return;
    }else if(!str.compare("(18")){
        this->readPeriodic(sst, infile);
        return;
    }
    else if (!str.compare("(45"))
    {
    }
    else
    {
        return;
    }
}

/*
 * readNodes: parses mesh or configuration input.
 * Params: sst, infile; returns: success or decision flag.
 * Flow:
 *   - decode the input record.
 *   - fill mesh arrays.
 *   - report parse status.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
bool meshImport::readNodes(stringstream &sst, ifstream &infile)
{
    string str = "", st = "";
    sst >> st; 
    sst >> str; 
    int l = hex2dcm(str); 
    sst >> str;
    int r = hex2dcm(str); 
    if (!st.compare("(0")){
        sst >> st >> str; 
        this->Npoint = r;
        this->pointXY = new mdouble *[r]; 
        for (int i = 0; i < r; i++){
            this->pointXY[i] = new mdouble[this->dim];
        }
        return true;
    }else if(!st.compare("(labels")){
        return true;
    }
    int step = l - 1;
    while (getline(infile, str) && step < r){ 
        if(!str.compare("(") || str.length() <= 2){
            continue;
        }
        sst.clear();
        sst.str(str); 
        sst >> this->pointXY[step][0] >> this->pointXY[step][1] >> this->pointXY[step][2];
        this->pointXY[step][0] *= ScaleFactor;
        this->pointXY[step][1] *= ScaleFactor;
        this->pointXY[step][2] *= ScaleFactor;
        step++;
    }
    return true;
}

/*
 * readCells: parses mesh or configuration input.
 * Params: sst, infile; returns: success or decision flag.
 * Flow:
 *   - decode the input record.
 *   - fill mesh arrays.
 *   - report parse status.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
bool meshImport::readCells(stringstream &sst, ifstream &infile)
{
    string str = "", st = "";
    int cellTypeLoc, l, r;
    sst >> st;
    sst >> str;
    l = hex2dcm(str);
    sst >> str;
    r = hex2dcm(str);
    if (!st.compare("(0"))
    {
        this->Ncell = r;
        this->Nghost = r;
        this->cells = new cell[r];
        this->TAG = new int[r];
        for (int i = 0; i < r; i++)
        {
            this->TAG[i] = 0;
        }
        return true;
    }else if(!st.compare("(labels")){
        return true;
    }else{
        sst >> str;
        sst >> str;
        if (!str.compare("0)(")){
            int nx = l - 1;
            while (getline(infile, str) && nx < r){
                if(!str.compare("(")  || str.length() <= 1){
                    continue;
                }
                sst.clear();
                sst.str(str);
                while (nx < r && (sst >> cellTypeLoc))
                {
                    if (nx >= 0 && nx < this->Ncell)
                    {
                        this->cells[nx].cellType = cellTypeLoc;
                        this->cells[nx].rawCellType = cellTypeLoc;
                    }
                    nx++;
                }
            }
        }
        else
        {
            sscanf(str.c_str(), "%d))", &cellTypeLoc);
            for (int i = l - 1; i < r && i < this->Ncell; i++)
            {
                this->cells[i].cellType = cellTypeLoc;
                this->cells[i].rawCellType = cellTypeLoc;
            }
        }
    }
    return true;
}

/*
 * readFaces: parses mesh or configuration input.
 * Params: sst, infile; returns: success or decision flag.
 * Flow:
 *   - decode the input record.
 *   - fill mesh arrays.
 *   - report parse status.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
bool meshImport::readFaces(stringstream &sst, ifstream &infile)
{
    string str = "", st = "";
    sst >> st;  
    sst >> str; 
    int l = hex2dcm(str);
    sst >> str; 
    int r = hex2dcm(str);
    if (!st.compare("(0"))
    {
        this->Nface = r;
        this->edges = new edge[r];
        return true;
    }else if(!st.compare("(labels")){
        return true;
    }
    int TT = -1;
    TT = hex2dcm(st);
    int bcTypeLoc, faceTypeLoc, faceTag;
    string p0s, p1s, p2s, p3s, c0s, c1s;
    int p1, p2, p3, p0, c1, c0;
    sst >> str; 
    bcTypeLoc = hex2dcm(str);
    if(bcTypeLoc == 9){
        if(this->iNface == 0){
        }else if(l == 1){
            this->iNface = 0;
        }else{
            l = 1;
            r = this->iNface;
        }
    }else{
        if(this->iNface != 0 && l < this->edge1stLocation && edge1stLocation != 0){
            l += this->iNface;
            r += this->iNface;
        }
    }
    sst >> str; 
    sscanf(str.c_str(), "%d)(", &faceTag); 
    faceTypeLoc = faceTag;
    int step = l - 1;
    while (getline(infile, str) && step < r)
    {
        if(!str.compare("(") || str.length() <= 2){
            continue;
        }
        sst.clear();
        sst.str(str);
        if (faceTag == 0)
        {
            sst >> faceTypeLoc;
        }
        if (faceTypeLoc == 3) 
        {
            sst >> p0s >> p1s >> p2s >> c0s >> c1s;
            p3s = "";
        }
        else
        {
            sst >> p0s >> p1s >> p2s >> p3s >> c0s >> c1s; 
        }
        p0 = hex2dcm(p0s) - 1;
        p1 = hex2dcm(p1s) - 1;
        p2 = hex2dcm(p2s) - 1;
        p3 = hex2dcm(p3s) - 1;
        c0 = hex2dcm(c0s) - 1;
        c1 = hex2dcm(c1s) - 1;
        if (c1 < 0)
        {
            c1 = this->Nghost ++; 
            this->bcnumber++;
            this->ghost2cell.push_back(step);
        }
        this->edges[step].faceMap[0] = p0;
        this->edges[step].faceMap[1] = p1;
        this->edges[step].faceMap[2] = p2;
        this->edges[step].faceMap[3] = p3;
        this->edges[step].faceMap[4] = c0;
        this->edges[step].faceMap[5] = c1;
        this->edges[step].faceTag = TT; 
        this->edges[step].bcType = bcTypeLoc;
        this->edges[step].faceType = faceTypeLoc;
        step++;
    }
    return true;
}

/*
 * readPeriodic: parses mesh or configuration input.
 * Params: sst, infile; returns: success or decision flag.
 * Flow:
 *   - decode the input record.
 *   - fill mesh arrays.
 *   - report parse status.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
bool meshImport::readPeriodic(stringstream &sst, ifstream &infile)
{
    string str = "", st = "";
    int l, r;
    int fl, fr;
    sst >> st;
    if (!st.compare("(0"))
    {
        return false;
    }else if(!st.compare("(labels")){
        return true;
    }
    l = 1;
    sst >> str;
    r = hex2dcm(str);
    int step = l - 1;
    string f1, f2;
    while (getline(infile, str) && step < r){
        if(!str.compare("(")){
            continue;
        }
        sst.clear();
        sst.str(str);
        sst >> f1 >> f2;
        fl = hex2dcm(f1) - 1;
        fr = hex2dcm(f2) - 1;
        edges[fl].no = fr;
        edges[fr].no = fl;
        step ++;
    }
    return true;
}

/*
 * getZoneid: performs one solver support operation.
 * Params: sst; returns: success or decision flag.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
bool meshImport::getZoneid(stringstream &sst)
{
    string st = "", st1 = "", st2 = "", st3 = "";
    int zoneid, firstidx, lastidx;
    sst >> st>> st1 >> st2 >> st3; 
    if (st1.compare("(0"))
    {
        ZONE zoneloc;
        zoneid = hex2dcm(st1); 
        firstidx = hex2dcm(st2);
        lastidx = hex2dcm(st3); 
        zoneloc.setvalue(zoneid, firstidx, lastidx);
        this->zonemap.push_back(zoneloc);
    }
    return true;
}

/*
 * hex2dcm: parses mesh or configuration input.
 * Params: str; returns: index, count, owner, or status value.
 * Flow:
 *   - decode the input record.
 *   - fill mesh arrays.
 *   - report parse status.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
int meshImport::hex2dcm(string str)
{
    int len = str.size(); 
    int num = 0, temp = 0;
    for (int i = 0; i < len; i++)
    {
        if(str[i] == ')'){
            break;
        }
        if(str[i] == '('){
            continue;
        }
        if (str[i] <= 'z' && str[i] >= 'a')
        {
            temp = str[i] - 'a' + 10;
        }
        else
        {
            temp = str[i] - '0';
        }
        num = num * BASE + temp;
    }
    return num;
}

/*
 * setFaceCellNew: works with mesh topology or geometric intersections.
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
bool meshImport::setFaceCellNew() 
{
    int p1, p2, c0, c1, iNode;
    double localCentroid[DIM], localSF[DIM], Cf[DIM];
    double localArea, localVolume;
    cell *cells = this->cells;
    edge *edges = this->edges;
    for (int iFace = 0; iFace < this->Nface; iFace++)
    {
        double localCentre[DIM] = {0, 0, 0};
        for (int i = 0; i < edges[iFace].faceType; i++)
        {
            iNode = edges[iFace].faceMap[i];
            localCentre[0] += this->pointXY[iNode][0] / edges[iFace].faceType;
            localCentre[1] += this->pointXY[iNode][1] / edges[iFace].faceType;
            localCentre[2] += this->pointXY[iNode][2] / edges[iFace].faceType;
        }
        double centroid[DIM] = {0, 0, 0}, SF[DIM] = {0, 0, 0}; double area = 0;
        for (int iTriangle = 0; iTriangle < edges[iFace].faceType; iTriangle++)
        {
            p1 = edges[iFace].faceMap[iTriangle];
            if (iTriangle < edges[iFace].faceType - 1)
            {
                p2 = edges[iFace].faceMap[iTriangle + 1];
            }
            else
            {
                p2 = edges[iFace].faceMap[0];
            }
            double x1, x2, x0, y1, y2, y0, z1, z2, z0;
            double xl, xr, yl, yr, zl, zr;
            x0 = localCentre[0];
            y0 = localCentre[1];
            z0 = localCentre[2];
            x1 = pointXY[p1][0];
            y1 = pointXY[p1][1];
            z1 = pointXY[p1][2];
            x2 = pointXY[p2][0];
            y2 = pointXY[p2][1];
            z2 = pointXY[p2][2];
            localCentroid[0] = (x1 + x2 + x0) / 3;
            localCentroid[1] = (y1 + y2 + y0) / 3;
            localCentroid[2] = (z1 + z2 + z0) / 3;
            xl = x1 - x0;
            yl = y1 - y0;
            zl = z1 - z0;
            xr = x2 - x0;
            yr = y2 - y0;
            zr = z2 - z0;
            double vx, vy, vz;
            vx = yr * zl - zr * yl;
            vy = zr * xl - zl * xr;
            vz = xr * yl - xl * yr;
            vx *= 0.5;
            vy *= 0.5;
            vz *= 0.5;
            localSF[0] = vx;
            localSF[1] = vy;
            localSF[2] = vz;
            localArea = sqrt(vx*vx + vy*vy + vz*vz);
            centroid[0] += localArea * localCentroid[0];
            centroid[1] += localArea * localCentroid[1];
            centroid[2] += localArea * localCentroid[2];
            SF[0] += localSF[0];
            SF[1] += localSF[1];
            SF[2] += localSF[2];
            area += localArea; 
        }
        centroid[0] /= area;
        centroid[1] /= area;
        centroid[2] /= area;
        double dS = sqrt(SF[0] * SF[0] + SF[1] * SF[1] + SF[2] * SF[2]);
        this->edges[iFace].length = area;
        this->edges[iFace].edgeNormal[0] = SF[0] / dS;
        this->edges[iFace].edgeNormal[1] = SF[1] / dS;
        this->edges[iFace].edgeNormal[2] = SF[2] / dS;
        this->edges[iFace].edgeCenter[0] = centroid[0];
        this->edges[iFace].edgeCenter[1] = centroid[1];
        this->edges[iFace].edgeCenter[2] = centroid[2];
    }
    for (int iFace=0; iFace<this->Nface; iFace++)
    {
        c0 = edges[iFace].faceMap[4]; 
        c1 = edges[iFace].faceMap[5];
        if (c1<this->Ncell)
        {
            cells[c1].cell2cell[TAG[c1]] = c0;
            cells[c1].cell2face[TAG[c1]] = iFace;
            cells[c1].cell2face_sgn[TAG[c1]] = -1;
            TAG[c1]++;
        }
        cells[c0].cell2cell[TAG[c0]] = c1;
        cells[c0].cell2face[TAG[c0]] = iFace;
        cells[c0].cell2face_sgn[TAG[c0]] = 1;
        TAG[c0]++;
    }
    for(int i=0; i<this->Ncell; i++){
        cells[i].cellType = TAG[i];
    }
    for (int iCell = 0; iCell<this->Ncell; iCell++)
    {
        double localCentre[DIM] = {0, 0, 0};
        for (int j = 0; j < this->TAG[iCell]; j++)
        {
            int faceIndex = cells[iCell].cell2face[j];
            localCentre[0] += edges[faceIndex].edgeCenter[0]/this->TAG[iCell];
            localCentre[1] += edges[faceIndex].edgeCenter[1]/this->TAG[iCell];
            localCentre[2] += edges[faceIndex].edgeCenter[2]/this->TAG[iCell];
        }
        double localVolumeCentroidSum[DIM] = {0, 0, 0};
        double localVolumeSum = 0;
        for (int j = 0; j<this->TAG[iCell]; j++)
        {
            int faceIndex = cells[iCell].cell2face[j];
            Cf[0] = edges[faceIndex].edgeCenter[0] - localCentre[0];
            Cf[1] = edges[faceIndex].edgeCenter[1] - localCentre[1];
            Cf[2] = edges[faceIndex].edgeCenter[2] - localCentre[2];
            localSF[0] = edges[faceIndex].edgeNormal[0];
            localSF[1] = edges[faceIndex].edgeNormal[1];
            localSF[2] = edges[faceIndex].edgeNormal[2];
            localVolume = fabs(localSF[0]*Cf[0] + localSF[1]*Cf[1] + localSF[2]*Cf[2]) * edges[faceIndex].length / 3;
            localCentroid[0] = 0.75*edges[faceIndex].edgeCenter[0] + 0.25*localCentre[0];
            localCentroid[1] = 0.75*edges[faceIndex].edgeCenter[1] + 0.25*localCentre[1];
            localCentroid[2] = 0.75*edges[faceIndex].edgeCenter[2] + 0.25*localCentre[2];
            localVolumeCentroidSum[0] += localCentroid[0]*localVolume;
            localVolumeCentroidSum[1] += localCentroid[1]*localVolume;
            localVolumeCentroidSum[2] += localCentroid[2]*localVolume;
            localVolumeSum += localVolume;
        }
        cells[iCell].area = localVolumeSum; 
        this->cellXY[iCell][0] = localVolumeCentroidSum[0] / localVolumeSum;
        this->cellXY[iCell][1] = localVolumeCentroidSum[1] / localVolumeSum;
        this->cellXY[iCell][2] = localVolumeCentroidSum[2] / localVolumeSum;
        cells[iCell].cellXY[0] = this->cellXY[iCell][0];
        cells[iCell].cellXY[1] = this->cellXY[iCell][1];
        cells[iCell].cellXY[2] = this->cellXY[iCell][2];
    }
    return true;
}

/*
 * preprocess: performs one solver support operation.
 * Params: nParts; returns: success or decision flag.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
bool meshImport::preprocess(idx_t nParts)
{
    this->cellXY = new mdouble *[this->Ncell];
    this->cellXYGhost = new mdouble *[this->Nghost];
    for (int i = 0; i < this->Ncell; i++){
        this->cellXY[i] = new mdouble[this->dim];
        this->cellXYGhost[i] = new mdouble[this->dim];
        this->cells[i].area = 0;
    }
    this->setFaceCellNew();
    this->metisPartition(nParts);
    for (int i = 0; i < this->Ncell; i++){
        for (int j = 0; j < DIM; j++){
            this->cellXYGhost[i][j] = this->cellXY[i][j];
        }
        this->cells[i].cellLengthEff = sqrt(this->cells[i].area / M_PI);
        this->Area += this->cells[i].area;
    }
    for (int i = this->Ncell; i < this->Nghost; i++){
        this->cellXYGhost[i] = new mdouble[this->dim];
        int e = this->ghost2cell[i - this->Ncell]; 
        for (int j = 0; j < DIM; j++){
            this->cellXYGhost[i][j] = 2 * this->edges[e].edgeCenter[j] - this->cellXY[this->edges[e].faceMap[NN - 2]][j]; 
        }
    }
    double xl, xr, yl, yr, zl, zr, vn;
    int vl, vr;
    for (int i = 0; i < this->Nface; i++){
        vl = this->edges[i].faceMap[NN - 2];
        vr = this->edges[i].faceMap[NN - 1];
        xl = this->cellXYGhost[vl][0];
        xr = this->cellXYGhost[vr][0];
        yl = this->cellXYGhost[vl][1];
        yr = this->cellXYGhost[vr][1];
        zl = this->cellXYGhost[vl][2];
        zr = this->cellXYGhost[vr][2];
        const double dx = xl - xr;
        const double dy = yl - yr;
        const double dz = zl - zr;
        this->edges[i].edgeDist = sqrt(dx*dx + dy*dy + dz*dz);
        this->edges[i].edgerij[0] = (xr - xl) / this->edges[i].edgeDist;
        this->edges[i].edgerij[1] = (yr - yl) / this->edges[i].edgeDist;
        this->edges[i].edgerij[2] = (zr - zl) / this->edges[i].edgeDist;
        this->edges[i].edgerL[0] = this->edges[i].edgeCenter[0] - xl; 
        this->edges[i].edgerL[1] = this->edges[i].edgeCenter[1] - yl;
        this->edges[i].edgerL[2] = this->edges[i].edgeCenter[2] - zl;
        this->edges[i].edgerR[0] = this->edges[i].edgeCenter[0] - xr;
        this->edges[i].edgerR[1] = this->edges[i].edgeCenter[1] - yr;
        this->edges[i].edgerR[2] = this->edges[i].edgeCenter[2] - zr;
        vn = edges[i].edgeNormal[0] * edges[i].edgerij[0] + edges[i].edgeNormal[1] * edges[i].edgerij[1] + edges[i].edgeNormal[2] * edges[i].edgerij[2];
        if(vn < 0){
            edges[i].edgeNormal[0] = -edges[i].edgeNormal[0];
            edges[i].edgeNormal[1] = -edges[i].edgeNormal[1];
            edges[i].edgeNormal[2] = -edges[i].edgeNormal[2];
        }
    }
    bool tag = true;
    double src[DIM][DIM], **cellXYGhost = this->cellXYGhost, dx, dy, dz, len;
    int *cell2cell = NULL, v;
    int ncell = this->Ncell, dim = DIM;
    for (int i = 0; i < ncell; i++){
        src[0][0] = 0; src[0][1] = 0; src[0][2] = 0;
        src[1][0] = 0; src[1][1] = 0; src[1][2] = 0;
        src[2][0] = 0; src[2][1] = 0; src[2][2] = 0;
        cell2cell = this->cells[i].cell2cell;
        for (int j = 0; j < TAG[i]; j++){
            v = cell2cell[j];
            if(v == -1 || v >= Nghost){
                continue;
            }
            dx = cellXYGhost[v][0] - cellXYGhost[i][0];
            dy = cellXYGhost[v][1] - cellXYGhost[i][1];
            dz = cellXYGhost[v][2] - cellXYGhost[i][2];
            len = 1.0 / sqrt(dx * dx + dy * dy + dz * dz);
            this->cells[i].dxyz[j][0] = dx * len; 
            this->cells[i].dxyz[j][1] = dy * len;
            this->cells[i].dxyz[j][2] = dz * len;
            if (v >= ncell){
                continue;
            }
            src[0][0] += dx * dx * len;
            src[0][1] += dx * dy * len;
            src[0][2] += dx * dz * len;
            src[1][1] += dy * dy * len;
            src[1][2] += dy * dz * len;
            src[2][2] += dz * dz * len;
        } 
        src[1][0] = src[0][1];
        src[2][0] = src[0][2];
        src[2][1] = src[1][2];
        tag = this->calcMatrixInversion(src, dim, this->cells[i].Ainv); 
        if (!tag){
            this->cells[i].Ainv[0][0] = 0; this->cells[i].Ainv[0][1] = 0; this->cells[i].Ainv[0][2] = 0;
            this->cells[i].Ainv[1][0] = 0; this->cells[i].Ainv[1][1] = 0; this->cells[i].Ainv[1][2] = 0;
            this->cells[i].Ainv[2][0] = 0; this->cells[i].Ainv[2][1] = 0; this->cells[i].Ainv[2][2] = 0;
            printf("matrix is not invertible!\n");
            printf("cell id = %d, vol = %e, %e, %e, %e, %e, %e, %e, %e, %e, %e, TAG[i] = %d, %d, %d, %d, %d, %d\n", i, cells[i].area, src[0][0], src[0][1], src[0][2], src[1][0], src[1][1], src[1][2], src[2][0], src[2][1], src[2][2], TAG[i], cell2cell[0], cell2cell[1],cell2cell[2],cell2cell[3],cell2cell[4]);
        }
    }
    ensureReMeshIndexIdentity(true);
    for (int i = 0; i < this->Ncell; i++){
        cells[i].num = TAG[i];
        this->reMeshIndex[i] = i;
        this->reMeshIndex2[i] = i;
    }
    this->reMesh();
    return true;
}

/*
 * out2dat: writes solver fields or diagnostics.
 * Params: filePath, w, sq; returns: success or decision flag.
 * Flow:
 *   - select fields.
 *   - format by mesh order.
 *   - flush output.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
bool meshImport::out2dat(const char *filePath, const double w[][NCELL], const double sq[][VAR2])
{
    int tolFacePoint = 0;
    for(int i=0; i<this->Nface; i++){
        tolFacePoint += edges[i].faceType;
    }
    ofstream fout(filePath);
    fout << "VARIABLES = X, Y, Z, density, U, V, W, Ttra, Trot, P, qtx, qty, qtz, qrx, qry, ";
    fout << "qrz, sxx, sxy, sxz, syy, syz, szz" << endl;
    fout << "ZONE T = \"Finite Element data\" " << endl;
    fout << "ZONETYPE = FEPOLYHEDRON" << endl;
    fout << " Nodes = " << this->Npoint << " Elements = " << this->Ncell;
    fout << " Faces = " << this->Nface << " TotalNumFaceNodes = " << tolFacePoint << endl;
    fout << "NumConnectedBoundaryFaces = 0" << endl;
    fout << "TotalNumBoundaryConnections = 0" << endl;
    fout << "VarLocation=(NODAL,NODAL,NODAL,CellCentered,CellCentered,CellCentered";
    fout << ",CellCentered,CellCentered,CellCentered,CellCentered,CellCentered,CellCentered,CellCentered,CellCentered";
    fout << ",CellCentered,CellCentered,CellCentered,CellCentered,CellCentered,CellCentered,CellCentered,CellCentered)" << endl;
    for (int i = 0; i < this->Npoint; i++)
    {
        fout << this->pointXY[i][0] << " "; 
        if (i % 10 == 0)
        {
            fout << endl;
        }
    }
    fout << endl;
    for (int i = 0; i < this->Npoint; i++)
    {
        fout << this->pointXY[i][1] << " "; 
        if (i % 10 == 0)
        {
            fout << endl;
        }
    }
    fout << endl;
    for (int i = 0; i < this->Npoint; i++)
    {
        fout << this->pointXY[i][2] << " "; 
        if (i % 10 == 0)
        {
            fout << endl;
        }
    }
    fout << endl;
    for (int j = 0; j < VAR; j++)
    {
        for (int i = 0; i < this->Ncell; i++)
        {
            fout << w[j][i] << " ";
            if (i % 10 == 0)
            {
                fout << endl;
            }
        }
        fout << endl;
    }
    fout << endl;
    for (int i = 0; i < this->Ncell; i++)
    {
        fout << w[0][i] * w[4][i] << " ";
        if (i % 10 == 0)
        {
            fout << endl;
        }
    }
    fout << endl;
    for (int j = 0; j < VAR2; j++)
    {
        for (int i = 0; i < this->Ncell; i++)
        {
            fout << sq[i][j] << " "; 
            if (i % 10 == 0)
            {
                fout << endl;
            }
        }
        fout << endl;
    }
    fout << endl;
    for (int i = 0; i < this->Nface; i++)
    {
        fout << edges[i].faceType << " ";
        if (i % 10 == 0)
        {
            fout << endl;
        }
    }
    fout << endl;
    for (int i = 0; i < this->Nface; i++)
    {
        for(int j=0; j<edges[i].faceType; j++){
            fout << edges[i].faceMap[j] + 1 << " ";
        }
        fout << endl;
    }
    fout << endl;
    for (int i = 0; i < this->Nface; i++)
    {
        fout << edges[i].faceMap[4] + 1 << " ";
        if (i % 10 == 0)
        {
            fout << endl;
        }
    }
    fout << endl;
    for (int i = 0; i < this->Nface; i++)
    {
        if (edges[i].faceMap[5] >= this->Ncell)
        {
            fout << 0 << " ";
        }
        else
        {
            fout << edges[i].faceMap[5] + 1 << " ";
        }
        if (i % 10 == 0)
        {
            fout << endl;
        }
    }
    fout.close();
    return true;
}

/*
 * mutualMapping: performs one solver support operation.
 * Params: pl, pr, vis, len; returns: success or decision flag.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
bool meshImport::mutualMapping(int *pl, int *pr, vector<int> vis, int &len)
{
    edge *edges = this->edges;
    int f = -1, *fm, v, temp = 0;
    for (int i = 0; i < vis.size(); i++)
    {
        f = vis[i];
        fm = edges[f].faceMap;
        for (int j = 0; j < edges[f].faceType; j++)
        {
            v = fm[j];
            if (pr[v] != -1)
            {
                continue;
            }
            pl[temp] = v;
            pr[v] = temp;
            temp++;
        }
    }
    len = temp;
    return true;
}
/*
 * out2Fluent_ns2: writes solver fields or diagnostics.
 * Params: filePath, qf; returns: success or decision flag.
 * Flow:
 *   - select fields.
 *   - format by mesh order.
 *   - flush output.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
bool meshImport::out2Fluent_ns2(const char *filePath, double *qf)
{
    int cellId, cellFirstIndex, cellLastIndex;
    ofstream fout(filePath);
    int ncell = this->Ncell, *rmi = NULL;
    ensureReMeshIndexIdentity();
    rmi = this->reMeshIndex;
    double temperatureScale = this->mess.T_in / this->mess.T_ref;
    if (!(temperatureScale > 0.0) || !isfinite(temperatureScale)) temperatureScale = 1.0;
    double velocityScale =
        sqrt(this->mess.gamma) * this->mess.v_in / this->mess.v_rms;
    if (!(velocityScale > 0.0) || !isfinite(velocityScale)) velocityScale = 1.0;
    for (int i = 0; i < this->zonemap.size(); i++){
        if (this->zonemap[i].id == 2){
            cellId = this->zonemap[i].id;
            cellFirstIndex = this->zonemap[i].firstidx;
            cellLastIndex = this->zonemap[i].lastidx;
        }
    }
    fout << "(0 \" (300 (var-id zone-id var-size 0 0 first-id last-id)(......))\" )" << endl;
    fout << "(300 (700 " << cellId << " " << 1 << " 0 0 " << cellFirstIndex << " " << cellLastIndex << ")" << endl;
    fout << "(" << endl;
    for(int i = cellFirstIndex -1; i < cellLastIndex; i++)
    {
        fout << qf[rmi[i] * var + 0] << endl;
    }
    fout << "))" << endl;
    fout << "(300 (701 " << cellId << " " << 3 << " 0 0 " << cellFirstIndex << " " << cellLastIndex << ")" << endl;
    fout << "(" << endl;
    for(int i = cellFirstIndex -1; i < cellLastIndex; i++)
    {
        fout << qf[rmi[i] * var + 1] / velocityScale << " "
             << qf[rmi[i] * var + 2] / velocityScale << " "
             << qf[rmi[i] * var + 3] / velocityScale << endl;
    }
    fout << "))" << endl;
    fout << "(300 (702 " << cellId << " " << 1 << " 0 0 " << cellFirstIndex << " " << cellLastIndex << ")" << endl;
    fout << "(" << endl;
    for(int i = cellFirstIndex -1; i < cellLastIndex; i++)
    {
        fout << qf[rmi[i] * var + 4] / temperatureScale << endl;
    }
    fout << "))" << endl;    
    fout << "(300 (703 " << cellId << " " << 1 << " 0 0 " << cellFirstIndex << " " << cellLastIndex << ")" << endl;
    fout << "(" << endl;
    for(int i = cellFirstIndex -1; i < cellLastIndex; i++)
    {
        fout << qf[rmi[i] * var + 5] / temperatureScale << endl;
    }
    fout << "))" << endl;
    fout << "(300 (704 " << cellId << " " << 1 << " 0 0 " << cellFirstIndex << " " << cellLastIndex << ")" << endl;
    fout << "(" << endl;
    for(int i = cellFirstIndex -1; i < cellLastIndex; i++)
    {
        fout << cells[rmi[i]].no << endl;
    }
    fout << "))" << endl;
    fout.close();
    return true;
}
/*
 * out2Fluent_dsmc: writes solver fields or diagnostics.
 * Params: filePath, w; returns: success or decision flag.
 * Flow:
 *   - select fields.
 *   - format by mesh order.
 *   - flush output.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
bool meshImport::out2Fluent_dsmc(const char *filePath, double *w)
{
    int cellId, cellFirstIndex, cellLastIndex;
    ofstream fout(filePath);
    double temperatureScale = this->mess.T_in / this->mess.T_ref;
    if (!(temperatureScale > 0.0) || !isfinite(temperatureScale)) temperatureScale = 1.0;
    double velocityScale =
        sqrt(this->mess.gamma / 2.0) * this->mess.v_in / this->mess.v_rms;
    if (!(velocityScale > 0.0) || !isfinite(velocityScale)) velocityScale = 1.0;
    const double stressScale = temperatureScale;
    const double heatFluxScale = temperatureScale * velocityScale;
    for (int i = 0; i < this->zonemap.size(); i++){
        if (this->zonemap[i].id == 2){
            cellId = this->zonemap[i].id;
            cellFirstIndex = this->zonemap[i].firstidx;
            cellLastIndex = this->zonemap[i].lastidx;
        }
    }
    fout << "(0 \" (300 (var-id zone-id var-size 0 0 first-id last-id)(......))\" )" << endl;
    fout << "(300 (700 " << cellId << " " << 1 << " 0 0 " << cellFirstIndex << " " << cellLastIndex << ")" << endl;
    fout << "(" << endl;
    for (int iCell = cellFirstIndex - 1; iCell < cellLastIndex; iCell++)
    {
        fout << w[iCell*18+0] << endl;
    }
    fout << "))" << endl;
    fout << "(300 (701 " << cellId << " " << 3 << " 0 0 " << cellFirstIndex << " " << cellLastIndex << ")" << endl;
    fout << "(" << endl;
    for (int iCell = cellFirstIndex - 1; iCell < cellLastIndex; iCell++)
    {
        fout << w[iCell*18+1] / velocityScale << " "
             << w[iCell*18+2] / velocityScale << " "
             << w[iCell*18+3] / velocityScale << " " << endl;
    }
    fout << "))" << endl;
    fout << "(300 (702 " << cellId << " " << 2 << " 0 0 " << cellFirstIndex << " " << cellLastIndex << ")" << endl;
    fout << "(" << endl;
    for (int iCell = cellFirstIndex - 1; iCell < cellLastIndex; iCell++)
    {
        fout << w[iCell*18+4] / temperatureScale << " "
             << w[iCell*18+14] / temperatureScale << " " <<  endl;
    }
    fout << "))" << endl;
    fout << "(300 (710 " << cellId << " " << 6 << " 0 0 " << cellFirstIndex << " " << cellLastIndex << ")" << endl;
    fout << "(" << endl;
    for (int iCell = cellFirstIndex - 1; iCell < cellLastIndex; iCell++)
    {
        fout << w[iCell*18+5] / stressScale << " "
             << w[iCell*18+6] / stressScale << " "
             << w[iCell*18+7] / stressScale << " "
             << w[iCell*18+8] / stressScale << " "
             << w[iCell*18+9] / stressScale << " "
             << w[iCell*18+10] / stressScale << " "   << endl;
    }
    fout << "))" << endl;
    fout << "(300 (711 " << cellId << " " << 3 << " 0 0 " << cellFirstIndex << " " << cellLastIndex << ")" << endl;
    fout << "(" << endl;
    for (int iCell = cellFirstIndex - 1; iCell < cellLastIndex; iCell++)
    {
        fout << w[iCell*18+11] / heatFluxScale << " "
             << w[iCell*18+12] / heatFluxScale << " "
             << w[iCell*18+13] / heatFluxScale << endl;
    }
    fout << "))" << endl;
    fout << "(300 (712 " << cellId << " " << 3 << " 0 0 " << cellFirstIndex << " " << cellLastIndex << ")" << endl;
    fout << "(" << endl;
    for (int iCell = cellFirstIndex - 1; iCell < cellLastIndex; iCell++)
    {
        fout << w[iCell*18+15] / heatFluxScale << " "
             << w[iCell*18+16] / heatFluxScale << " "
             << w[iCell*18+17] / heatFluxScale << endl;
    }
    fout << "))" << endl;
    fout.close();
    return true;
}

/*
 * out2Fluent_ns6: writes solver fields or diagnostics.
 * Params: filePath, w; returns: success or decision flag.
 * Flow:
 *   - select fields.
 *   - format by mesh order.
 *   - flush output.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
bool meshImport::out2Fluent_ns6(const char *filePath, double *w)
{
    int cellId, cellFirstIndex, cellLastIndex;
    ofstream fout(filePath);
    double temperatureScale = this->mess.T_in / this->mess.T_ref;
    if (!(temperatureScale > 0.0) || !isfinite(temperatureScale)) temperatureScale = 1.0;
    double velocityScale =
        sqrt(this->mess.gamma) * this->mess.v_in / this->mess.v_rms;
    if (!(velocityScale > 0.0) || !isfinite(velocityScale)) velocityScale = 1.0;
    for (int i = 0; i < this->zonemap.size(); i++){
        if (this->zonemap[i].id == 2){
            cellId = this->zonemap[i].id;
            cellFirstIndex = this->zonemap[i].firstidx;
            cellLastIndex = this->zonemap[i].lastidx;
        }
    }
    fout << "(0 \" (300 (var-id zone-id var-size 0 0 first-id last-id)(......))\" )" << endl;
    fout << "(300 (700 " << cellId << " " << 1 << " 0 0 " << cellFirstIndex << " " << cellLastIndex << ")" << endl;
    fout << "(" << endl;
    for (int iCell = cellFirstIndex - 1; iCell < cellLastIndex; iCell++)
    {
        fout << w[iCell*var+0] << endl;
    }
    fout << "))" << endl;
    fout << "(300 (701 " << cellId << " " << 3 << " 0 0 " << cellFirstIndex << " " << cellLastIndex << ")" << endl;
    fout << "(" << endl;
    for (int iCell = cellFirstIndex - 1; iCell < cellLastIndex; iCell++)
    {
        fout << w[iCell*var+1] / velocityScale << " "
             << w[iCell*var+2] / velocityScale << " "
             << w[iCell*var+3] / velocityScale << " " << endl;
    }
    fout << "))" << endl;
    fout << "(300 (702 " << cellId << " " << 2 << " 0 0 " << cellFirstIndex << " " << cellLastIndex << ")" << endl;
    fout << "(" << endl;
    for (int iCell = cellFirstIndex - 1; iCell < cellLastIndex; iCell++)
    {
        fout << w[iCell*var+4] / temperatureScale << " "
             << w[iCell*var+5] / temperatureScale << " " <<  endl;
    }
    fout << "))" << endl;
    fout.close();
    return true;
}
/*
 * setMa_Kn_CFL: performs one solver support operation.
 * Params: ma, kn, cfl, cfl_psu, cfl_ns; returns: none.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
void meshImport::setMa_Kn_CFL(double ma, double kn, double cfl, double cfl_psu, double cfl_ns)
{
    this->Ma = ma;
    this->Kn = kn;
    this->cfl = cfl;
    this->cfl_psu = cfl_psu;
    this->cfl_ns = cfl_ns;
}

/*
 * setMeshMessage: performs one solver support operation.
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
void meshImport::setMeshMessage()
{
    dr = 2, dv = 1.626, zr = 3.5, zv = 2 * zr, omega = 0.74; 
    f_tra = 2.365, f_rot = 1.435;
    gamma = (5.0 + dr) / (3.0 + dr);
    delta_rp = pow(M_PI * 0.5, 0.5) / Kn, delta_sm = 1.0 / 1.33;
    omeag0 = 1.0 - 2.0 * zr * (2.5 / f_tra - 1.0);
    omeag1 = 1 - (1 / (f_rot * delta_sm) - 1) * (delta_sm / (1 - delta_sm)) * zr;
    cv = 1 / (gamma - 1.0), cp = cv * gamma;
    pr = (5.0 + dr) * 0.5 / (1.5 * f_tra + dr * 0.5 * f_rot);
    meshMessage &mess = this->mess;
    mess.Ncell = this->Ncell;
    mess.Nface = this->Nface;
    mess.Npoint = this->Npoint;
    mess.Nk = this->Nk;
    mess.bcnumber = this->bcnumber;
    mess.Nghost = this->Nghost;
    mess.Ma = this->Ma;
    mess.Kn = Kn;
    mess.cfl = this->cfl;
    mess.Area = this->Area;
    mess.gamma = this->gamma;
    mess.omega = this->omega;
    mess.dr = this->dr;
    mess.dv = this->dv;
    mess.zr = this->zr;
    mess.zv = this->zv;
    mess.var = this->var;
    mess.qn = this->qn;
    mess.cfl_ns = this->cfl_ns;
    mess.f_tra = this->f_tra;
    mess.f_rot = this->f_rot;
    mess.pr = pr;
    mess.delta_rp = delta_rp;
    mess.delta_sm = delta_sm;
    mess.omeag0 = omeag0;
    mess.omeag1 = omeag1;
    mess.cp = cp;
    mess.cv = cv;
    mess.eNface = this->bcnumber;
    this->iNface = this->Nface - this->bcnumber;
    mess.iNface = this->iNface;
    double kB = 1.3806e-23,T_ref = 273.15,p_mass = 4.65e-26,p_mass_r = 0.5*p_mass;
    double L0 = 3.912;
    double d_ref = 4.17e-10,alpha = 1.0, Twall_ref = 300.0 / T_ref,eta = omega - 0.5;
    double T_in = 142.2;
    double v_rms = sqrt(2*kB*T_ref/p_mass);
    double v_in = sqrt(2*kB*T_in/p_mass);
    const double d_ref2 = d_ref * d_ref;
    double miu_ref = 5.0 * (alpha+1.0) * (alpha+2.0) * sqrt(p_mass*kB*T_ref/M_PI)
    /4.0/alpha/ (5.0-2.0*omega)/(7.0-2.0*omega)/d_ref2;
    miu_ref *= pow(T_in/T_ref,omega);
    double p_ref = miu_ref / (mess.Kn * L0) * sqrt(M_PI * kB / p_mass * T_in / 2.0);
    double n_ref = p_ref / (kB * T_in);
    double tau = miu_ref*pow((T_ref/T_ref),omega)/(n_ref*kB*T_ref);
    double tauc = tau * alpha*(5.0-2.0*omega)*(7.0-2.0*omega)/(5.0*(alpha+1.0)*(alpha+2.0));
    double dt_ref = (1/Kn)*sqrt(M_PI)/2*miu_ref/p_ref*sqrt(T_in/T_ref);
    double Z_sparta = tau/tauc * zr;
    double P_relax = 1/Z_sparta;
    P_relax = 0.18;
    mess.kB = kB;
    mess.T_ref = T_ref;
    mess.T_in = T_in;
    mess.v_in = v_in;
    mess.n_ref = n_ref;
    mess.p_ref = p_ref;
    mess.d_ref = d_ref;
    mess.p_mass = p_mass;
    mess.p_mass_r = p_mass_r;
    mess.miu_ref = miu_ref;
    mess.v_rms = v_rms;
    mess.eta = eta;
    mess.Twall_ref = Twall_ref;
    mess.alpha = alpha;
    mess.P_relax = P_relax;
    mess.dt_ref = dt_ref;
    setParticleMessage();
    cout<<"the initial velocity is: "<< mess.Ma*sqrt(2/gamma)<<endl;
}

/*
 * setParticleMessage: updates particles or particle-derived state.
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
void meshImport::setParticleMessage()
{
    meshMessage &mess = this->mess;
    int p1, p2; 
    double mintag = DBL_MAX;
    for(int iFace = 0; iFace < this->Nface ; iFace++)  
    {
        for (int iTriangle = 0; iTriangle < edges[iFace].faceType; iTriangle++)
        {
            p1 = edges[iFace].faceMap[iTriangle];
            if (iTriangle < edges[iFace].faceType - 1)
            {
                p2 = edges[iFace].faceMap[iTriangle + 1];
            }
            else
            {
                p2 = edges[iFace].faceMap[0];
            }
            double x1, x2, y1, y2, z1, z2, v1;
            x1 = pointXY[p1][0];
            y1 = pointXY[p1][1];
            z1 = pointXY[p1][2];
            x2 = pointXY[p2][0];
            y2 = pointXY[p2][1];
            z2 = pointXY[p2][2];
            v1 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
            if(v1 < mintag)
            {
                mintag = v1;
            }
        }
    }
    this->miniLength = mintag;
    this->Npc_initial = Npinitial;
    const int realCellCount = this->Ncell;
    this->Nps = Npc_initial * realCellCount; 
    this->Neff = mess.Area * mess.n_ref / Nps;
    this->dtime = miniLength*0.2;
    if (this->Npc_exa != NULL)
    {
        delete[] this->Npc_exa;
        this->Npc_exa = NULL;
    }
    this->Npc_exa = new int [realCellCount];
    for(int i = 0; i < realCellCount;i++)
    {
        int meshIndex = (this->reMeshIndex != NULL) ? this->reMeshIndex[i] : i;
        if (meshIndex < 0 || meshIndex >= realCellCount)
        {
            meshIndex = i;
        }
        double S = this->cells[meshIndex].area;
        double Ns = S*Nps/this->Area;
        double r = static_cast<double>(rand())/ RAND_MAX;
        if (r<(Ns-floor(Ns))){this->Npc_exa[i] = floor(Ns) + 1;}
        else{this->Npc_exa[i] = floor(Ns);} 
    }
    mess.dtime = this->dtime;
    mess.Neff = this->Neff;
    this->calGridpar();
}

/*
 * calGridpar: performs one solver support operation.
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
void meshImport::calGridpar()
{
    if (this->c_size <= 0 || this->cells == NULL || this->Ncell <= 0)
    {
        cout << "CALGRIDPAR_ERROR invalid c_size/cells/Ncell: "
             << " c_size=" << this->c_size
             << " Ncell=" << this->Ncell << endl;
        return;
    }
    if (this->startGrid != NULL)
    {
        delete[] this->startGrid;
        delete[] this->endGrid;
    }
    this->startGrid = new int[this->c_size];
    this->endGrid = new int[this->c_size];
    vector<int> counts(this->c_size, 0);
    bool sorted = true;
    int prev = -1;
    for (int i = 0; i < this->Ncell; ++i)
    {
        int owner = this->cells[i].no;
        if (owner < 0 || owner >= this->c_size)
        {
            cout << "CALGRIDPAR_ERROR invalid cell owner: cell=" << i
                 << " owner=" << owner
                 << " c_size=" << this->c_size << endl;
            return;
        }
        if (owner < prev)
        {
            sorted = false;
        }
        prev = owner;
        counts[owner]++;
    }
    if (!sorted)
    {
        cout << "CALGRIDPAR_ERROR cells are not sorted by cells[i].no. "
             << "startGrid/endGrid cannot represent contiguous ranges." << endl;
        return;
    }
    int offset = 0;
    for (int j = 0; j < this->c_size; ++j)
    {
        this->startGrid[j] = offset;
        offset += counts[j];
        this->endGrid[j] = offset;
    }
    if (offset != this->Ncell)
    {
        cout << "CALGRIDPAR_ERROR counted cells != Ncell: "
             << offset << " " << this->Ncell << endl;
    }
}

/*
 * calcMatrixInversion: works with mesh topology or geometric intersections.
 * Params: src, n, des; returns: success or decision flag.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
bool meshImport::calcMatrixInversion(double src[DIM][DIM], int n, double des[DIM][DIM])
{
    double flag = this->calcDeterminant(src, n);
    if (fabs(flag) < 1e-8)
    {
        return false;
    }
    else
    {
        this->calcCofactor(src, n, des);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                des[i][j] /= flag;
            }
        }
    }
    return true;
}

/*
 * calcDeterminant: prepares derived solver state.
 * Params: arcs, n; returns: computed scalar.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
double meshImport::calcDeterminant(double arcs[DIM][DIM], int n)
{
    if (n == 1)
    {
        return arcs[0][0];
    }
    double ans = 0;
    double temp[DIM][DIM] = {0.0};
    int i, j, k;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n - 1; j++)
        {
            for (k = 0; k < n - 1; k++)
            {
                temp[j][k] = arcs[j + 1][(k >= i) ? k + 1 : k];
            }
        }
        double t = this->calcDeterminant(temp, n - 1);
        if (i % 2 == 0)
        {
            ans += arcs[0][i] * t;
        }
        else
        {
            ans -= arcs[0][i] * t;
        }
    }
    return ans;
}

/*
 * calcCofactor: prepares derived solver state.
 * Params: arcs, n, ans; returns: none.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
void meshImport::calcCofactor(double arcs[DIM][DIM], int n, double ans[DIM][DIM])
{
    if (n == 1)
    {
        ans[0][0] = 1;
        return;
    }
    int i, j, k, t;
    double temp[DIM][DIM];
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            for (k = 0; k < n - 1; k++)
            {
                for (t = 0; t < n - 1; t++)
                {
                    temp[k][t] = arcs[k >= i ? k + 1 : k][t >= j ? t + 1 : t];
                }
            }
            ans[j][i] = this->calcDeterminant(temp, n - 1);
            if ((i + j) % 2 == 1)
            {
                ans[j][i] = -ans[j][i];
            }
        }
    }
}

/*
 * changeFluent: performs one solver support operation.
 * Params: filePath, tag; returns: index, count, owner, or status value.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
int meshImport::changeFluent(const char *filePath, bool tag)
{
    if(!tag){
        return 0;
    }
    int inface = 0;
    ifstream infile;
    string s, nl, nr;
    infile.open(filePath, ios::in);
    if (!infile.is_open())
    {
        printf("open failed !");
        return false;
    }
    stringstream sst;
    int type, l ,r;
    while (!infile.eof())
    while (getline(infile, s))
    {
        sst.clear();
        sst.str(s);
        sst >> s;
        if(s.compare("(13")){
            continue;
        }
        sst >> s;
        if(!s.compare("(0")){
            continue;
        }
        sst >> nl >> nr >> s;
        type = hex2dcm(s);
        if(type != 2){
            continue;
        }
        l = hex2dcm(nl);
        r = hex2dcm(nr);
        inface = r - l + 1;
        this->edge1stLocation = l;
        break;
    }
    infile.close();
    return inface;
}
/*
 * metisPartition: updates partition ownership or load data.
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
void meshImport::metisPartition(idx_t nParts)
{
    if(nParts <= 1){
        return;
    }
    int ncell = this->Ncell;
    int nEdges = 0;
    for(int i=0; i<ncell; i++){
        for(int j=0; j<TAG[i]; j++){
            if(cells[i].cell2cell[j] < 0 || cells[i].cell2cell[j] >= ncell){
                continue;
            }
            nEdges ++;
        }
    }
    idx_t *xadj = new idx_t[ncell + 1];
    idx_t *adjncy = new idx_t[nEdges * 2];
    idx_t *part = new idx_t[ncell];
    int nCells = 0; xadj[0] = 0;
    for(int i=0; i<ncell; i++){
        for(int j=0; j<TAG[i]; j++){
            if(cells[i].cell2cell[j] < 0 || cells[i].cell2cell[j] >= ncell){
                continue;
            }
            adjncy[nCells] = cells[i].cell2cell[j];
            nCells ++;
        }
        xadj[i + 1] = nCells;
    }
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
    options[METIS_OPTION_CTYPE] = METIS_CTYPE_SHEM;
    idx_t ncon = 1, objval;
    int ret = -1;
    if(nParts > 8){
        ret =  METIS_PartGraphKway (&ncell, &ncon, xadj, adjncy, NULL, NULL, NULL, &nParts, NULL, NULL, options, &objval, part);
    }else{
        ret =  METIS_PartGraphRecursive (&ncell, &ncon, xadj, adjncy, NULL, NULL, NULL, &nParts, NULL, NULL, options, &objval, part);
    }
    if (ret != METIS_OK) {
        cout << "MESH_PARTITION_METIS_FAIL ret=" << ret << endl;
        delete[] xadj;
        delete[] adjncy;
        delete[] part;
        return;
    }
    for(int i=0; i<ncell; i++){
        cells[i].no = part[i];
    }
    delete[] xadj; delete[] adjncy; delete[] part;
}

/*
 * cmp2: performs one solver support operation.
 * Params: c1, c2; returns: success or decision flag.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
bool cmp2(const cellNode &c1, const cellNode &c2){
    return c1.no == c2.no ? c1.id < c2.id : c1.no < c2.no;
}

/*
 * reMesh: performs one solver support operation.
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
void meshImport::reMesh()
{
    this->originalCells = vector<cell>(this->cells, this->cells + this->Ncell);
    this->originalEdges = vector<edge>(this->edges, this->edges + this->Nface);
    int ncell = this->Ncell, nface = this->Nface;
    ensureReMeshIndexIdentity(true);
    int *rmi = this->reMeshIndex;
    cellNode *nodes = new cellNode[ncell];
    for(int i=0; i<ncell; i++){
        nodes[i].no = cells[i].no;
        nodes[i].id = i;
    }
    vector<cellNode> vis(nodes, nodes + ncell); 
    sort(vis.begin(), vis.end(), cmp2);
    vector<cell> V;
    for(int i=0; i<ncell; i++){
        V.push_back(cells[i]);
    }
    for(int i=0; i<ncell; i++){
        cells[i] = V[vis[i].id];
        this->reMeshIndex[vis[i].id] = i;
        this->reMeshIndex2[i] = vis[i].id;
    }
    int *curr;
    for(int i=0; i<ncell; i++){
        curr = cells[i].cell2cell;
        for(int j=0; j<cells[i].num; j++){
            if(curr[j] >= ncell || curr[j] < 0){
                continue;
            }
            if(cells[i].cell2face[j] == -1 && cells[i].no != cells[rmi[curr[j]]].no){
                printf("0-------cells[i].no != cells[curr[j]].no ! curr[j] = %d, no1 = %d, no2 = %d\n", curr[j], cells[i].no, cells[curr[j]].no);
            }
            cells[i].cell2cell[j] = rmi[curr[j]];   
            if(cells[i].cell2face[j] == -1 && cells[i].no != cells[curr[j]].no){
                printf("1-----cells[i].no != cells[curr[j]].no ! curr[j] = %d, no1 = %d, no2 = %d\n", curr[j], cells[i].no, cells[curr[j]].no);
            }
        }
    }
    int l, r;
    for(int i=0; i<nface; i++){
        l = edges[i].faceMap[NN - 2];
        edges[i].faceMap[NN - 2] = rmi[l];
        r = edges[i].faceMap[NN - 1];
        if(r >= ncell){
            continue;
        }
        edges[i].faceMap[NN - 1] = rmi[r];
    }
    V.clear();
    vis.clear();
    delete[] nodes;
}
