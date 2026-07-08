#include "MacroSolver.h"
#include <cmath>

MacroSolver::MacroSolver()
{
}

MacroSolver::MacroSolver(meshImport *mesh, meshMessage mess, MessagePassing *mpass, const MpiContext& mpiCtx)
{
    this->mpi = &mpiCtx;
    this->mesh = mesh;
    this->mess = mess;
    this->mpass = mpass;
    this->var = mess.var;
    this->myGroup = mpiCtx.calGroup;
    this->comm = mpiCtx.comm;
    this->iRank = mpiCtx.c_rank;
    this->iSize = mpiCtx.c_size;
    this->rank = mpiCtx.rank;
    this->size = mpiCtx.size;

    if (active())
    {
        if(this->nParts == NULL)
        {
            this->nParts = new int[this->iSize + 1]; 
            this->sendCount = new int[this->iSize + 1];
            this->recCount = new int[this->iSize + 1];
        }

    }

    this->dyparcells();
    this->cell_cell_initial();

    if (active())
    {
        this->mallocSpace();
        for(int i = 0; i < this->iNcell; i++){
            this->impVis.push_back(i);
        }
        for(int i = this->iNcell - 1; i >= 0; i--){
            this->impVis.push_back(i);
        }
        this->initialParts();
        this->initialParallel();
    }
}


MacroSolver::~MacroSolver()
{
    if (active())
    {
        int ncell = this->Ncell, nface = this->Nface, var = this->var;
        int incell = this->iNcell, inface = this->iNface, enface = this->eNface;
        if(RHS != NULL)
        {
            for(int i = 0; i < ncell; i++){
                delete[] this->Qf[i];
                delete[] this->Qc[i];
                delete[] this->impf[i];
                delete[] this->ns_sigmaq[i];
                delete[] this->d_sigmaq[i];
            }
            for(int i = 0; i < incell; i++){
                delete[] this->RHS[i];
                delete[] this->Grad[i];
                delete[] this->Umaxmin[i];
            }
            for (int i = 0; i < nface; i++){
                delete[] this->qfl[i];
                delete[] this->qfr[i];
            }
            delete[] this->delta_t;
            delete[] this->Qf;
            delete[] this->Qc;
            delete[] this->impf;
            delete[] this->ns_sigmaq;
            delete[] this->d_sigmaq;
            delete[] this->RHS;
            delete[] this->Grad;
            delete[] this->Umaxmin;
            delete[] this->qfl;
            delete[] this->qfr;
        }

        if(this->edges != NULL){
            delete[] this->edges;
            this->edges = NULL;
        }
        if(this->cells != NULL){
            delete[] this->cells;
            this->cells = NULL;
        }
        if(this->fluxTotal != NULL){
            delete[] this->fluxTotal;
        }
        if(this->nParts!= NULL){
            delete[] this->nParts;
            delete[] this->sendCount;
            delete[] this->recCount;
            delete[] this->sendBuf;
            delete[] this->recBuf;
        }
        if(this->status != NULL){
            this->freeParallel();
            delete[] this->status;
            delete[] this->request;
        }
        if(this->tVector != NULL){
            delete[] this->tVector;
        }
    }
}

void MacroSolver::scatter_counts(vector<int> counts_or_null,int& my, MPI_Comm comm) {

    if (root()) {
        MPI_Scatter(counts_or_null.data(), 1, MPI_INT, &my, 1, MPI_INT, 0, comm);
    } else {
        MPI_Scatter(nullptr, 0, MPI_INT, &my, 1, MPI_INT, 0, comm);
    }
}

void MacroSolver::dyparcells()
{
    if(this->NsreMeIndex2 == NULL)
    {
        this->NsreMeIndex2 = new int [this->mess.Ncell];
        this->NsreMeIndex = new int [this->mess.Ncell];
    }

    if(root())
    {
        for(int i = 0; i < this->mess.Ncell;i ++)
        {
            this->NsreMeIndex2[i] = this->mesh->reMeshIndex2[i];
            this->NsreMeIndex[i] = this->mesh->reMeshIndex[i];
        }
    }

    MPI_Bcast(this->NsreMeIndex2, this->mess.Ncell, MPI_INT, 0, comm);
    MPI_Bcast(this->NsreMeIndex, this->mess.Ncell, MPI_INT, 0, comm);

    vector<int> sendstbuf(this->size,0);
    vector<int> sendedbuf(this->size,0);

    if (root()) 
    {
        for (int r = 1; r < this->size; ++r) 
        {
            sendstbuf[r] = mesh->startGrid[r-1];
            sendedbuf[r] = mesh->endGrid[r-1];
        }
    }
    
    scatter_counts(sendstbuf,this->Nl, this->comm);
    scatter_counts(sendedbuf,this->Nr, this->comm);

    if(active())
    {   
        this->iNcell = this->Nr - this->Nl;
    }   
}

void MacroSolver::cell_cell_initial()
{
    MPI_Datatype mycellMessage;
    vector<cell> cells_sendbuf;
    vector<cell> cells_recvbuf;
    mpass->commitMyCell(mycellMessage);

    int recvnumber = 0;

    vector<int> cells_sendcount(this->size,0);

    vector<int> displs(this->size,0);
    if(root())
    {
        for(int i = 1; i < this->size; i++ )
        {
            cells_sendcount[i] = mesh->endGrid[i-1] - mesh->startGrid[i-1];
        }

        for (int r = 1; r < this->size; ++r) {displs[r] = displs[r-1] + cells_sendcount[r-1];}

        for(int j = 0; j < this->iSize; j++)   
        {
            for (int i = 0; i < this->mess.Ncell;i++)
            {  
                if (mesh->cells[i].no < j)
                {
                    continue;                    
                }
                if(mesh->cells[i].no > j)
                {
                    break;                       
                }
                cells_sendbuf.push_back(mesh->cells[i]);
            }
        }

        if(cells_sendbuf.size() != this->mess.Ncell)
        {cout<<"the error size"<< cells_sendbuf.size()<<endl;} 
    }

    if(active())
    {
        recvnumber = this->iNcell;
        cells_recvbuf.resize(recvnumber);
    }
    
    if (root()) 
    {
        MPI_Scatterv(cells_sendbuf.data(), cells_sendcount.data(), displs.data(), mycellMessage,
                     MPI_IN_PLACE, 0, mycellMessage, 0, comm);
    } 
    else 
    {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
            cells_recvbuf.data(), recvnumber, mycellMessage, 0, comm);
    }

    if(active())
    {
        if(this->cells != NULL)
        {
            delete[] this->cells;
        }
        this->cells = new cell[this->iNcell];

        for(int i  = 0; i < this->iNcell ;i++){
            this->cellDeepCopy(this->cells[i], cells_recvbuf[i]);
            
        }
    }

    edge_cell_initial();

    MPI_Type_free(&mycellMessage);
    MPI_Barrier(comm);
}

bool cmp4(const fNode &c1, const fNode &c2){
    return c1.no == c2.no ? c1.fid < c2.fid : c1.no < c2.no;
}

void MacroSolver::edge_cell_initial()
{

    MPI_Datatype myedgeMessage;
    MPI_Datatype myvisMessage;
    mpass->commitMyEdge(myedgeMessage);
    mpass->commitMyvis(myvisMessage);

    vector<edge> edges_sendbuf;vector<fNode> vis_sendbuf;vector<fNode> cvis_sendbuf;
    vector<edge> edges_recvbuf;vector<fNode> vis_recvbuf;vector<fNode> cvis_recvbuf;
    vector<int> edges_sendcount(this->size,0);
    vector<int> cvis_sendcount(this->size,0);
    
    vector<int> displs(this->size,0);
    vector<int> displ_vis(this->size,0);

    vector<int> bface_size(size,0);
    vector<int> iface_size(size,0);
    vector<int> eface_size(size,0);
    vector<int> ecell_size(size,0);

    int ecell = 0, bface = 0, iface = 0, eface = 0;
    
    if(root())
    {
        vector<unordered_map<int,int>> seen_gid(this->iSize);
        vector<vector<int>> ordered_gid(this->iSize);
        unordered_map<int, bool> isVis, fisVis, esVis;
        vector<vector<fNode>> vis, cvis;
        vis.resize(this->iSize);
        cvis.resize(this->iSize);

        for(int r = 0;r < this->iSize;r ++)
        {
            seen_gid[r].reserve(this->mess.Nface*4/this->iSize);
            fisVis.clear();
            isVis.clear();
            esVis.clear();
            bface = 0, iface = 0, eface = 0,ecell = 0;

            for(int i = mesh->startGrid[r], no; i < mesh->endGrid[r]; i++) 
            {
                for(int j=0, ci, fi; j<mesh->cells[i].num; j++){
                    ci = mesh->cells[i].cell2cell[j];
                    fi = mesh->cells[i].cell2face[j];
                    if(fi == -1){
                        break;
                    }
                    if(isVis[fi]){
                        continue;
                    }
                    isVis[fi] = true;

                    fNode fn;
                    fn.fid = fi;

                    if(ci >= this->mess.Ncell){              
                        fn.no = this->iSize;
                        vis[r].push_back(fn);
                        bface ++;
                        continue;
                    }

                    if(mesh->cells[ci].no != r){            
                        no = mesh->cells[ci].no;
                    }else{
                        no = -1;
                    }
                    fn.no = no;
                    vis[r].push_back(fn);

                    if(mesh->cells[ci].no != r){          
                        eface ++; 
                        continue;
                    }
                    iface ++;

                }
            }
            bface_size[r+1] = bface;
            iface_size[r+1] = iface;
            eface_size[r+1] = eface;
        
            for(int i=mesh->startGrid[r], no; i<mesh->endGrid[r]; i++) 
            {
                for(int j=0, ci, fi; j<mesh->cells[i].num; j++){
                    ci = mesh->cells[i].cell2cell[j];
                    fi = mesh->cells[i].cell2face[j];
                    if(fi == -1){
                        break;
                    }

                    if (ci < this->mess.Ncell) 
                    { 
                        if (mesh->cells[ci].no != r && !esVis[ci]) 
                        {
                            ++ecell;
                            esVis[ci] = true;
                        }
                    }

                    if(fisVis[ci]){
                        continue;
                    }
                    fisVis[ci] = true;
                    fNode cn;
                    cn.fid = ci;

                    if(ci >= this->mess.Ncell){
                        cn.no = this->iSize;
                        cvis[r].push_back(cn);
                        continue;
                    }

                    if(mesh->cells[ci].no != r){
                        no = mesh->cells[ci].no;
                    }else{
                        no = -1;
                    }
                    cn.no = no;
                    cvis[r].push_back(cn);
                }
            }

            ecell_size[r+1] = ecell;

            for(int i=mesh->startGrid[r],node_number; i<mesh->endGrid[r]; i++) 
            {
                for(int j=0, ci, fi; j<mesh->cells[i].num; j++)
                {
                    ci = mesh->cells[i].cell2cell[j];
                    fi = mesh->cells[i].cell2face[j];
                    if(fi == -1){
                        break;
                    }
                    node_number = mesh->edges[fi].faceType;
                    for (int k = 0; k < node_number; ++k) 
                        {
                            int gid = mesh->edges[fi].faceMap[k];
                            if (gid < 0) {cout<<"node error"<<endl; continue;}
                            if (seen_gid[r].find(gid) == seen_gid[r].end())
                            {
                                int local_idx = ordered_gid[r].size();
                                seen_gid[r][gid] = local_idx;   
                                ordered_gid[r].push_back(gid);
                            }
                        }   
                }
            }
            
            sort(vis[r].begin(), vis[r].end(), cmp4); 
            sort(cvis[r].begin(), cvis[r].end(), cmp4); 

            for(int i = 0; i < vis[r].size();i++)
            {
                edges_sendbuf.push_back(mesh->edges[vis[r][i].fid]);
                vis_sendbuf.push_back(vis[r][i]);
            }

            for(int i = 0; i < cvis[r].size();i++)
            {
                cvis_sendbuf.push_back(cvis[r][i]);
            }
            cvis_sendcount[r+1] = cvis[r].size();
            edges_sendcount[r+1] = vis[r].size();
        }

    }

    int recvnumber;
    scatter_counts(ecell_size,ecell, comm);
    scatter_counts(bface_size,bface, comm);
    scatter_counts(iface_size,iface, comm);
    scatter_counts(eface_size,eface, comm);
    int visnumber;
    scatter_counts(cvis_sendcount,visnumber, comm);

    if(active())
    {
        this->Ncell = this->iNcell + ecell;
        this->iNface = iface;
        this->eNface = this->iNface + eface;
        this->Nface = bface + iface + eface;
        recvnumber = this->Nface;
        edges_recvbuf.resize(recvnumber);
        vis_recvbuf.resize(recvnumber);
        cvis_recvbuf.resize(visnumber);
    }

    if(root())
    {   
        for (int r = 1; r < this->size; ++r) {
            displ_vis[r] = displ_vis[r-1] + cvis_sendcount[r-1];
            displs[r] = displs[r-1] + edges_sendcount[r-1];
        }
    }

    if (root()) 
    {
        MPI_Scatterv(edges_sendbuf.data(), edges_sendcount.data(), displs.data(), myedgeMessage,
                     MPI_IN_PLACE, 0, myedgeMessage, 0, comm);
        MPI_Scatterv(vis_sendbuf.data(), edges_sendcount.data(), displs.data(), myvisMessage,
                     MPI_IN_PLACE, 0, myvisMessage, 0, comm);
        MPI_Scatterv(cvis_sendbuf.data(), cvis_sendcount.data(), displ_vis.data(), myvisMessage,
                     MPI_IN_PLACE, 0, myvisMessage, 0, comm);
    } else 
    {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
            edges_recvbuf.data(), recvnumber, myedgeMessage, 0, comm);
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
            vis_recvbuf.data(), recvnumber, myvisMessage, 0, comm);
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
            cvis_recvbuf.data(), visnumber, myvisMessage, 0, comm);
    }

    if(active())
    {
        for(int i=0; i<this->iSize+1; i++){
            this->nParts[i] = 0;
        }
    
        for(int i = this->iNface; i< this->eNface; i++){
            this->nParts[vis_recvbuf[i].no + 1] ++;
        }
    
        this->nParts[0] = this->iNface;
    
        for(int i=1; i<this->iSize+1; i++){
            this->nParts[i] += this->nParts[i-1];
        }
        
        if(this->edges != NULL)
        {
            delete[] this->edges;
        }
    
        this->edges = new edge[this->Nface];
    
        unordered_map<int, int> vvv, cvvv;
        for(int i=0; i<this->Nface; i++){
            this->edgeDeepCopy(this->edges[i], edges_recvbuf[i]);
            
            vvv[vis_recvbuf[i].fid] = i;

            if(i >= this->eNface){
                this->bcVis.push_back(vis_recvbuf[i].fid - (mess.Nface - mess.eNface));
            }
        }

        for(int i=0; i<cvis_recvbuf.size(); i++){
            cvvv[cvis_recvbuf[i].fid] = i; 
        }

        for(int i=0; i<this->iNcell; i++)
        {
            for(int j=0, ci, fi; j<this->cells[i].num; j++)
            {
                ci = this->cells[i].cell2cell[j];
                fi = this->cells[i].cell2face[j];
    
                if(fi == -1 && ci < 0){ 
                    printf("Error: fi == -1 && ci < 0\n");
                    break;
                }
                if((fi == -1 && this->cells[i].no != this->cells[ci].no)){
                    printf("Error: ci < 0 || (fi == -1 && Cells[i].no != cells[ci].no), no = %d, no1 = %d, no2 = %d, ci = %d, cvvv[ci] = %d\n", cells[i+Nl].no, this->cells[i].no, this->cells[ci].no, ci, cvvv[ci]);
                    break;
                }
    
                if(fi != -1)
                {
                    this->cells[i].cell2face[j] = vvv[fi];  
                }
    
                this->cells[i].cell2cell[j] = cvvv[ci]; 
            }
        }
        
        for(int i=0, l, r; i<this->Nface; i++)
        {  
            l = this->edges[i].faceMap[NN - 2];  
            r = this->edges[i].faceMap[NN - 1];
            this->edges[i].faceMap[NN - 2] = cvvv[l];
            this->edges[i].faceMap[NN - 1] = cvvv[r];
        }
    }

    MPI_Type_free(&myedgeMessage);
    MPI_Type_free(&myvisMessage);
    MPI_Barrier(MPI_COMM_WORLD);
}


bool MacroSolver::mallocSpace()
{
    int ncell = this->Ncell, nface = this->Nface, var = this->var;
    int incell = this->iNcell, inface = this->iNface, enface = this->eNface;

        this->Qf = new double*[ncell];
        this->Qc = new double*[ncell];
        this->impf = new double*[ncell];
        this->ns_sigmaq = new double*[ncell];
        this->d_sigmaq = new double*[ncell];

        this->RHS = new double*[incell];
        this->delta_t = new double[incell];
        this->Grad = new double*[incell];
        this->Umaxmin = new double*[incell];
        this->qfl = new double*[nface];
        this->qfr = new double*[nface];

        for(int i = 0; i < ncell; i++){
            this->Qf[i] = new double[var];
            this->Qc[i] = new double[var];
            this->impf[i] = new double[var];
            this->ns_sigmaq[i] = new double[VAR2];
            this->d_sigmaq[i] = new double[VAR2];
        }
        for(int i = 0; i < incell; i++){
            this->RHS[i] = new double[var];
            this->Grad[i] = new double[var * DIM];
            this->Umaxmin[i] = new double[var * 2];
        }
        for (int i = 0; i < nface; i++){
            this->qfl[i] = new double[var];
            this->qfr[i] = new double[var];
        }

    return true;
}

void MacroSolver::bcClassification()
{
    int nface = this->Nface, enface = this->eNface;

    this->farVis.clear();
    this->OutWall.clear();
    this->wallVis.clear();

    for (int i = enface; i < nface; i++){
        const BoundaryCondition& bc = this->boundaryTable.byTag(edges[i].faceTag);
        switch (bc.nsRole){
            case BoundaryRole::Inlet:
                this->farVis.push_back(i);
                break;
            case BoundaryRole::Outlet:
                this->OutWall.push_back(i);
                break;
            case BoundaryRole::Wall:
                this->wallVis.push_back(i);
                break;
            default:
                break;
        }
    }
}

bool MacroSolver::applyNSEG13BoundaryFluxes()
{
    vector<int> wallFaces;
    vector<int> freestreamInletFaces;
    vector<int> openOutletFaces;

    wallFaces.reserve(this->wallVis.size());
    freestreamInletFaces.reserve(this->farVis.size());
    openOutletFaces.reserve(this->OutWall.size());

    auto collectByModel = [this](const vector<int>& source, BoundaryRole role,
                                 BoundaryModel model, vector<int>& target)
    {
        for (int faceIdx : source)
        {
            if (faceIdx < 0 || faceIdx >= this->Nface) continue;
            const BoundaryCondition& bc = this->boundaryTable.byTag(this->edges[faceIdx].faceTag);
            if (bc.nsRole == role && bc.nsModel == model)
                target.push_back(faceIdx);
        }
    };

    for (int faceIdx : this->wallVis)
    {
        if (faceIdx < 0 || faceIdx >= this->Nface) continue;
        const BoundaryCondition& bc = this->boundaryTable.byTag(this->edges[faceIdx].faceTag);
        if (bc.nsRole == BoundaryRole::Wall)
            wallFaces.push_back(faceIdx);
    }
    collectByModel(this->farVis, BoundaryRole::Inlet, BoundaryModel::FreestreamInlet, freestreamInletFaces);
    collectByModel(this->OutWall, BoundaryRole::Outlet, BoundaryModel::OpenOutlet, openOutletFaces);

    this->Flux_NSEG13_bcWallWithT(this->mess.Twall_ref, wallFaces);
    this->Flux_NSEG13_inlet(freestreamInletFaces);
    this->Flux_NSEG13_outlet(openOutletFaces);
    return true;
}

bool MacroSolver::initialW()
{
    int incell = this->iNcell, ncell = this->Ncell, var = this->var;
    double Ma = mess.Ma*sqrt(mess.gamma);
    double rho, ux, uy, uz, t, x, y, z, theta = 30;

    double *w = new double[mess.Ncell * var];
    if(activeLeader()){
        int *rmi = this->NsreMeIndex2;
        for(int i=0; i<mess.Ncell; i++){
            rho = 1; ux = Ma *this->mess.v_in/this->mess.v_rms  * sqrt(3) / 2; uy = Ma *this->mess.v_in/this->mess.v_rms / 2; uz = 0; t = this->mess.T_in/this->mess.T_ref;
            
            
            
            
            
            

            
            
            
            
            
            
            
            

            w[i * var + 0] = rho;
            w[i * var + 1] = ux; 
            w[i * var + 2] = uy; 
            w[i * var + 3] = uz;
            w[i * var + 4] = t; 
            w[i * var + 5] = t; 
        }
    }
    MPI_Bcast(w, mess.Ncell * var, MPI_DOUBLE, 0, this->myGroup);
    for(int i=Nl; i<Nr; i++){
        for(int j=0; j<var; j++){
            this->Qf[i - Nl][j] = w[i * var + j];
        }
    }
    delete[] w;

    
    
    
    
    

    
    
    
    
    
    
    

    this->origin2Conservation();

    for (int i = 0, k = 0; i < ncell; i++){
        for (int j = 0; j < VAR2; j++){
            ns_sigmaq[i][j] = 0;
            d_sigmaq[i][j] = 0;
        }
    }

    return true;
}

bool MacroSolver::initialBC()
{
    double **qfr = this->qfr;
    double Ma = mess.Ma, theta = 0;
    
    
    
    double rho = 1, ux = Ma*sqrt(mess.gamma) * this->mess.v_in / this->mess.v_rms * sqrt(3) / 2, uy = Ma*sqrt(mess.gamma) * this->mess.v_in / this->mess.v_rms / 2, uz = 0,t = this->mess.T_in/this->mess.T_ref;
    for (int i = 0, faceIdx; i < this->farVis.size(); i++){
        faceIdx  = this->farVis[i];
        qfr[faceIdx][0] = rho;
        qfr[faceIdx][1] = ux;
        qfr[faceIdx][2] = uy;
        qfr[faceIdx][3] = uz;
        qfr[faceIdx][4] = t;
        qfr[faceIdx][5] = t;
    }

    return true;
}

bool MacroSolver::updateBC(bool gsisTag)
{
    if (gsisTag){
        return true;
    }else{
        double *qfLoc = new double [this->var], t = 1.0;
        qfLoc[0] = 1; qfLoc[1] = 0; qfLoc[2] = 0; qfLoc[3] = 0; qfLoc[4] = t; qfLoc[5] = t;

        this->noSlipIsothermalWall2(qfLoc, this->wallVis);

        
        
        this->pressureOutlet(0.01493, this->Pout);

        this->massWall(0.33, 1, this->Pin);

        

        

        
        

        

        
        

        

        

        

        
        

        

        

        

        

        
        
        delete[] qfLoc;
    }

    
    return true;
}
bool MacroSolver::noSlipIsothermalWall2(double *qf, vector<int> wallVis)
{
    double var = this->var;
    double **qfr = this->qfr;
    double rho = qf[0], ux = qf[1], uy = qf[2], uz = qf[3], ttra = qf[4], trot = qf[5];
    double rate = 1.01, ul, vl , wl, ur, vr, wr, *enormal = NULL;
    double x1, x2, x3, y1, y2, y3, z1, z2, z3;

    if(this->tVector == NULL){
        this->CalculateTangentialVector();
    }
    

    for(int i = 0, l, faceIdx, index; i < wallVis.size(); i++){
        faceIdx = wallVis[i];
        l = edges[faceIdx].faceMap[NN - 2];
        if(l >= iNcell || faceIdx >= Nface || l < 0){
            printf("Error: noSlipIsothermalWall2() invalid face, l = %d, fi = %d\n", l, faceIdx);
        }
        enormal = edges[faceIdx].edgeNormal;
        index = (faceIdx - eNface) * 2 * DIM;
        x1 = enormal[0], y1 = enormal[1], z1 = enormal[2];
        x2 = tVector[index + 0], y2 = tVector[index + 1], z2 = tVector[index + 2];
        x3 = tVector[index + 3], y3 = tVector[index + 4], z3 = tVector[index + 5];

        ul = Qf[l][1] * x1 + Qf[l][2] * y1 + Qf[l][3] * z1;
        vl = Qf[l][1] * x2 + Qf[l][2] * y2 + Qf[l][3] * z2;
        wl = Qf[l][1] * x3 + Qf[l][2] * y3 + Qf[l][3] * z3;

        ur = -ul; vr = -vl; wr = -wl;

        qfr[faceIdx][1] = ur * x1 + vr * x2 + wr * x3;
        qfr[faceIdx][2] = ur * y1 + vr * y2 + wr * y3;
        qfr[faceIdx][3] = ur * z1 + vr * z2 + wr * z3;

        qfr[faceIdx][4] = 2 * ttra - Qf[l][4];
        qfr[faceIdx][5] = 2 * trot - Qf[l][5];

        if (qfr[faceIdx][4] <= rate * ttra){
            qfr[faceIdx][4] = ttra;
        }
        if (qfr[faceIdx][5] <= rate * trot){
            qfr[faceIdx][5] = trot;
        }
        qfr[faceIdx][0] = Qf[l][0] * Qf[l][4] / qfr[faceIdx][4];
    }

    return true;
}
bool MacroSolver::pressureOutlet(double pout, vector<int> wallVis)
{
    double **fg = this->qfr;
    double gamma = mess.gamma, rho, tt, tr;
    int l = 0, r = 0, v;
    double c, ux, uy = 0.0, uz, c2, **qf = this->Qf;
    for(int i=0; i<wallVis.size(); i++){
        v = wallVis[i];
        l = edges[v].faceMap[NN - 2];
        if(l >= iNcell || v >= Nface || l < 0){
            printf("Error: pressureOutlet() invalid face, l = %d, fi = %d\n", l, v);
        }
        rho = qf[l][0]; ux = qf[l][1]; uy = qf[l][2]; uz = qf[l][3];
        tt = qf[l][4]; tr = qf[l][5];

        c = sqrt(gamma * tt);
        c2 = (rho * tt - pout) / rho / c;

        ux += edges[v].edgeNormal[0] * c2;
        uy += edges[v].edgeNormal[1] * c2;
        uz += edges[v].edgeNormal[2] * c2;
        rho += (pout - rho * tt) / pow(c, 2);
        tt = pout / rho;

        

        fg[v][0] = rho; fg[v][1] = ux; fg[v][2] = uy; fg[v][3] = uz;
        fg[v][4] = tt; fg[v][5] = tr;
    }

    return true;
}
bool MacroSolver::massWall(double fin, double t, vector<int> wallVis)
{

    

    double **fg = this->qfr;
    int l = 0, r = 0, v;
    double rho, tt, tr, ux, uy = 0.0, uz, u, **qf = this->Qf;

    
    
    
    
    
    

    
    
    
    

    for (int i = 0; i < wallVis.size(); i++){
        v = wallVis[i];
        l = edges[v].faceMap[NN - 2];
        rho = qf[l][0];

        
        tt = t; tr = t;

        u = fin / rho;
        ux = -edges[v].edgeNormal[0] * u;
        uy = -edges[v].edgeNormal[1] * u;
        uz = -edges[v].edgeNormal[2] * u;

        fg[v][0] = rho; fg[v][1] = ux; fg[v][2] = uy; fg[v][3] = uz;
        fg[v][4] = tt; fg[v][5] = tr;
    }

    return true;
}
bool MacroSolver::symmetry(vector<int> wallVis)
{
    double **fg = this->qfr;
    double rho, tt, tr, **qf = this->Qf;
    int l = 0, r = 0, v;
    double c, ux, uy = 0.0, uz, uxt, uyt, uzt, un;
    for(int i=0; i<wallVis.size(); i++){
        v = wallVis[i];
        l = edges[v].faceMap[NN - 2];
        rho = qf[l][0]; ux = qf[l][1]; uy = qf[l][2]; uz = qf[l][3];
        tt = qf[l][4]; tr = qf[l][5];

        un = ux * edges[v].edgeNormal[0] + uy * edges[v].edgeNormal[1] + uz * edges[v].edgeNormal[2];
        uxt = ux - 2.0 * un * edges[v].edgeNormal[0];
        uyt = uy - 2.0 * un * edges[v].edgeNormal[1];
        uzt = uz - 2.0 * un * edges[v].edgeNormal[2];

        fg[v][0] = rho; fg[v][1] = uxt; fg[v][2] = uyt; fg[v][3] = uzt;
        fg[v][4] = tt; fg[v][5] = tr;
    }

    return true;
}
bool MacroSolver::reconstruction(bool gsisTag)
{
    
    this->limiter2(gsisTag);
    return true;
}
bool MacroSolver::noreconstrucion(bool gsisTag)
{
    int ncell = this->Ncell, nface = this->Nface, var = this->var;
    int incell = this->iNcell, inface = this->iNface, enface = this->eNface;
    edge *edges = this->edges; int nl = this->Nl;
    double **qfl = this->qfl, **qfr = this->qfr;

    int l, r;
    for (int i = 0; i < nface; i++){
        l = edges[i].faceMap[NN - 2];
        r = edges[i].faceMap[NN - 1];

        for (int j = 0; j < var; j++){
            if(l < incell){
                qfl[i][j] = Qf[l][j];
            }else if(l < ncell){
                qfl[i][j] = 0;
            }
            if(r < incell){
                qfr[i][j] = Qf[r][j];
            }else if(r < ncell){
                qfr[i][j] = 0;
            }
        }
    }

    return true;
}

bool MacroSolver::calcTimestep()
{
    int ncell = this->Ncell, incell = this->iNcell, var = this->var;
    double *dt = this->delta_t, pi = M_PI;
    double omega = mess.omega, delta_rp = mess.delta_rp,zr = mess.zr, gamma = mess.gamma;
    double cfl = mess.cfl_ns;
    double rho, ux, uy, uz, ttra, mu, lamv, lamc, c, c2, l;
    double **qf = this->Qf;

    for (int i = 0; i < incell; i++){
        rho = qf[i][0]; ux = qf[i][1]; uy = qf[i][2]; uz = qf[i][3]; ttra = qf[i][4];
        mu = pow(fabs(ttra), omega) / delta_rp;
        lamv = 0.0 * zr * mu / rho / cells[i].area;
        c2 = pow(ux, 2) + pow(uy, 2) + pow(uz, 2.0);
        c = sqrt(fabs(gamma * ttra));
        lamc = sqrt(c2) + c;
        l = sqrt(cells[i].area / pi);
        
        dt[i] = cfl * l / (lamv + lamc);
    }

    return true;
}

void MacroSolver::rhs2Zero()
{
    int ncell = this->Ncell, incell = this->iNcell, var = this->var;
    double **Q = this->RHS;

    for (int i = 0; i < incell; i++){
        for(int j=0; j<var; j++){
            Q[i][j] = 0;
        }
    }
}

bool MacroSolver::calcSource()
{
    int ncell = this->Ncell, incell = this->iNcell, var = this->var;
    double omega = mess.omega, delta_rp = mess.delta_rp, dr = mess.dr, zr = mess.zr;
    double rho, ttra, trot, ttot, tau;
    double **Q = this->RHS;
    double **qf = this->Qf;

    for (int i = 0; i < incell; i++){
        rho = qf[i][0]; ttra = qf[i][4]; trot = qf[i][5];
        ttot = (3.0 * ttra + dr * trot) / (dr + 3.0);
        tau = pow(fabs(ttra), omega - 1) / rho / delta_rp;

        
        Q[i][5] += rho * dr / 2.0 * (ttot - trot) / zr / tau * cells[i].area;
    }

    return true;
}

bool MacroSolver::convectionFlux(int tag, bool gsisTag)
{
    switch(tag){
        case 1:
            this->Rusanov(gsisTag);
            break;
        case 2:
            this->AUSM_Plus_Up(gsisTag);
            break;
        case 3:
            this->AUSMPWPlus(gsisTag);
            break;  
        default:
            break;
    }
    return true;
}

bool MacroSolver::Rusanov(bool gsisTag)
{
    double gamma = mess.gamma;
    int inface = this->iNface, nface = this->Nface, enface = this->eNface;
    int ncell = this->Ncell, incell = this->iNcell, var = this->var;
    edge *edges = this->edges;
    cell *cells = this->cells;
    double **qfl = this->qfl, **qfr = this->qfr, **rhs = this->RHS;
    double unl, unr, un, c, pl, pr, flux, rhol ,rhor;
    if(gsisTag){
        nface = this->eNface;
    }else{
        nface = this->Nface;
    }

    double *qflLoc = new double[var], *qfrLoc = new double[var];
    double *QclLoc = new double[var], *QcrLoc = new double[var];
    int l, r;
    double *sbuf = this->sendBuf;
    
    
    
    
    

    for (int i = 0; i < nface; i++){
        l = edges[i].faceMap[NN - 2];
        r = edges[i].faceMap[NN - 1];
        if (r >= ncell && gsisTag){
            continue;
        }

        for (int j = 0; j < var; j++){
            qflLoc[j] = qfl[i][j];
            qfrLoc[j] = qfr[i][j];
        }

        this->qf2Qc(qflLoc, QclLoc);
        this->qf2Qc(qfrLoc, QcrLoc);
        
        unl = this->convectionFunction(qflLoc, QclLoc, edges[i].edgeNormal);
        unr = this->convectionFunction(qfrLoc, QcrLoc, edges[i].edgeNormal);

        unl = Qf[l][1] * edges[i].edgeNormal[0] + Qf[l][2] * edges[i].edgeNormal[1] + Qf[l][3] * edges[i].edgeNormal[2];
        rhol = Qf[l][0];
        pl = Qf[l][0] * Qf[l][4];
        if(r < ncell){
            unr = Qf[r][1] * edges[i].edgeNormal[0] + Qf[r][2] * edges[i].edgeNormal[1] + Qf[r][3] * edges[i].edgeNormal[2];
            rhor = Qf[r][0];
            pr = Qf[r][0] * Qf[r][4];
        }else{
            unr = unl;
            rhor = rhol;
            pr = pl;
        }

        un = fabs(unl + unr) * 0.5;
        
        c = gamma * (pl + pr) / (rhol + rhor);
        if (c < 0){
            c = gamma;
        }
        c = sqrt(c);
        c += un;
        for (int j = 0; j < var; j++){
            flux = 0.5 * (qflLoc[j] + qfrLoc[j] - c * (QcrLoc[j] - QclLoc[j])) * edges[i].length;

            if(l < incell){
                rhs[l][j] -= flux;
            }else if(l < ncell){
                sbuf[(i - inface) * var + j] = -flux;
            }
            if (r < incell){
                rhs[r][j] += flux;
            }else if(r < ncell){
                sbuf[(i - inface) * var + j] = flux;
            }
            
        }
        
    }

    delete[] qflLoc; delete[] qfrLoc;
    delete[] QclLoc; delete[] QcrLoc;

    return true;
}

bool MacroSolver::viscousFlux(bool gsisTag)
{
    int ncell = this->Ncell, incell = this->iNcell, var = this->var;
    int nface = this->Nface, inface = this->iNface, enface = this->eNface;

    double omega = mess.omega, dr = mess.dr, delta_rp = mess.delta_rp;
    double **sigmaq = this->ns_sigmaq, **dsigmaq = this->d_sigmaq;
    double **qf = this->Qf, **rhs = this->RHS;
    double u, v, w;
    double txx, txy, txz, tyx, tyy, tyz, tzx, tzy, tzz;
    double qtx, qty, qtz, qrx, qry, qrz, div, phix, phiy, phiz;
    double *ds = new double[VAR2], *flux = new double[VAR], *enorm = NULL, *sbuf = this->sendBuf;
    int l, r;

    if(gsisTag){
        nface = this->eNface;
    }else{
        nface = this->Nface;
    }

    for (int i = 0; i < nface; i++){
        l = edges[i].faceMap[NN - 2];
        r = edges[i].faceMap[NN - 1];
        if (r >= ncell && gsisTag){
            continue;
        }
        for (int j = 0; j < VAR2; j++){
            if (r >= ncell){
                ds[j] = dsigmaq[l][j] + sigmaq[l][j];
            }else{
                ds[j] = (dsigmaq[l][j] + dsigmaq[r][j]) * 0.5 + (sigmaq[l][j] + sigmaq[r][j]) * 0.5;
            }
        }
        enorm = edges[i].edgeNormal;

        if(r >= ncell){
            u = qf[l][1]; v = qf[l][2]; w = qf[l][3];
        }else{
            u = (qf[l][1] + qf[r][1]) * 0.5;
            v = (qf[l][2] + qf[r][2]) * 0.5;
            w = (qf[l][3] + qf[r][3]) * 0.5;
        }
        
        qtx = ds[0]; qty = ds[1]; qtz = ds[2];
        qrx = ds[3]; qry = ds[4]; qrz = ds[5];
        txx = ds[6]; txy = tyx = ds[7]; txz = tzx = ds[8];
        tyy = ds[9]; tyz = tzy = ds[10]; tzz = ds[11];
        phix = u * txx + v * txy + w * txz + qtx + qrx;
        phiy = u * tyx + v * tyy + w * tyz + qty + qry;
        phiz = u * tzx + v * tzy + w * tzz + qtz + qrz;

        flux[0] = 0;
        flux[1] = (enorm[0] * txx + enorm[1] * txy + enorm[2] * txz) * edges[i].length;
        flux[2] = (enorm[0] * tyx + enorm[1] * tyy + enorm[2] * tyz) * edges[i].length;
        flux[3] = (enorm[0] * tzx + enorm[1] * tzy + enorm[2] * tzz) * edges[i].length;
        flux[4] = (enorm[0] * phix + enorm[1] * phiy + enorm[2] * phiz) * edges[i].length;
        flux[5] = (enorm[0] * qrx + enorm[1] * qry + enorm[2] * qrz) * edges[i].length;

        for (int j = 0; j < var; j++){
            if(l < incell){
                rhs[l][j] += flux[j];
            }else if(l < ncell){
                sbuf[(i - inface) * var + j] += flux[j];
            }
            if (r < incell){
                rhs[r][j] -= flux[j];
            }else if(r < ncell){
                sbuf[(i - inface) * var + j] -= flux[j];
            }
        }
    }

    delete[] ds;
    delete[] flux;
    return true;
}
bool MacroSolver::implicitSolver(bool gsisTag)
{
    int maxIter = 3;
    if(iSize == 1){
        maxIter = 1;
    }
    int ncell = this->Ncell, incell = this->iNcell, var = this->var;
    int nface = this->Nface, inface = this->iNface, enface = this->eNface;
    double dr = mess.dr, zr = mess.zr, omega = mess.omega, gamma = mess.gamma, delta_rp = mess.delta_rp; 
    double **rhs = this->RHS, **dQ = this->impf, *dt = this->delta_t, **qf = this->Qf;
    double *lambda = new double[nface], *cell_la = new double[ncell];
    double *qflLoc = new double[var], *QclLoc = new double[var];
    double *qfrLoc = new double[var], *QcrLoc = new double[var];
    double *norm = new double[DIM], *cres = new double[var];
    for (int i = 0; i < ncell; i++){
        cell_la[i] = 0;
    }

    this->eigen(lambda);

    int l, r;
    for (int i = 0; i < nface; i++){
        l = edges[i].faceMap[NN - 2];
        r = edges[i].faceMap[NN - 1];
        cell_la[l] += lambda[i];
        if(r < ncell){
            cell_la[r] += lambda[i];
        }
    }

    for (int i = 0; i < ncell; i++){
        for (int j = 0; j < var; j++){
            dQ[i][j] = 0;
        }
    }

    double coe_omega = 3.0, di,rho, ttra, trot, ttot, tau, h4;
    double *D = new double[var];
    if(gsisTag){
        coe_omega = 8.0;
    }
    for (int iter = 1, i, j, ei, si; iter <= maxIter; iter++){
        if(iter != 1){
            this->packDeltaW();
            MPI_Startall(2 * iSize, &this->request[4 * iSize]);
            MPI_Waitall(2 * iSize, &this->request[4 * iSize], &this->status[4 * iSize]);
            this->unPackDeltaW();
        }
        for (int v = 0; v < this->impVis.size(); v++){
            i = this->impVis[v];
            
            
            
            
            
            
            for (int k = 0; k < var; k++){
                cres[k] = 0;
            }
            for (int u = 0; u < cells[i].num; u++){
                j = cells[i].cell2cell[u];
                ei = cells[i].cell2face[u];
                si = cells[i].cell2face_sgn[u];

                if(j >= ncell || ei == -1){
                    break;
                }

                
                
                
                
                norm[0] = edges[ei].edgeNormal[0] * si;
                norm[1] = edges[ei].edgeNormal[1] * si;
                norm[2] = edges[ei].edgeNormal[2] * si;

                for (int k = 0; k < var; k++){
                    qflLoc[k] = qf[j][k];
                    QclLoc[k] = Qc[j][k];
                    QcrLoc[k] = Qc[j][k] + dQ[j][k];
                }
                this->Qc2qf(QcrLoc, qfrLoc);
                this->normalFlux(qflLoc, QclLoc, norm);
                this->normalFlux(qfrLoc, QcrLoc, norm);
                for(int k = 0; k < var; k++){
                    cres[k] += (qfrLoc[k] - qflLoc[k]) * edges[ei].length - lambda[ei] * dQ[j][k];
                }
            }
            
            di = cells[i].area / dt[i] + coe_omega * 0.5 * cell_la[i];
            rho = qf[i][0]; ttra = qf[i][4]; trot = qf[i][5];
            ttot = (3 * ttra + dr * trot) / (3 + dr);
            tau = pow(fabs(ttra), omega - 1) / rho / delta_rp;
            h4 = (-3.0 / (3.0 + dr) + 2.0 * (omega - 1.0) / 3.0 * (ttot - trot) / ttra) / tau / zr * cells[i].area;
            D[0] = D[1] = D[2] = D[3] = D[4] = D[5] = di;
            D[5] -= h4;

            for (int k = 0; k < var; k++){
                if (maxIter == 1){
                    if (v >= incell){
                        dQ[i][k] -= cres[k] * 0.5 / D[k];
                    }else{
                        dQ[i][k] = (rhs[i][k] - cres[k] * 0.5) / D[k];
                    }
                }else{
                    dQ[i][k] = (rhs[i][k] - cres[k] * 0.5) / D[k];
                }
            }
        }
    }

    for(int i = 0; i < incell; i++){
        for(int j = 0; j < var; j++){
            Qc[i][j] += dQ[i][j];
        }
    }

    this->conservation2Origin();
    
    
    
    
    delete[] lambda; delete[] cell_la;
    delete[] qflLoc; delete[] QclLoc;
    delete[] qfrLoc; delete[] QcrLoc;
    delete[] norm; delete[] cres;
 
    return true;
}

bool MacroSolver::nsProcess(int maxIter, double maxError, bool gsisTag, int ITER)
{
        if (!active()) return true;
        int ncell = this->Ncell, incell = this->iNcell, var = this->var;
        int size = this->iSize, rank = this->iRank;
        int iter = 0;
        double err = 1;
        meshImport *mesh = this->mesh;

        char filename[100];
        
        double **wt = NULL;
        wt = new double *[incell];
        for (int i = 0; i < incell; i++){
            wt[i] = new double[var];
        }

        if (!gsisTag){
            this->bcClassification();
            this->initialW();
            this->initialBC();
        }

        if(gsisTag){
            maxError = 1e-6;
        }

        while(iter < maxIter && err > maxError){
            iter++;

            this->packQfQc();
            MPI_Startall(2 * size, &this->request[0]);

            this->rhs2Zero();

            this->numpyDeepCopy2D(incell, var, wt, Qc);

            this->updateBC(gsisTag);

            MPI_Waitall(2 * size, &this->request[0], &this->status[0]);
            this->unPackQfQc();

            if(!gsisTag){
                
                
                
                
                
                
                
                this->noreconstrucion(gsisTag);
            }else{
                this->leastSquareGrad();
                this->reconstruction(gsisTag);
                this->calcCellViscous();
                
            }

            if(!gsisTag){
                this->convectionFlux(1, gsisTag);
                
                
                
                
            }else{
                this->convectionFlux(1, gsisTag);
                this->viscousFlux(gsisTag);
            }

            MPI_Startall(2 * size, &this->request[2 * size]);

            this->calcSource();
            this->calcTimestep();

            if (gsisTag){
                
                this->applyNSEG13BoundaryFluxes();
            }

            MPI_Waitall(2 * size, &this->request[2 * size], &this->status[2 * size]);
            this->unPackFlux();

            
            this->implicitSolver(gsisTag);

            if (gsisTag && iter % 20 == 0 && iter != 0)
            {
                err = this->calcMaxError(wt, Qc);
                if(activeLeader()){
                    printf("ns : iter = %d, Error = %e \n", iter, err);
                }
            }

            if (!gsisTag && iter % 100 == 0 && iter != 0)
            {
                err = this->calcMaxError(wt, Qc);
                if(activeLeader()){
                    printf("ns : iter = %d, Error = %e \n", iter, err);
                }

                
                
                
                
                
            }
            
        }

        for (int i = 0; i < incell; i++) {
            delete[] wt[i];
        }
        delete[] wt;
    
    
    return true;
}

bool MacroSolver::nsProcessG13(int maxIter, double maxError, bool gsisTag, int ITER)
{
    if (active())
    {
        int ncell = this->Ncell, incell = this->iNcell, var = this->var;
        int size = this->iSize, rank = this->iRank;
        int iter = 0;
        double err = 1;

        char filename[100];
        
        double **wt = NULL;
        wt = new double *[incell];
        for (int i = 0; i < incell; i++){
            wt[i] = new double[var];
        }

        if (!gsisTag){
            this->bcClassification();
            this->initialW();
            this->initialBC();
        }

        if(gsisTag)
        {
            maxError = 1e-6;
        }

        while(iter < maxIter && err > maxError){
            iter++;
            
            this->packQfQc();
            MPI_Startall(2 * size, &this->request[0]);

            this->rhs2Zero();

            this->numpyDeepCopy2D(incell, var, wt, Qc);

            MPI_Waitall(2 * size, &this->request[0], &this->status[0]);
            this->unPackQfQc();

            
            
            
            
            
            
            
            this->leastSquareGrad();

            this->noreconstrucion(gsisTag);
            
            this->calcCellViscous();

            this->convectionFlux(1, true);

            this->viscousFlux(true);
            
            
            
            
            

            MPI_Startall(2 * size, &this->request[2 * size]);
            this->calcSource();
            this->calcTimestep();

            this->applyNSEG13BoundaryFluxes();

            MPI_Waitall(2 * size, &this->request[2 * size], &this->status[2 * size]);
            this->unPackFlux();

            
            this->implicitSolver(gsisTag);

            if (!gsisTag && iter % 50 == 0 && iter != 0)
            {
                err = this->calcMaxError(wt, Qc);
                if(activeLeader())
                {
                    printf("ns : iter = %d, Error = %e \n", iter, err);
                }
                
                
                
                
            }
            
            
            
            
            
        }
        

        for (int i = 0; i < incell; i++) {
            delete[] wt[i];
        }
        delete[] wt;
    }
    
    MPI_Barrier(comm);
    this->nsout2dat(0,this->Qf,this->Nl,this->Nr);

    return true;
}

bool MacroSolver::AUSMPWPlus(bool gsisTag)
{
    int inface = this->iNface, nface = this->Nface, enface = this->eNface;
    int ncell = this->Ncell, incell = this->iNcell, var = this->var;
    edge *edges = this->edges; cell *cells = this->cells;
    double **fgl = this->qfl, **fgr = this->qfr;
    double **rhs = this->RHS, flux;
    double ps = 1.0, mp, mn;

    double *fl = new double[var], *fr = new double[var];

    double *pe = new double[var];
    pe[0] = pe[4] = pe[5] = 0.0;

    double *sbuf = this->sendBuf;

    if(gsisTag){
        nface = this->eNface;
    }else{
        nface = this->Nface;
    }
    
    int l, r;
    for(int i=0; i<nface; i++){
        l = edges[i].faceMap[NN - 2];
        r = edges[i].faceMap[NN - 1];
        if (r >= ncell && gsisTag){
            continue;
        }
        pe[1] = edges[i].edgeNormal[0]; pe[2] = edges[i].edgeNormal[1],pe[3] = edges[i].edgeNormal[2];;
        for(int j=0; j<var; j++){
            fl[j] = Qf[l][j];
            if(r < ncell){
                fr[j] = Qf[r][j];
            }else{
                
                fr[j] = qfr[i][j];
            }
        }

        this->ausmFunction(fl, fr, mp, mn, ps, edges[i].edgeNormal);
        this->clacSlauPsi(qfl[i], fl);
        this->clacSlauPsi(qfr[i], fr);
        
        for(int j=0; j<var; j++){
            if(i < inface || i >= enface){
                flux = (mp * fl[j] * qfl[i][0] + mn * fr[j] * qfr[i][0] + ps * pe[j]) * edges[i].length;
            }else{
                flux = (mp * fl[j] * qfl[i][0] + mn * fr[j] * qfr[i][0] + ps * pe[j] * 0.5) * edges[i].length;
            }
            
            if(l < incell){
                rhs[l][j] -= flux;
            }else if(l < ncell){
                sbuf[(i - inface) * var + j] = -flux;
            }
            if (r < incell){
                rhs[r][j] += flux;
            }else if(r < ncell){
                sbuf[(i - inface) * var + j] = flux;
            }
        }
    }

    delete[] fl; delete[] fr; 
    
    delete[] pe;
    return true;
}

void MacroSolver::AUSM_Plus_Up(bool gsisTag)
{
    int inface = this->iNface, nface = this->Nface, enface = this->eNface;
    int ncell = this->Ncell, incell = this->iNcell, var = this->var;
    edge *edges = this->edges; cell *cells = this->cells;
    double **fgl = this->qfl, **fgr = this->qfr;
    double **rhs = this->RHS, flux;
    double ps = 1.0, mp, mn;

    double *fl = new double[var], *fr = new double[var];

    double *pe = new double[var];
    pe[0] = pe[4] = pe[5] = 0.0;

    double *sbuf = this->sendBuf;

    if(gsisTag){
        nface = this->eNface;
    }else{
        nface = this->Nface;
    }
    
    int l, r;
    for(int i=0; i<nface; i++){
        l = edges[i].faceMap[NN - 2];
        r = edges[i].faceMap[NN - 1];
        if (r >= ncell && gsisTag){
            continue;
        }
        pe[1] = edges[i].edgeNormal[0]; pe[2] = edges[i].edgeNormal[1];
        pe[3] = edges[i].edgeNormal[2];
        for(int j=0; j<var; j++){
            fl[j] = Qf[l][j];
            if(r < ncell){
                fr[j] = Qf[r][j];
            }else{
                
                fr[j] = qfr[i][j];
            }
        }

        this->ausmPlusUpFunction(fl, fr, mp, mn, ps, edges[i].edgeNormal);
        this->clacSlauPsi(qfl[i], fl);
        this->clacSlauPsi(qfr[i], fr);

        
        

        for(int j=0; j<var; j++){
            if(i < inface || i >= enface){
                flux = (mp * fl[j] * qfl[i][0] + mn * fr[j] * qfr[i][0] + ps * pe[j]) * edges[i].length;
            }else{
                flux = (mp * fl[j] * qfl[i][0] + mn * fr[j] * qfr[i][0] + ps * pe[j] * 0.5) * edges[i].length;
            }

            
            
            
            if(l < incell){
                rhs[l][j] -= flux;
            }else if(l < ncell){
                sbuf[(i - inface) * var + j] = -flux;
            }
            if (r < incell){
                rhs[r][j] += flux;
            }else if(r < ncell){
                sbuf[(i - inface) * var + j] = flux;
            }
        }
        
        
    }

    delete[] fl; delete[] fr; 
    delete[] pe;
}

void MacroSolver::clacSlauPsi(const double *qf, double *psi)
{
    double rho = qf[0], u = qf[1], v = qf[2], w = qf[3], ttra = qf[4], trot = qf[5];
    double ht, hr, c, c2, tot;
    double gamma = mess.gamma, dr = mess.dr, cv = mess.cv;

    c2 = u * u + v * v + w * w;
    tot = (3.0 * ttra + dr * trot) / (3.0 + dr);
    
    ht = cv * tot + 0.5 * c2 + ttra;
    
    
    
    hr = dr * 0.5 * trot;

    if(fabs(rho) < 1e-10){
        psi[0] = 0; psi[4] = 0; psi[5] = 0;
    }else{
        psi[0] = 1.0; psi[4] = ht; psi[5] = hr;
    }
    psi[1] = qf[1]; psi[2] = qf[2]; psi[3] = qf[3];
}

void MacroSolver::ausmPlusUpFunction(double *fl, double *fr, double &mlpt, double &mrnt, double &ps, const double *norm)
{
    
    double mlp, mrn, plp, prn, cl ,cr, pl, pr, mal, mar, unl, unr, gamma = mess.gamma;
    double M2, Mo, Mo2, Mh, rhol, rhor, rhoh, ch, ch2, fa, mh;
    double Kp = 0.25, Ku = 0.75, sig = 1.0, Mi = mess.Ma, Mco2 = 0.09, ka = 0.1;
    double alpha, beta = 0.125;
    double Hl, Hr, asl, asr, asl2, asr2, cv = mess.cv, Tl, Tr, dr = mess.dr, Ht, as;

    
    Mco2 = max(0.09, mess.Ma * mess.Ma / mess.gamma);
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    rhol = fl[0]; rhor = fr[0]; rhoh = (rhol + rhor) * 0.5;
    cl = sqrt(gamma * fl[4]); cr = sqrt(gamma * fr[4]);
    
    unl = fl[1] * norm[0] + fl[2] * norm[1] + fl[3] * norm[2]; 
    unr = fr[1] * norm[0] + fr[2] * norm[1] + fr[3] * norm[2];


    Hl = (cv * (3.0 * fl[4] + dr * fl[5]) / (3.0 + dr) + fl[4]);
    Hr = (cv * (3.0 * fr[4] + dr * fr[5]) / (3.0 + dr) + fr[4]);
    Ht = 0.5 * (Hl + Hr);
    as = sqrt(2.0 * (gamma - 1.0) / (gamma + 1.0) * Ht);
    asl = as * as / max(as, unl);
    asr = as * as / max(as, -unr);
    ch = min(asl, asr); ch2 = ch * ch;


    

    mal = unl / ch; mar = unr / ch;
    M2 = (unl * unl + unr * unr) / ch2 * 0.5;
    
    Mo = sqrt(min(1.0, max(M2, Mco2)));
    fa = Mo * (2.0 - Mo);
    pl = fl[0] * fl[4]; pr = fr[0] * fr[4];
    mlp = this->maSplit4(mal, 1, beta);
    mrn = this->maSplit4(mar, -1, beta);
    Mh = mlp + mrn - Kp / fa * max(1.0 - sig * M2, 0.0) * (pr - pl) / (rhoh * ch2);

    if(Mh > 0){
        mlpt = ch * Mh; mrnt = 0;
    }else{
        mlpt = 0; mrnt = ch * Mh;
    }

    alpha = 0.1875 * (-4.0 + 5.0 * fa * fa);
    plp = this->pSplit5(mal, 1, alpha); prn = this->pSplit5(mar, -1, alpha);

    ps = plp * pl + prn * pr - Ku * plp * prn * (rhol + rhor) * fa * ch * (unr - unl);

}

void MacroSolver::ausmFunction(double *fl, double *fr, double &mlpt, double &mrnt, double &ps, const double *norm)
{
    
    double mlp, mrn, Fl, Fr, w, plp, prn, cl ,cr, pl, pr, c, mal, mar, ma, unl, unr;
    double gamma = mess.gamma, dr = mess.dr, cv = mess.cv;

    pl = fl[0] * fl[4]; pr = fr[0] * fr[4];
    cl = sqrt(gamma * fl[4]); cr = sqrt(gamma * fr[4]);
    c = (cl + cr) * 0.5;
    unl = fl[1] * norm[0] + fl[2] * norm[1] + fl[3] * norm[2]; unr = fr[1] * norm[0] + fr[2] * norm[1] + fr[3] * norm[2];
    mal = unl / c; mar = unr / c;
    mlp = this->msLeft(mal); mrn = this->msRight(mar);
    ma = mlp + mrn;
    plp = this->psLeft(mal); prn = this->psRight(mar);
    ps = plp * pl + prn * pr;
    Fl = this->ausmF(pl, ps, mal); Fr = this->ausmF(pr, ps, mar);
    w = 1.0 - pow(min(pl / pr, pr / pl), 3);

    if(ma < 0){
        mlpt = mlp * w * (1.0 + Fl);
        mrnt = mrn + mlp * ((1.0 - w) * (1.0 + Fl) - Fr);
    }else{
        mlpt = mlp + mrn * ((1.0 - w) * (1.0 + Fr) - Fl);
        mrnt = mrn * w * (1.0 + Fr);
    }

    mlpt *= c; mrnt *= c;
}
double MacroSolver::ausmF(const double &p, const double &ps, const double &m)
{
    double f = 0;
    if(fabs(m) <= 1 && ps != 0){
        f = p / ps - 1;
    }

    return f;
}

double MacroSolver::psLeft(const double ma)
{
    double p = 0;
    if(fabs(ma) < 1.0){
        p = 0.25 * pow(ma + 1.0, 2) * (2.0 - ma);
    }else if(ma >= 1.0){
        p = 1.0;
    }else{
        p = 0;
    }

    return p;
}
double MacroSolver::psRight(const double ma)
{
    double p = 0;
    if(fabs(ma) < 1.0){
        p = 0.25 * pow(ma - 1.0, 2) * (2.0 + ma);
    }else if(ma >= 1.0){
        p = 0;
    }else{
        p = 1.0;
    }

    return p;
}

double MacroSolver::msLeft(const double ma)
{
    double p = 0;
    if(fabs(ma) < 1.0){
        p = 0.25 * pow(ma + 1.0, 2);
    }else if(ma >= 1.0){
        p = ma;
    }else{
        p = 0;
    }

    return p;
}
double MacroSolver::msRight(const double ma)
{
    double p = 0;
    if(fabs(ma) < 1.0){
        p = -0.25 * pow(ma - 1.0, 2);
    }else if(ma >= 1.0){
        p = 0.0;
    }else{
        p = ma;
    }

    return p;
}


double MacroSolver::maSplit1(double ma, const int &sgn)
{
    return 0.5 * (ma + sgn * fabs(ma));
}


double MacroSolver::maSplit2(double ma, const int &sgn)
{
    return 0.25 * sgn * (ma + sgn * 1.0) * (ma + sgn * 1.0);
}


double MacroSolver::maSplit4(double ma, const int &sgn, const double &beta)
{
    if(fabs(ma) >= 1.0){
        return this->maSplit1(ma, sgn);
    }
    return this->maSplit2(ma, sgn) * (1.0 - sgn * 16.0 * beta * this->maSplit2(ma, -sgn));
}


double MacroSolver::pSplit5(double ma, const int &sgn, const double &alpha)
{
    if(fabs(ma) >= 1.0){
        return this->maSplit1(ma, sgn) / ma;
    }
    return this->maSplit2(ma, sgn) * ((sgn * 2 - ma) - sgn * 16.0 * alpha * ma * this->maSplit2(ma, -sgn));
}

double MacroSolver::convectionFunction(double *qf, double *Qc, double *enormal)
{
    double rho = qf[0], u = qf[1], v = qf[2], w = qf[3], ttra= qf[4], trot = qf[5];
    double un = u * enormal[0] + v * enormal[1] + w * enormal[2], p = rho * ttra;


    qf[0] = un * rho;
    qf[1] = un * Qc[1] + p * enormal[0];
    qf[2] = un * Qc[2] + p * enormal[1];
    qf[3] = un * Qc[3] + p * enormal[2];
    qf[4] = un * (Qc[4] + p);
    qf[5] = un * Qc[5];

    return un;
}

void MacroSolver::eigen(double *lambda)
{
    int ncell = this->Ncell, incell = this->iNcell, var = this->var;
    int nface = this->Nface, inface = this->iNface, enface = this->eNface;
    double dr = mess.dr, zr = mess.zr, omega = mess.omega, pr = mess.pr;
    double gamma = mess.gamma, delta_rp = mess.delta_rp, **qf = this->Qf;

    int l, r;
    double unl, unr, p, c, un, rho, ttra, mu;
    double lamA, coeA = 1, rexA = 0, visA;

    for (int i = 0; i < nface; i++){
        l = edges[i].faceMap[NN - 2];
        r = edges[i].faceMap[NN - 1];
        if (r >= ncell){
            r = l;
        }
        rho = (qf[l][0] + qf[r][0]) * 0.5;
        ttra = (qf[l][4] + qf[r][4]) * 0.5;
        mu = pow(fabs(ttra), omega) / delta_rp;
        un =  qf[l][1] * edges[i].edgeNormal[0] + qf[l][2] * edges[i].edgeNormal[1] + qf[l][3] * edges[i].edgeNormal[2];
        un += qf[r][1] * edges[i].edgeNormal[0] + qf[r][2] * edges[i].edgeNormal[1] + qf[r][3] * edges[i].edgeNormal[2];
        un = fabs(un) * 0.5;
        p = qf[l][0] * qf[l][4] + qf[r][0] * qf[r][4];
        c = sqrt(fabs(gamma * p / rho / 2.0));
        lamA = un + c;
        visA = 1.0 * zr * max(4.0 / 3.0, gamma) * mu / rho / edges[i].edgeDist / pr;

        lambda[i] = (2.0 * lamA + coeA * visA + rexA) * edges[i].length;
    }

}

bool MacroSolver::normalFlux(double *qf, double *Qc, double *enormal)
{
    double rho = qf[0], u = qf[1], v = qf[2], w = qf[3], ttra= qf[4], trot = qf[5];
    double un = u * enormal[0] + v * enormal[1] + w * enormal[2], p = rho * ttra;

    qf[0] = un * rho;
    qf[1] = un * Qc[1] + p * enormal[0];
    qf[2] = un * Qc[2] + p * enormal[1];
    qf[3] = un * Qc[3] + p * enormal[2];
    qf[4] = un * (Qc[4] + p);
    qf[5] = un * Qc[5];

    return true;
}

bool MacroSolver::origin2Conservation()
{
    int ncell = this->Ncell, incell = this->iNcell, var = this->var;
    double *qfLoc = new double[var], *QcLoc = new double[var];
    double **qf = this->Qf;

    for (int i = 0; i < incell; i++){
        for (int j = 0; j < var; j++){
            qfLoc[j] = qf[i][j];
        }
        this->qf2Qc(qfLoc, QcLoc);
        for (int j = 0; j < var; j++){
            Qc[i][j] = QcLoc[j];
        }
    }

    delete[] qfLoc, delete[] QcLoc;
    return true;
}

bool MacroSolver::conservation2Origin()
{
    int ncell = this->Ncell, incell = this->iNcell, var = this->var;
    double *qfloc = new double[var], *Qcloc = new double[var];
    double **qf = this->Qf;

    for (int i = 0; i < incell; i++){
        for (int j = 0; j < var; j++){
            Qcloc[j] = Qc[i][j];
        }
        this->Qc2qf(Qcloc, qfloc);
        for (int j = 0; j < var; j++){
            qf[i][j] = qfloc[j];
        }
    }

    delete[] qfloc;
    delete[] Qcloc;
    return true;
}

bool MacroSolver::qf2Qc(double *qf, double *Qc)
{
    double dr = mess.dr, cv = mess.cv;
    double rho = qf[0], ux = qf[1], uy = qf[2], uz = qf[3], ttra = qf[4], trot = qf[5];
    double c2 = pow(ux, 2) + pow(uy, 2) + pow(uz, 2), ttot = (3.0 * ttra + dr * trot) / (3.0 + dr);
    
    Qc[0] = rho;
    Qc[1] = rho * ux; 
    Qc[2] = rho * uy;
    Qc[3] = rho * uz;
    Qc[4] = rho * (cv * ttot + 0.5 * c2);
    Qc[5] = 0.5 * dr * rho * trot;

    return true;
}

bool MacroSolver::Qc2qf(double *Qc, double *qf)
{
    double dr = mess.dr, cv = mess.cv, cp = mess.cp;
    double rho = Qc[0], ux = Qc[1] / Qc[0], uy = Qc[2] / Qc[0], uz = Qc[3] / Qc[0], ttra, trot, ttot;
    double c2 = pow(ux, 2) + pow(uy, 2) + pow(uz, 2); 
    
    ttot = (Qc[4] / rho - 0.5 * c2) / cv;
    trot = Qc[5] / rho / dr * 2.0;
    ttra = ((3.0 + dr) * ttot - dr * trot) / 3.0;

    qf[0] = rho; 
    qf[1] = ux; 
    qf[2] = uy; 
    qf[3] = uz; 
    qf[4] = ttra; 
    qf[5] = trot;

    return true;
}


void MacroSolver::initialParts()
{
    int ncell = this->Ncell, nface = this->Nface, var = this->var;
    int incell = this->iNcell, inface = this->iNface, enface = this->eNface;
    
    int *send = this->sendCount, * recv = this->recCount;
    int encell = ncell - incell, size = iSize, rank = iRank;
    
    unordered_map<int, bool> vis;
    
    for(int k=1; k<size+1; k++){
        send[k] = 0; recv[k] = 0;
        for(int i=nParts[k-1], l, r; i<nParts[k]; i++){
            l = edges[i].faceMap[NN - 2];
            r = edges[i].faceMap[NN - 1];

            if(l >= incell){
                if(!vis[l]){
                    vis[l] = true;
                    this->recCell.push_back(l);
                    recv[k] ++;
                }
                if(!vis[r]){
                    vis[r] = true;
                    this->sendCell.push_back(r);
                    send[k] ++;
                }
                continue;
            }
            if(!vis[l]){
                vis[l] = true;
                this->sendCell.push_back(l);
                send[k] ++;
            }
            if(!vis[r]){
                vis[r] = true;
                this->recCell.push_back(r);
                recv[k] ++;
            }
            if(l < incell && r < incell){
                printf("Error: ns initialParts failed\n");
                break;
            }
        }

        vis.clear();
    }


    send[0] = 0; recv[0] = 0;
    for(int i=1; i<size+1; i++){
        send[i] += send[i - 1];
        recv[i] += recv[i - 1];
    }

    if(this->request == NULL){
        this->request = new MPI_Request[3 * 2 * size];
        this->status = new MPI_Status[3 * 2 * size];
    }


    if((enface - inface) > 2 * this->sendCell.size() || (enface - inface) > 2 * this->recCell.size()){
        printf("Error: (enface - inface) > 2 * (ncell - incell)\n");
        cout<<" rank "<< this->iRank <<" (enface - inface) "<< enface - inface <<" sendCell "<< 2 * this->sendCell.size()<<endl;
    }

    int num = var + var;
    if(this->sendBuf == NULL){
        this->sendBuf = new double[this->sendCell.size() * num];
        this->recBuf = new double[this->recCell.size() * num];  
    }

}


void MacroSolver::initialParallel()
{
    int ncell = this->Ncell, nface = this->Nface, var = this->var;
    int incell = this->iNcell, inface = this->iNface, enface = this->eNface;
    int *nparts = this->nParts;
    int encell = ncell - incell, size = iSize, rank = iRank;
    int num = var + var, *sc = this->sendCount, *rc = this->recCount; 

    

    
    


    
    

    
    
    
    

    double *sbuf = this->sendBuf, *rbuf = this->recBuf;


    for(int i=0, count = 0, dest = -1; i<size; i++){
        count = nparts[i + 1] - nparts[i];
        dest = i;
        if(i == rank || count == 0){
            dest = MPI_PROC_NULL;
        }


        MPI_Send_init(&sbuf[sc[i] * num], (sc[i+1] - sc[i]) * num, MPI_DOUBLE, dest, rank, this->myGroup, &this->request[i]);
        MPI_Recv_init(&rbuf[rc[i] * num], (rc[i+1] - rc[i]) * num, MPI_DOUBLE, dest, i, this->myGroup, &this->request[i + 1 * size]);


        
        


        MPI_Send_init(&sbuf[(nparts[i] - inface) * var], count * var, MPI_DOUBLE, dest, rank, this->myGroup, &this->request[i + 2 * size]);
        MPI_Recv_init(&rbuf[(nparts[i] - inface) * var], count * var, MPI_DOUBLE, dest, i, this->myGroup, &this->request[i + 3 * size]);


        MPI_Send_init(&sbuf[sc[i] * var], (sc[i+1] - sc[i]) * var, MPI_DOUBLE, dest, rank, this->myGroup, &this->request[i + 4 * size]);
        MPI_Recv_init(&rbuf[rc[i] * var], (rc[i+1] - rc[i]) * var, MPI_DOUBLE, dest, i, this->myGroup, &this->request[i + 5 * size]);
    }
}


void MacroSolver::freeParallel()
{
    int size = iSize;

    for(int i=0; i<3*2*size; i++){
        MPI_Request_free(&this->request[i]);
    }
}


void MacroSolver::packQfQc()
{
    int ncell = this->Ncell, nface = this->Nface, var = this->var;
    int incell = this->iNcell, inface = this->iNface, enface = this->eNface;
    int num = var + var;
    double *sbuf = this->sendBuf;

    vector<int> &vis = this->sendCell;

    for(int i=0, ci, lb; i<vis.size(); i++){
        ci = vis[i];
        lb = i * num;
        
        
        for(int j=0; j<var; j++){
            sbuf[lb + j] = Qf[ci][j];
        }
        lb += var;
        for(int j=0; j<var; j++){
            sbuf[lb + j] = Qc[ci][j];
        }
    }
}


void MacroSolver::unPackQfQc()
{
    int ncell = this->Ncell, nface = this->Nface, var = this->var;
    int incell = this->iNcell, inface = this->iNface, enface = this->eNface;
    int num = var + var;
    double *rbuf = this->recBuf;

    vector<int> &vis = this->recCell;

    for(int i=0, ci, lb; i<vis.size(); i++){
        ci = vis[i];
        lb = i * num;
        for(int j=0; j<var; j++){
            Qf[ci][j] = rbuf[lb + j];
        }
        lb += var;
        for(int j=0; j<var; j++){
            Qc[ci][j] = rbuf[lb + j];
        }
    }
}


void MacroSolver::unPackFlux()
{
    int ncell = this->Ncell, nface = this->Nface, var = this->var;
    int incell = this->iNcell, inface = this->iNface, enface = this->eNface;
    double *rbuf = this->recBuf, **rhs = this->RHS;;



    for(int i=inface, l, r, lb; i<enface; i++){
        l = edges[i].faceMap[NN - 2];
        r = edges[i].faceMap[NN - 1];
        lb = (i - inface) * var;

        for (int j = 0; j < var; j++){

            if(l < incell){
                rhs[l][j] += rbuf[lb + j];
                continue;
            }
            rhs[r][j] += rbuf[lb + j];
        }
    }
}


void MacroSolver::packDeltaW()
{
    int ncell = this->Ncell, incell = this->iNcell, var = this->var;
    int nface = this->Nface, inface = this->iNface, enface = this->eNface;
    double *sbuf = this->sendBuf, **impf = this->impf;

    vector<int> &vis = this->sendCell;
    for(int i=0, ci, lb; i<vis.size(); i++){
        ci = vis[i];
        lb = i * var;
        for(int j=0; j<var; j++){
            sbuf[lb + j] = impf[ci][j];
        }
    }
}


void MacroSolver::unPackDeltaW()
{
    int ncell = this->Ncell, nface = this->Nface, var = this->var;
    int incell = this->iNcell, inface = this->iNface, enface = this->eNface;
    double *rbuf = this->recBuf, **impf = this->impf;

    vector<int> &vis = this->recCell;

    for(int i=0, ci, lb; i<vis.size(); i++){
        ci = vis[i];
        lb = i * var;
        for(int j=0; j<var; j++){
            impf[ci][j] = rbuf[lb + j];
        }
    }
}


bool MacroSolver::leastSquareGrad()
{
    int ncell = this->Ncell, incell = this->iNcell, var = this->var;
    int nface = this->Nface, inface = this->iNface, enface = this->eNface;
    int *cell2cellLoc = NULL, dim = DIM;
    
    double **fg = this->Qf;
    double **grad = this->Grad, *b = new double[var * dim];
    double dx, dy, dz, df;


    for (int i = 0, v; i < incell; i++){
        cell2cellLoc = cells[i].cell2cell;
        for (int j = 0; j < var * dim; j++){
            b[j] = 0;
        }
        for (int j = 0; j < cells[i].num; j++){
            v = cell2cellLoc[j];
            if (v >= ncell || v < 0){
                continue;
            }
            dx = cells[i].dxyz[j][0];
            dy = cells[i].dxyz[j][1];
            dz = cells[i].dxyz[j][2];
            for(int k = 0; k < var; k++){
                df = fg[v][k] - fg[i][k];
                b[k * dim + 0] += df * dx;
                b[k * dim + 1] += df * dy;
                b[k * dim + 2] += df * dz;
            }
        }
        for (int k = 0; k < var; k++){
            grad[i][k * dim + 0] = cells[i].Ainv[0][0] * b[k * dim + 0] + cells[i].Ainv[0][1] * b[k * dim + 1] + cells[i].Ainv[0][2] * b[k * dim + 2];
            grad[i][k * dim + 1] = cells[i].Ainv[1][0] * b[k * dim + 0] + cells[i].Ainv[1][1] * b[k * dim + 1] + cells[i].Ainv[1][2] * b[k * dim + 2];
            grad[i][k * dim + 2] = cells[i].Ainv[2][0] * b[k * dim + 0] + cells[i].Ainv[2][1] * b[k * dim + 1] + cells[i].Ainv[2][2] * b[k * dim + 2];

            if (fabs(grad[i][k * dim + 0]) < 1e-12){
                grad[i][k * dim + 0] = 0;
            }
            if (fabs(grad[i][k * dim + 1]) < 1e-12){
                grad[i][k * dim + 1] = 0;
            }
            if (fabs(grad[i][k * dim + 2]) < 1e-12){
                grad[i][k * dim + 2] = 0;
            }
        }
    }
    
    delete[] b;
    return true;
}


void MacroSolver::calcCellViscous()
{
    int ncell = this->Ncell, incell = this->iNcell, var = this->var;
    int nface = this->Nface, inface = this->iNface, enface = this->eNface;

    double omega = mess.omega, dr = mess.dr, delta_rp = mess.delta_rp;
    double **gm = this->Grad, **nss = this->ns_sigmaq;
    double kappat = 3  * 0.5 * mess.f_tra;
    double kappar = dr * 0.5 * mess.f_rot;
    double **qf = this->Qf;
    double u, v, ttra, mu;
    double ux, uy, uz, vx, vy, vz, wx, wy, wz, tx, ty, tz, rx, ry, rz;
    double txx, txy, txz, tyx, tyy, tyz, tzx, tzy, tzz;
    double qtx, qty, qtz, qrx, qry, qrz, div;
    int l, r, dim = DIM; 

    for (int i = 0; i < incell; i++){
        ttra = qf[i][4];
        mu = pow(fabs(ttra), omega) / delta_rp;
        ux = gm[i][1 * dim + 0];  uy = gm[i][1 * dim + 1]; uz = gm[i][1 * dim + 2];
        vx = gm[i][2 * dim + 0];  vy = gm[i][2 * dim + 1]; vz = gm[i][2 * dim + 2];
        wx = gm[i][3 * dim + 0];  wy = gm[i][3 * dim + 1]; wz = gm[i][3 * dim + 2];
        tx = gm[i][4 * dim + 0];  ty = gm[i][4 * dim + 1]; tz = gm[i][4 * dim + 2];
        rx = gm[i][5 * dim + 0];  ry = gm[i][5 * dim + 1]; rz = gm[i][5 * dim + 2];
        div = ux + vy + wz;

        txx = 2.0 * mu * (ux - 1.0 / 3.0 * div);
        tyy = 2.0 * mu * (vy - 1.0 / 3.0 * div);
        tzz = 2.0 * mu * (wz - 1.0 / 3.0 * div);
        txy = tyx = mu * (uy + vx);
        txz = tzx = mu * (uz + wx);
        tyz = tzy = mu * (vz + wy);

        qtx = kappat * mu * tx; qty = kappat * mu * ty; qtz = kappat * mu * tz;
        qrx = kappar * mu * rx; qry = kappar * mu * ry; qrz = kappar * mu * rz;

        nss[i][0] = qtx; nss[i][1] = qty; nss[i][2] = qtz;
        nss[i][3] = qrx; nss[i][4] = qry; nss[i][5] = qrz;
        nss[i][6] = txx; nss[i][7] = txy; nss[i][8] = txz;
        nss[i][9] = tyy; nss[i][10] = tyz; nss[i][11] = tzz;

        
        
        
        
    }
}

bool MacroSolver::limiter(bool gsisTag)
{
    int ncell = this->Ncell, incell = this->iNcell, var = this->var;
    int nface = this->Nface, inface = this->iNface, enface = this->eNface;
    int dim = DIM;
    double **qfl = this->qfl, **qfr = this->qfr, **qf = this->Qf;
    double **grad = this->Grad, **um = this->Umaxmin;

    if(gsisTag){
        nface = enface;
    }

    for (int i = 0; i < incell; i++){
        for(int j = 0; j < var; j++){
            um[i][j * 2 + 0] = qf[i][j];
            um[i][j * 2 + 1] = qf[i][j];
        }
    }

    int l, r;

    for (int i = 0; i < enface; i++){
        l = edges[i].faceMap[NN - 2];
        r = edges[i].faceMap[NN - 1];
        if (r >= ncell){
            continue;
            
        }
        for (int k = 0; k < var; k++){
            if(l < incell){
                um[l][k * 2 + 0] = max(um[l][k * 2 + 0], qf[r][k]);
                um[l][k * 2 + 1] = min(um[l][k * 2 + 1], qf[r][k]);
            }
            if(r < incell){
                um[r][k * 2 + 0] = max(um[r][k * 2 + 0], qf[l][k]);
                um[r][k * 2 + 1] = min(um[r][k * 2 + 1], qf[l][k]);
            }
        }
    }

    double g, limiter = 1;

    for(int i = 0; i < nface; i++){
        l = edges[i].faceMap[NN - 2];
        r = edges[i].faceMap[NN - 1];
        for (int k = 0; k < var; k++){
            if(l < incell){
                g = grad[l][k * dim + 0] * edges[i].edgerL[0] + grad[l][k * dim + 1] * edges[i].edgerL[1] + grad[l][k * dim + 2] * edges[i].edgerL[2];
                
                
                qfl[i][k] = qf[l][k] + limiter * g;
            }else if(l < ncell){
                qfl[i][k] = 0;
            }
            if(r < incell){
                g = grad[r][k * dim + 0] * edges[i].edgerR[0] + grad[r][k * dim + 1] * edges[i].edgerR[1] + grad[r][k * dim + 2] * edges[i].edgerR[2];
                
                
                qfr[i][k] = qf[r][k] + limiter * g;
            }else if(r < ncell){
                qfr[i][k] = 0;
            }
        }
    }

    return true;    
}

bool MacroSolver::limiter2(bool gsisTag)
{
    int ncell = this->Ncell, incell = this->iNcell, var = this->var;
    int nface = this->Nface, inface = this->iNface, enface = this->eNface;
    int dim = DIM;
    double **qfl = this->qfl, **qfr = this->qfr, **qf = this->Qf;
    double **grad = this->Grad, **um = this->Umaxmin;

    if(gsisTag){
        nface = enface;
    }

    for (int i = 0; i < incell; i++){
        for(int j = 0; j < var; j++){
            um[i][j * 2 + 0] = qf[i][j];
            um[i][j * 2 + 1] = qf[i][j];
        }
    }

    int l, r;

    for (int i = 0; i < enface; i++){
        l = edges[i].faceMap[NN - 2];
        r = edges[i].faceMap[NN - 1];
        if (r >= ncell){
            continue;
        }
        for (int k = 0; k < var; k++){
            if(l < incell){
                um[l][k * 2 + 0] = max(um[l][k * 2 + 0], qf[r][k]);
                um[l][k * 2 + 1] = min(um[l][k * 2 + 1], qf[r][k]);
            }
            if(r < incell){
                um[r][k * 2 + 0] = max(um[r][k * 2 + 0], qf[l][k]);
                um[r][k * 2 + 1] = min(um[r][k * 2 + 1], qf[l][k]);
            }
        }
    }

    double g, uax, uin, limiter = 1.0;
    double *ls = new double[incell * var];
    for (int i = 0; i < incell * var; i++){
        ls[i] = 1.0;
    }
    int fi = 0, fsign = 0;
    for(int i = 0; i < nface; i++){
        l = edges[i].faceMap[NN - 2];
        r = edges[i].faceMap[NN - 1];
        for (int k = 0; k < var; k++){
            if(l < incell){
                g = grad[l][k * dim + 0] * edges[i].edgerL[0] + grad[l][k * dim + 1] * edges[i].edgerL[1] + grad[l][k * dim + 2] * edges[i].edgerL[2];
                limiter = this->Wang(um[l][k * 2 + 0], um[l][k * 2 + 1], qf[l][k], g);
                
                
                ls[l * var + k] = min(ls[l * var + k], limiter);
            }

            if(r < incell){
                g = grad[r][k * dim + 0] * edges[i].edgerR[0] + grad[r][k * dim + 1] * edges[i].edgerR[1] + grad[r][k * dim + 2] * edges[i].edgerR[2];
                limiter = this->Wang(um[r][k * 2 + 0], um[r][k * 2 + 1], qf[r][k], g);
                
                
                ls[r * var + k] = min(ls[r * var + k], limiter);
            }
        }
    }

    for (int i = 0; i < nface; i++){
        l = edges[i].faceMap[NN - 2];
        r = edges[i].faceMap[NN - 1];

        for (int k = 0; k < var; k++){
            if(l < incell){
                g = grad[l][k * dim + 0] * edges[i].edgerL[0] + grad[l][k * dim + 1] * edges[i].edgerL[1] + grad[l][k * dim + 2] * edges[i].edgerL[2];
                qfl[i][k] = qf[l][k] + ls[l * var + k] * g;
            }else if(l < ncell){
                qfl[i][k] = 0;
            }
            if(r < incell){
                g = grad[r][k * dim + 0] * edges[i].edgerR[0] + grad[r][k * dim + 1] * edges[i].edgerR[1] + grad[r][k * dim + 2] * edges[i].edgerR[2];
                qfr[i][k] = qf[r][k] + ls[r * var + k] * g;
            }else if(r < ncell){
                qfr[i][k] = 0;
            }
        }
    }

    delete[] ls;
    return true;    
}


void MacroSolver::cellDeepCopy(cell &c1, const cell &c2)
{
    c1.num = c2.num;
    c1.dim = c2.dim;
    for(int i=0; i<NV; i++){
        c1.cell2node[i] = c2.cell2node[i];
    }
    for(int i=0; i<NN; i++){
        c1.cell2cell[i] = c2.cell2cell[i];
        c1.cell2face[i] = c2.cell2face[i];
        c1.cell2face_sgn[i] = c2.cell2face_sgn[i];
    }
    c1.area = c2.area;
    c1.cellLengthEff = c2.cellLengthEff;
    for(int i=0; i<DIM; i++){
        for(int j=0; j<DIM; j++){
            c1.Ainv[i][j] = c2.Ainv[i][j];
        }
    }
    for(int i=0; i<NN; i++){
        for(int j=0; j<DIM; j++){
            c1.dxyz[i][j] = c2.dxyz[i][j];
        }
    }
    c1.cellType = c2.cellType;
    c1.rawCellType = c2.rawCellType;
    c1.no = c2.no;
}


void MacroSolver::edgeDeepCopy(edge &e1, const edge &e2)
{
    e1.no = e2.no;
    e1.faceTag = e2.faceTag;
    e1.dim = e2.dim;
    e1.faceType = e2.faceType;
    e1.bcType = e2.bcType;
    e1.length = e2.length;
    for(int i=0; i<NN; i++){
        e1.faceMap[i] = e2.faceMap[i];
    }
    for(int i=0; i<DIM; i++){
        e1.edgeCenter[i] = e2.edgeCenter[i];
        e1.edgeNormal[i] = e2.edgeNormal[i];
        
        e1.edgerij[i] = e2.edgerij[i];
        e1.edgerL[i] = e2.edgerL[i];
        e1.edgerR[i] = e2.edgerR[i];
    }
    e1.edgeDist = e2.edgeDist;
}

bool MacroSolver::numpyDeepCopy2D(int m, int n, double **A, double **B)
{
    for(int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            A[i][j] = B[i][j];
        }
    }
    return true;
}

bool MacroSolver::out2dat(const char *filename)
{
    int ncell = mess.Ncell, var = this->var;
    double *w = new double[ncell * var];
    for(int i=0, lb; i<ncell; i++){
        lb = i * var;
        for(int j=0; j<var; j++){
            if(i >= Nl && i < Nr){
                w[lb + j] = Qf[i - Nl][j];
                continue;
            }
            w[lb + j] = 0;
        }
        
    }

    MPI_Allreduce(MPI_IN_PLACE, w, ncell * var, MPI_DOUBLE, MPI_SUM, this->myGroup);

    if(activeLeader()){
        
        this->mesh->out2Fluent_ns2(filename, w);
    }
    
    delete[] w;
    return true;
}

bool MacroSolver::nsout2dat(int istep, double **Q,int startcell, int endcell)
{
    int ncell = this->mess.Ncell, var = this->mess.var;
    double *w = new double[ncell * var];

    if(active())
    {
        for(int i=0, lb; i<ncell; i++)
        {
            lb = i * var;
            for(int j=0; j < var; j++){
                if(i >= startcell && i < endcell)
                {
                    w[lb + j] = Q[i - startcell][j];
                    continue;
                }
                w[lb + j] = 0;
            }
        }
    }
    else
    {
        for(int j=0; j < var*ncell; j++)
        {
            w[j] = 0;
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, w, ncell * var, MPI_DOUBLE, MPI_SUM, comm);

    if (root())
    {   
        double *q = new double[ncell * var];

        for(int i=0; i<ncell; i++)
        {   
            for(int j=0; j < var; j++)
            {
                q[this->NsreMeIndex2[i]*var + j] = w[i*var + j];
            }
        }
        
        char filename[100];
        sprintf(filename, "./nsTemp/ns_Kn%.3f_iter%d.dat", mess.Kn, istep);
        if (this->mesh == NULL)
        {
            return false;
        }

        this->mesh->out2Fluent_ns6(filename, q);
        delete[] q;
    }

    delete[] w;
    return true;
}

double MacroSolver::calcMaxError(double **wt, double **qf)
{
    double rho = this->calcError(0, wt, qf);
    double ux = this->calcError(3, wt, qf);
    double ttra = this->calcError(4, wt, qf);


    rho = max(rho, ux);
    rho = max(rho, ttra);

    return rho;
}

double MacroSolver::calcError(int i, double **wt, double **qf)
{
    int ncell = this->Ncell, incell = this->iNcell, var = this->var;
    double err, A[2];
    A[0] = 0; A[1] = 0;
    
    
    
    
    
    
    for(int j=0; j<incell; j++){
        A[0] += pow(qf[j][i] - wt[j][i], 2);
        A[1] += pow(wt[j][i], 2);
    }

    MPI_Allreduce(MPI_IN_PLACE, A, 2, MPI_DOUBLE, MPI_SUM, this->myGroup);

    if (A[1] <= 1e-8){
        return sqrt(A[0]);
    }
    err = sqrt(A[0] / A[1]);

    return err;
}
void MacroSolver::calcCellHot(int istep)
{
    int ncell = this->Ncell, incell = this->iNcell, var = this->var;
    int nface = this->Nface, inface = this->iNface, enface = this->eNface;
    int size = this->iSize;

    this->packQfQc();
    MPI_Startall(2 * size, &this->request[0]);

    MPI_Waitall(2 * size, &this->request[0], &this->status[0]);
    this->unPackQfQc();

    this->leastSquareGrad();

    double omega = mess.omega, dr = mess.dr, delta_rp = mess.delta_rp;
    double **grad = this->Grad, **nss = this->d_sigmaq;
    double kappat = 3  * 0.5 * mess.f_tra;
    double kappar = dr * 0.5 * mess.f_rot;
    double **qf = this->Qf;
    double u, v, ttra, mu;
    double ux, uy, uz, vx, vy, vz, wx, wy, wz, tx, ty, tz, rx, ry, rz;
    double txx, txy, txz, tyx, tyy, tyz, tzx, tzy, tzz;
    double qtx, qty, qtz, qrx, qry, qrz, div;
    int l, r, dim = DIM;
    double highOrderWeight = 1.0;
    if (EnableHighOrderRamp && NHIGH_ORDER_RAMP_END > NHIGH_ORDER_RAMP_START)
    {
        if (istep <= NHIGH_ORDER_RAMP_START)
        {
            highOrderWeight = HIGH_ORDER_RAMP_MIN;
        }
        else if (istep < NHIGH_ORDER_RAMP_END)
        {
            const double x = static_cast<double>(istep - NHIGH_ORDER_RAMP_START) /
                             static_cast<double>(NHIGH_ORDER_RAMP_END - NHIGH_ORDER_RAMP_START);
            highOrderWeight = HIGH_ORDER_RAMP_MIN + (1.0 - HIGH_ORDER_RAMP_MIN) * x;
        }
    }

    for (int i = 0; i < incell; i++){
        ttra = qf[i][4];
        mu = pow(fabs(ttra), omega) / delta_rp;
        
        ux = grad[i][1 * dim + 0]; uy = grad[i][1 * dim + 1]; uz = grad[i][1 * dim + 2];
        vx = grad[i][2 * dim + 0]; vy = grad[i][2 * dim + 1]; vz = grad[i][2 * dim + 2];
        wx = grad[i][3 * dim + 0]; wy = grad[i][3 * dim + 1]; wz = grad[i][3 * dim + 2];
        tx = grad[i][4 * dim + 0]; ty = grad[i][4 * dim + 1]; tz = grad[i][4 * dim + 2];
        rx = grad[i][5 * dim + 0]; ry = grad[i][5 * dim + 1]; rz = grad[i][5 * dim + 2];
        div = (ux + vy + wz) / 3.0;

        txx = 2 * mu * (ux - div);
        tyy = 2 * mu * (vy - div);
        tzz = 2 * mu * (wz - div);
        txy = tyx = mu * (uy + vx);
        txz = tzx = mu * (uz + wx);
        tyz = tzy = mu * (vz + wy);


        qtx = kappat * mu * tx; qty = kappat * mu * ty; qtz = kappat * mu * tz;
        qrx = kappar * mu * rx; qry = kappar * mu * ry; qrz = kappar * mu * rz;

        nss[i][0] -= qtx; nss[i][1] -= qty; nss[i][2] -= qtz;
        nss[i][3] -= qrx; nss[i][4] -= qry; nss[i][5] -= qrz;
        nss[i][6] -= txx; nss[i][7] -= txy; nss[i][8] -= txz;
        nss[i][9] -= tyy; nss[i][10] -= tyz; nss[i][11] -= tzz;

        for (int j = 0; j < VAR2; j++)
        {
            nss[i][j] *= highOrderWeight;
        }

        
        
        
        
    }
}
double MacroSolver::Wang(const double &umax, const double &umin, const double &u, const double &g)
{
    double uax = umax - u, uin = umin - u, ep = 0.2;
    double limiter = 1.0, e2 = ep * (umax - umin), du = 0.0;

    if(fabs(g) < 1e-16){
        return limiter;
    }else if(g > 0){
        du = uax;
    }else{
        du = uin;
    }
    
    limiter = (du * du + 2 * du * g + e2) / (du * du + 2 * g * g + du * g + e2);

    return limiter;
}

void MacroSolver::calcG13Sigma(int l, double *gm, double *ds, double *sq)
{
    
    
    
    
    
    
    

    
    
    
    
    
    

    for(int i=0; i<VAR2; i++){
        sq[i] = -(ns_sigmaq[l][i] + d_sigmaq[l][i]);
        
        
    }

}

bool MacroSolver::Flux_NSEG13_bcWallWithT(double T, vector<int> wallVis)
{
    edge *edges = this->edges;
    int l = 0, r= 0, v = 0;
    double Frho, Frhou, Frhov, Frhow, FrhoE, FrhoEr;
    
    double *IFvl = new double[21], *IFvr = new double[21], *a = new double[2];
    double lam_tt, length, *norm = NULL; 
    int dim = DIM;
    double *gm = new double[var * dim];
    double *ds = new double[VAR2];
    double *wrw = new double[VAR], *sqrw = new double[VAR2], *sqlw = new double[VAR2];
    double *A = new double[8], localuw;
    bool left = true, right = false;

    double rho = 1, ux = 0, uy = 0, uz = 0, t = T;
    double x1, y1, z1, x2, y2, z2, x3, y3, z3;

    for(int i=0, index; i<wallVis.size(); i++){
        v = wallVis[i];
        index = (v - eNface) * 2 * DIM;
        l = edges[v].faceMap[NN - 2];
        length = edges[v].length;
        norm = edges[v].edgeNormal;

        this->calcG13Sigma(l, gm, ds, sqlw);          
    
        
        this->calcFlux_IFV2(Qf[l], sqlw, norm, IFvl, A, left, index);

        wrw[1] = ux; wrw[2] = uy; wrw[3] = uz; wrw[4] = t; wrw[5] = t;
        sqrw[0] = 0; sqrw[1] = 0; sqrw[2] = 0; sqrw[3] = 0; sqrw[4] = 0; sqrw[5] = 0; sqrw[6] = 0; sqrw[7] = 0;
        sqrw[8] = 0; sqrw[9] = 0; sqrw[10] = 0; sqrw[11] = 0;

        lam_tt = 1.0 / (2.0 * wrw[4]);
        localuw = wrw[1] * norm[0] + wrw[2] * norm[1] + wrw[3] * norm[2];
        a[0] = 0.5 * erfc(sqrt(lam_tt) * localuw);
        a[1] = localuw * a[0] - 0.50 * exp(-lam_tt * localuw * localuw) / sqrt(M_PI * lam_tt);
        wrw[0] = (-IFvl[0]) / a[1];

        this->calcFlux_IFV2(wrw, sqrw, norm, IFvr, A, right, index);

        x1 = norm[0], y1 = norm[1], z1 = norm[2];
        x2 = tVector[index + 0], y2 = tVector[index + 1], z2 = tVector[index + 2];
        x3 = tVector[index + 3], y3 = tVector[index + 4], z3 = tVector[index + 5];

        Frho = IFvl[0] + IFvr[0];
        Frhou = (IFvl[1] + IFvr[1]) * x1 + (IFvl[2] + IFvr[2]) * x2 + (IFvl[3] + IFvr[3]) * x3;
        Frhov = (IFvl[1] + IFvr[1]) * y1 + (IFvl[2] + IFvr[2]) * y2 + (IFvl[3] + IFvr[3]) * y3;
        Frhow = (IFvl[1] + IFvr[1]) * z1 + (IFvl[2] + IFvr[2]) * z2 + (IFvl[3] + IFvr[3]) * z3;
        FrhoE = IFvl[4] + IFvr[4];
        FrhoEr = IFvl[5] + IFvr[5];

        RHS[l][0] -= Frho * length;
        RHS[l][1] -= Frhou * length;
        RHS[l][2] -= Frhov * length;
        RHS[l][3] -= Frhow * length;
        RHS[l][4] -= FrhoE * length;
        RHS[l][5] -= FrhoEr * length;                                                                  
    }

    delete[] a; delete[] wrw; delete[] sqlw; delete[] sqrw;
    delete[] IFvl; delete[] IFvr; 
    delete[] gm; delete[] ds; delete[] A;
    return true;
}

bool MacroSolver::Flux_NSEG13_inlet(vector<int> wallVis)
{
    edge *edges = this->edges;
    int l = 0, r= 0, v = 0;
    double Frho, Frhou, Frhov, Frhow, FrhoE, FrhoEr;
    
    double *IFvl = new double[21], *IFvr = new double[21], *a = new double[2];
    double lam_tt, length, *norm = NULL, Ma = mess.Ma*sqrt(mess.gamma);
    int dim = DIM;
    double *gm = new double[var * dim];
    double *ds = new double[VAR2];
    double *wrw = new double[VAR], *sqrw = new double[VAR2], *sqlw = new double[VAR2];
    double *A = new double[8];
    bool left = true, right = false;

    double theta = 30;
    
    
    double rho = 1, ux = Ma * this->mess.v_in/this->mess.v_rms *sqrt(3) /2 , uy = Ma * this->mess.v_in/this->mess.v_rms /2 , uz = 0, t = this->mess.T_in/this->mess.T_ref;
    double x1, y1, z1, x2, y2, z2, x3, y3, z3;
    
    for(int i=0, index; i<wallVis.size(); i++){
        v = wallVis[i];
        index = (v - eNface) * 2 * DIM;
        l = edges[v].faceMap[NN - 2];
        length = edges[v].length;
        norm = edges[v].edgeNormal;  

        this->calcG13Sigma(l, gm, ds, sqlw);           

        this->calcFlux_IFV2(Qf[l], sqlw, norm, IFvl, A, left, index); 

        wrw[0] = rho; wrw[1] = ux; wrw[2] = uy; wrw[3] = uz; wrw[4] = t; wrw[5] = t; 
        sqrw[0] = 0; sqrw[1] = 0; sqrw[2] = 0; sqrw[3] = 0; sqrw[4] = 0; sqrw[5] = 0; sqrw[6] = 0; sqrw[7] = 0;
        sqrw[8] = 0; sqrw[9] = 0; sqrw[10] = 0; sqrw[11] = 0;

        this->calcFlux_IFV2(wrw, sqrw, norm, IFvr, A, right, index);

        x1 = norm[0], y1 = norm[1], z1 = norm[2];
        x2 = tVector[index + 0], y2 = tVector[index + 1], z2 = tVector[index + 2];
        x3 = tVector[index + 3], y3 = tVector[index + 4], z3 = tVector[index + 5];

        Frho = IFvl[0] + IFvr[0];
        Frhou = (IFvl[1] + IFvr[1]) * x1 + (IFvl[2] + IFvr[2]) * x2 + (IFvl[3] + IFvr[3]) * x3;
        Frhov = (IFvl[1] + IFvr[1]) * y1 + (IFvl[2] + IFvr[2]) * y2 + (IFvl[3] + IFvr[3]) * y3;
        Frhow = (IFvl[1] + IFvr[1]) * z1 + (IFvl[2] + IFvr[2]) * z2 + (IFvl[3] + IFvr[3]) * z3;
        FrhoE = IFvl[4] + IFvr[4];
        FrhoEr = IFvl[5] + IFvr[5];

        RHS[l][0] -= Frho * length;
        RHS[l][1] -= Frhou * length;
        RHS[l][2] -= Frhov * length;
        RHS[l][3] -= Frhow * length;
        RHS[l][4] -= FrhoE * length;
        RHS[l][5] -= FrhoEr * length;                                                                   
    }

    delete[] a; delete[] wrw; delete[] sqlw; delete[] sqrw;
    delete[] IFvl; delete[] IFvr; 
    delete[] gm; delete[] ds; delete[] A;
    return true;
}

bool MacroSolver::Flux_NSEG13_outlet(vector<int> wallVis)
{
    edge *edges = this->edges;
    int l = 0, r= 0, v = 0;
    double Frho, Frhou, Frhov, Frhow, FrhoE, FrhoEr;
    double *IFvl = new double[21], *IFvr = new double[21], *a = new double[2];
    double lam_tt, length, *norm = NULL, Ma = mess.Ma*sqrt(mess.gamma);
    int dim = DIM;
    double *gm = new double[var * dim];
    double *ds = new double[VAR2];
    double *wrw = new double[VAR], *sqrw = new double[VAR2], *sqlw = new double[VAR2];
    double *A = new double[8];
    bool left = true, right = false;

    double x1, y1, z1, x2, y2, z2, x3, y3, z3;

    for(int i=0, index; i<wallVis.size(); i++){
        v = wallVis[i];
        index = (v - eNface) * 2 * DIM;
        l = edges[v].faceMap[NN - 2];
        length = edges[v].length;
        norm = edges[v].edgeNormal;  

        this->calcG13Sigma(l, gm, ds, sqlw);           

        
        this->calcFlux_IFV2(Qf[l], sqlw, norm, IFvl, A, left, index);

        wrw[0] = Qf[l][0]; wrw[1] = Qf[l][1]; wrw[2] = Qf[l][2]; wrw[3] = Qf[l][3]; wrw[4] = Qf[l][4]; wrw[5] = Qf[l][5];
        sqrw[0] = 0; sqrw[1] = 0; sqrw[2] = 0; sqrw[3] = 0; sqrw[4] = 0; sqrw[5] = 0; sqrw[6] = 0; sqrw[7] = 0;
        sqrw[8] = 0; sqrw[9] = 0; sqrw[10] = 0; sqrw[11] = 0;


        
        this->calcFlux_IFV2(wrw, sqlw, norm, IFvr, A, right, index);


        
        

        x1 = norm[0], y1 = norm[1], z1 = norm[2];
        x2 = tVector[index + 0], y2 = tVector[index + 1], z2 = tVector[index + 2];
        x3 = tVector[index + 3], y3 = tVector[index + 4], z3 = tVector[index + 5];

        Frho = IFvl[0] + IFvr[0];
        Frhou = (IFvl[1] + IFvr[1]) * x1 + (IFvl[2] + IFvr[2]) * x2 + (IFvl[3] + IFvr[3]) * x3;
        Frhov = (IFvl[1] + IFvr[1]) * y1 + (IFvl[2] + IFvr[2]) * y2 + (IFvl[3] + IFvr[3]) * y3;
        Frhow = (IFvl[1] + IFvr[1]) * z1 + (IFvl[2] + IFvr[2]) * z2 + (IFvl[3] + IFvr[3]) * z3;
        FrhoE = IFvl[4] + IFvr[4];
        FrhoEr = IFvl[5] + IFvr[5];

        RHS[l][0] -= Frho * length;
        RHS[l][1] -= Frhou * length;
        RHS[l][2] -= Frhov * length;
        RHS[l][3] -= Frhow * length;
        RHS[l][4] -= FrhoE * length;
        RHS[l][5] -= FrhoEr * length;                                                     
    }

    delete[] a; delete[] wrw; delete[] sqlw; delete[] sqrw;
    delete[] IFvl; delete[] IFvr; 
    delete[] gm; delete[] ds; delete[] A;
    return true;
}
bool MacroSolver::calcFlux_IFV2(double *WI, double *sq, double *enormal, double *IFv, double *a, bool tag, int index)
{
    if(this->tVector == NULL){
        this->CalculateTangentialVector();
    }
    double x1 = enormal[0], y1 = enormal[1], z1 = enormal[2];
    double x2 = tVector[index + 0], y2 = tVector[index + 1], z2 = tVector[index + 2];
    double x3 = tVector[index + 3], y3 = tVector[index + 4], z3 = tVector[index + 5];
    
    double Den = WI[0], U = WI[1] * x1 + WI[2] * y1 + WI[3] * z1, V = WI[1] * x2 + WI[2] * y2 + WI[3] * z2;
    double W = WI[1] * x3 + WI[2] * y3 + WI[3] * z3, T_tt = WI[4], T_tr = WI[5], pi = M_PI;
    double P_tt = Den * T_tt, lam_tt = 1.0 / (2.0 * T_tt);

    double qtx = sq[0], qty = sq[1], qtz = sq[2],  qrx = sq[3], qry = sq[4], qrz = sq[5];
    double sxx = sq[6], sxy = sq[7], sxz = sq[8], syy = sq[9], syz = sq[10];
    double szz = sq[11], syx = sxy, szx = sxz, szy = syz;   

    double dr = mess.dr, temp = 1.0 /  (P_tt * T_tt);

    double t11 = (x1 * x1 * sxx + x1 * y1 * sxy + x1 * z1 * sxz + y1 * x1 * syx + y1 * y1 * syy + y1 * z1 * syz + z1 * x1 * szx + y1 * z1 * syz + z1 * z1 * szz) * 0.5 * temp;
    double t12 = (x1 * x2 * sxx + x1 * y2 * sxy + x1 * z2 * sxz + y1 * x2 * syx + y1 * y2 * syy + y1 * z2 * syz + z1 * x2 * szx + z1 * y2 * szy + z1 * z2 * szz) * 0.5 * temp;
    double t13 = (x1 * x3 * sxx + x1 * y3 * sxy + x1 * z3 * sxz + y1 * x3 * syx + y1 * y3 * syy + y1 * z3 * syz + z1 * x3 * szx + z1 * y3 * szy + z1 * z3 * szz) * 0.5 * temp;
    double t22 = (x2 * x2 * sxx + x2 * y2 * sxy + x2 * z2 * sxz + y2 * x2 * syx + y2 * y2 * syy + y2 * z2 * syz + z2 * x2 * szx + z2 * y2 * szy + z2 * z2 * szz) * 0.5 * temp;
    double t23 = (x2 * x3 * sxx + x2 * y3 * sxy + x2 * z3 * sxz + y2 * x3 * syx + y2 * y3 * syy + y2 * z3 * syz + z2 * x3 * szx + z2 * y3 * szy + z2 * z3 * szz) * 0.5 * temp;
    double t33 = 0.0 - t11 - t22, t21 = t12, t31 = t13, t32 = t23;

    double q1 = (x1 * qtx + y1 * qty + z1 * qtz) * temp;
    double q2 = (x2 * qtx + y2 * qty + z2 * qtz) * temp;
    double q3 = (x3 * qtx + y3 * qty + z3 * qtz) * temp;

    double qr1 = (x1 * qrx + y1 * qry + z1 * qrz) / (P_tt);
    double qr2 = (x2 * qrx + y2 * qry + z2 * qrz) / (P_tt);
    double qr3 = (x3 * qrx + y3 * qry + z3 * qrz) / (P_tt);

    if(tag){
        a[0] = 0.5 * (1.0 + erf(sqrt(lam_tt) * U));
        a[1] = U * a[0] + 0.5 * exp(-lam_tt * U * U) / sqrt(pi * lam_tt);
    }else{
        a[0] = 0.5 * erfc(sqrt(lam_tt) * U);
        a[1] = U * a[0] - 0.5 * exp(-lam_tt * U * U) / sqrt(pi * lam_tt);
    }
    

    for (int k = 0; k < 6; k++){
        a[k + 2] = U * a[k + 1] + (k + 1.0) / (2.0 * lam_tt) * a[k];
    }

    for(int i=0; i<5; i++){
        Cx[i][0] = a[i];
        Cx[i][1] = a[i + 1] - U * a[i];
        Cx[i][2] = a[i + 2] - 2 * U * a[i + 1] + U * U * a[i];
        Cx[i][3] = a[i + 3] - 3 * U * a[i + 2] + 3 * U * U * a[i + 1] - U * U * U * a[i];
    }

    Cy[0] = 1.0;
    Cy[1] = 0.0;
    Cy[2] = T_tt;
    Cy[3] = 0.0;
    Cy[4] = 3.0 * T_tt * T_tt;
    Cy[5] = 0.0;
    Cy[6] = 15.0 * T_tt * T_tt * T_tt;
    Cy[7] = 0.0;
    

    Cz[0] = 1.0;
    Cz[1] = 0.0;
    Cz[2] = T_tt;
    Cz[3] = 0.0;
    Cz[4] = 3.0 * T_tt * T_tt;
    Cz[5] = 0.0;
    Cz[6] = 15.0 * T_tt * T_tt * T_tt;
    Cz[7] = 0.0;
    

    for(int i=0; i<4; i++){
        for(int j=0; j<3; j++){
            for(int k=0; k<3; k++){
                qxyz[i][j][k] = Cx[i][0] * Cy[j] * Cz[k] +  t11 * Cx[i][2] * Cy[j] * Cz[k] + 2.0 * t12 * Cx[i][1] * Cy[j + 1] * Cz[k] + 2.0 * t13 * Cx[i][1] * Cy[j] * Cz[k + 1];
                qxyz[i][j][k] += t22 * Cx[i][0] * Cy[j + 2] * Cz[k] + 2.0 * t23 * Cx[i][0] * Cy[j + 1] * Cz[k + 1] + t33 * Cx[i][0] * Cy[j] * Cz[k + 2];
                qxyz[i][j][k] += -q1 * Cx[i][1] * Cy[j] * Cz[k] - q2 * Cx[i][0] * Cy[j + 1] * Cz[k] - q3 * Cx[i][0] * Cy[j] * Cz[k + 1];
                qxyz[i][j][k] += 0.2 * q1 / T_tt * (Cx[i][3] * Cy[j] * Cz[k] + Cx[i][1] * Cy[j + 2] * Cz[k] + Cx[i][1] * Cy[j] * Cz[k + 2]);
                qxyz[i][j][k] += 0.2 * q2 / T_tt * (Cx[i][2] * Cy[j + 1] * Cz[k] + Cx[i][0] * Cy[j + 3] * Cz[k] + Cx[i][0] * Cy[j + 1] * Cz[k + 2]);
                qxyz[i][j][k] += 0.2 * q3 / T_tt * (Cx[i][1] * Cy[j] * Cz[k + 1] + Cx[i][0] * Cy[j + 2] * Cz[k + 1] + Cx[i][0] * Cy[j] * Cz[k + 3]);
            } 
        } 
    }    

    for(int i=0; i<4; i++){
        for(int k=0; k<3; k++){
            mxyz[i][0][k] = (qxyz[i][0][k]);
            mxyz[i][1][k] = (qxyz[i][1][k] + V * qxyz[i][0][k]);
            mxyz[i][2][k] = (qxyz[i][2][k] + 2.0 * V * qxyz[i][1][k] + V * V * qxyz[i][0][k]);
        } 
    }

    for(int i=0; i<4; i++){
        for(int j=0; j<3; j++){
            txyz[i][j][0] = (mxyz[i][j][0]);
            txyz[i][j][1] = (mxyz[i][j][1] + W * mxyz[i][j][0]);
            txyz[i][j][2] = (mxyz[i][j][2] + 2.0 * W * mxyz[i][j][1] + W * W * mxyz[i][j][0]);
        } 
    }

    for(int i=0; i<4; i++){
        for(int j=0; j<3; j++){
            for(int k=0; k<3; k++){
                wxyz[i][j][k] = qr1 * Cx[i][1] * Cy[j] * Cz[k] + qr2 * Cx[i][0] * Cy[j + 1] * Cz[k] + qr3 * Cx[i][0] * Cy[j] * Cz[k + 1];
            }
        }
    }

    for(int i=0; i<4; i++){
        for(int k=0; k<3; k++){
            nxyz[i][0][k] = (wxyz[i][0][k]);
            nxyz[i][1][k] = (wxyz[i][1][k] + V * wxyz[i][0][k]);
            nxyz[i][2][k] = (wxyz[i][2][k] + 2.0 * V * wxyz[i][1][k] + V * V * wxyz[i][0][k]);
        } 
    }

    for(int i=0; i<4; i++){
        for(int j=0; j<3; j++){
            rxyz[i][j][0] = (nxyz[i][j][0]);
            rxyz[i][j][1] = (nxyz[i][j][1] + W * nxyz[i][j][0]);
            rxyz[i][j][2] = (nxyz[i][j][2] + 2.0 * W * nxyz[i][j][1] + W * W * nxyz[i][j][0]);
        }
    }

    for(int i=0; i<4; i++){
        for(int j=0; j<3; j++){
            for(int k=0; k<3; k++){
                txyz[i][j][k] *= Den;
                rxyz[i][j][k] *= Den;
            }
        }
    }

    double weight = 1.0;

    IFv[0] = txyz[1][0][0] * weight;
    IFv[1] = txyz[2][0][0] * weight;
    IFv[2] = txyz[1][1][0] * weight;
    IFv[3] = txyz[1][0][1] * weight;
    IFv[4] = (0.5 * (txyz[3][0][0] + txyz[1][2][0] + txyz[1][0][2]) + 0.5 * dr * T_tr * txyz[1][0][0] + rxyz[1][0][0]) * weight;
    IFv[5] = (0.5 * dr * T_tr * txyz[1][0][0] + rxyz[1][0][0]) * weight;

    return true;
}

void MacroSolver::CalculateTangentialVector()
{
    int ncell = this->Ncell, incell = this->iNcell, var = this->var;
    int nface = this->Nface, inface = this->iNface, enface = this->eNface;

    if(this->tVector == NULL){
        this->tVector = new double[(nface - enface) * 2 * DIM];
    }else{
        return;
    }

    double x1, y1, z1, x2, y2, z2, x3, y3, z3, c;
    for(int i=enface, v; i<nface; i++){
        v = (i - enface) * 2 * DIM;
        x1 = edges[i].edgeNormal[0]; y1 = edges[i].edgeNormal[1]; z1 = edges[i].edgeNormal[2];
        x2 = 1.0; y2 = 2.0;
        if(fabs(z1) < 1e-8){
            if(fabs(y1) < 1e-8){
                y2 = 1; z2 = 2;
                x2 = (0 - y1 * y2 - z1 * z2) / x1;
            }else{
                x2 = 1.0; z2 = 2.0;
                y2 = (0 - x1 * x2 - z1 * z2) / y1;
            }
            
        }else{
            z2 = (0 - x1 * x2 - y1 * y2) / z1;
        }
        
        c = sqrt(x2 * x2 + y2 * y2 + z2 * z2);
        x2 /= c; y2 /= c; z2 /= c;
        x3 = y2 * z1 - z2 * y1;
        y3 = z2 * x1 - z1 * x2;
        z3 = x2 * y1 - x1 * y2;

        if(fabs(x1 * x2 + y1 * y2 + z1 * z2) > 1e-6 || fabs(x1 * x3 + y1 * y3 + z1 * z3) > 1e-6 || fabs(x3 * x2 + y3 * y2 + z3 * z2) > 1e-6){
            printf("Warning: tVector basis is not orthogonal\n");
        }
        if(fabs(x1 * x1 + y1 * y1 + z1 * z1 - 1) > 1e-8 || fabs(x2 * x2 + y2 * y2 + z2 * z2 - 1) > 1e-8 || fabs(x3 * x3 + y3 * y3 + z3 * z3 - 1) > 1e-8){
            printf("Warning: tVector basis is not normalized\n");
        }


        tVector[v + 0] = x2; tVector[v + 1] = y2; tVector[v + 2] = z2;
        tVector[v + 3] = x3; tVector[v + 4] = y3; tVector[v + 5] = z3;
    }
}
