#include "meshImport.h"
#include "MacroSolver.h"
#include "ProcessDSMC.h"
#include "ProcessGSIS.h"
#include "MeshparticalInitial.h"
#include "MessagePassing.h"
#include "dynamicDSMC.h"
#include "MpiContext.h"
#include <iomanip>







void statistic_Result(int istep, ProcessDSMC* process);
void ns_result(int istep, MacroSolver *nsprocess);
double maxError = 1e-6;

static void reportStepTimingMax(const MpiContext& mpiCtx,
                                int istep,
                                double& duration_preprocess,
                                double& duration_advection,
                                double& duration_cache2chain,
                                double& duration_collision,
                                double& duration_statisticmacro)
{
    if (istep % NDSIPLAY != 0)
        return;

    double localTime[5] = {
        duration_preprocess,
        duration_advection,
        duration_cache2chain,
        duration_collision,
        duration_statisticmacro
    };
    double maxTime[5] = {0.0, 0.0, 0.0, 0.0, 0.0};

    if (mpiCtx.active())
    {
        MPI_Reduce(localTime, maxTime, 5, MPI_DOUBLE, MPI_MAX, 0, mpiCtx.calGroup);

        if (mpiCtx.activeLeader())
        {
            cout << "Finished " << istep << " of " << NTOTAL << " steps." << endl;
            cout << "Preprocess: " << std::setprecision(4) << std::fixed << maxTime[0];
            cout << "   Advection: "<< std::setprecision(4) << std::fixed << maxTime[1];
            cout << "   cache2chain: "<< std::setprecision(4) << std::fixed << maxTime[2];
            cout << "   Collision: "<< std::setprecision(4) << std::fixed << maxTime[3];
            cout << "   Statistic_macro: "<< std::setprecision(4) << std::fixed << maxTime[4] << endl;
        }
    }

    duration_preprocess = 0.0;
    duration_advection = 0.0;
    duration_cache2chain = 0.0;
    duration_collision = 0.0;
    duration_statisticmacro = 0.0;
}

int main(int argc, char **argv)
{
    int rank, size;
    double startTime, endTime;
    
    
    MPI_Comm comm = MPI_COMM_WORLD;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    MPI_Comm compute_comm = MPI_COMM_NULL;
    const bool active = MpiContext::activeFromWorldRank(rank);
    int color = active ? 0 : MPI_UNDEFINED;  
    MPI_Comm_split(comm, color, rank, &compute_comm);

    MpiContext mpiCtx(comm, compute_comm);
    
    MessagePassing *mpass = new MessagePassing();
    meshImport *mesh = NULL;
    meshMessage mess;

    int c_rank = mpiCtx.c_rank; 
    int c_size = mpiCtx.c_size;
    int istep = 0;
    if (mpiCtx.root())
    {
        const char *filePath = (argc > 1) ? argv[1] : "./mesh/3dapollo372500.cas";
        mesh = new meshImport(filePath, size - 1, 1);  
        mesh->rank = rank; mesh->c_size = size - 1;
        mesh->setMa_Kn_CFL(5, 0.01, 1e2, 1e3, 5);
        mesh->setMeshMessage();
        mess = mesh->mess;
    }

    MPI_Datatype myMessage;
    mpass->commitMyMesssge(myMessage);

    MPI_Bcast(&mess, 1, myMessage, 0, comm);
    MPI_Type_free(&myMessage);

    dynamicDSMC *dyprocess = NULL;
    MacroSolver *nsprocess = NULL;
    MeshparticalInitial *partinit = NULL;
    ProcessDSMC *process = NULL;
    ProcessGSIS *progsis = NULL;

    nsprocess = new MacroSolver(mesh, mess,mpass,mpiCtx);
    partinit = new MeshparticalInitial(mesh, mess, mpass, mpiCtx);
    if (!partinit->partitionReady)
    {
        if (mpiCtx.root())
            cout << "MESH_PARTITION_INITIAL_ABORT" << endl;
        delete partinit;
        delete nsprocess;
        delete mesh;
        delete mpass;
        if (mpiCtx.active()) MPI_Comm_free(&compute_comm);
        MPI_Finalize();
        return 1;
    }
    process = new ProcessDSMC(mesh,mess,partinit,mpiCtx);
    dyprocess = new dynamicDSMC(mesh, mpass, mess, partinit, process,mpiCtx);
    progsis = new ProcessGSIS(mesh, process, partinit, nsprocess,mpiCtx);

    nsprocess->nsProcessG13(1e5, 1e-5, false,istep);
    progsis->NS2DSMC();
    progsis->molecular_velocity_change();
    process->statistic_macroPre();
    process->out2dat(istep);
    dyprocess->dynamic_rankload_distribute(c_size);
    progsis->regsisvariables();

    double preprocess_start, preprocess_end;
    double advection_start, advection_end;
    double cache2chain_start, cache2chain_end;
    double collision_start, collision_end;
    double statisticmacro_start, statisticmacro_end;
    double duration_preprocess=0.0, duration_advection=0.0, duration_cache2chain=0.0, duration_collision=0.0, duration_statisticmacro=0.0;
    double dynamic_start,dynamic_end;
    double dsmc_step_start, dsmc_step_end;
    MPI_Barrier(comm);
    if(mpiCtx.root()){cout << "DSMC_begin "<< endl;}
    startTime = MPI_Wtime();

    for (istep = 1; istep <= NTOTAL; istep++)
    {   
        if(istep == NSS)
        {
            dsmc_step_start = MPI_Wtime();
        }

        if (mpiCtx.active()) 
        {
            preprocess_start=MPI_Wtime();
            process->preprocesseffquad(istep); 
            preprocess_end=MPI_Wtime();  
            duration_preprocess += (preprocess_end - preprocess_start);
            MPI_Barrier(compute_comm);

            advection_start=MPI_Wtime(); 
            process->advection(istep);
            advection_end=MPI_Wtime(); 
            duration_advection += (advection_end - advection_start);

            cache2chain_start=MPI_Wtime();
            process->cache2chain();
            cache2chain_end=MPI_Wtime(); 
            duration_cache2chain += (cache2chain_end - cache2chain_start);

            collision_start=MPI_Wtime();   
            process->collisionDSMC();
            collision_end=MPI_Wtime(); 
            duration_collision += (collision_end - collision_start);

            preprocess_start=MPI_Wtime();
            process->statistic_macro(istep);
            preprocess_end=MPI_Wtime(); 
            duration_preprocess += (preprocess_end - preprocess_start);
        }

        MPI_Barrier(comm);
        statisticmacro_start=MPI_Wtime(); 
        statistic_Result(istep,process);
        statisticmacro_end=MPI_Wtime(); 
        duration_statisticmacro += (statisticmacro_end - statisticmacro_start);

        reportStepTimingMax(mpiCtx,
            istep,
            duration_preprocess,
            duration_advection,
            duration_cache2chain,
            duration_collision,
            duration_statisticmacro);
        
        if (ifgsis == 1 && istep%NGSIS==0 && istep > NSCHEME)
        {
            if (c_rank ==0) {cout << "The macroscopic equations are solving after " << istep << " DSMC steps" << endl;}

            if (compute_comm != MPI_COMM_NULL)
            {
                progsis->macro_iter_process(maxError,istep);
            }

            MPI_Barrier(comm);
            ns_result(istep, nsprocess);
            if (c_rank ==0) {cout << "The macroscopic acceleration is done after " << istep << " DSMC steps" << endl;}
        }

        if(istep < 1000)
        {
            if(istep%200== 0)
            {
                MPI_Barrier(comm);
                dynamic_start = MPI_Wtime();
                dyprocess->dynamic_rankload_distribute(c_size);
                progsis->regsisvariables();
                dynamic_end = MPI_Wtime();
                if(c_rank == 0)
                {
                    cout<< "dynamic_times   "<< dynamic_end - dynamic_start <<endl;
                }
            }
        }
        else
        {
            if(istep%1000 == 0)
            {
                MPI_Barrier(comm);
                dynamic_start = MPI_Wtime();
                dyprocess->dynamic_rankload_distribute(c_size);
                progsis->regsisvariables();
                dynamic_end = MPI_Wtime();
                if(c_rank == 0)
                {
                    cout<< "dynamic_times   "<< dynamic_end - dynamic_start <<endl;
                }
            }
        }
    }
    dsmc_step_end = MPI_Wtime();


    MPI_Barrier(comm);
    endTime = MPI_Wtime();
    if(mpiCtx.root()){
        printf("Total running time = %lf\n", (endTime - startTime));
        printf("Total time pre = %lf\n", (dsmc_step_end - dsmc_step_start));
    }
    
    if(mpiCtx.activeLeader()){
        printf("----------------------------\n");
    }


    delete mesh;
    delete mpass;
    delete dyprocess;
    delete partinit;
    delete process;
    delete progsis;
    delete nsprocess;
    if (mpiCtx.active()) MPI_Comm_free(&compute_comm);

    MPI_Finalize();
    return 0; 
}


void ns_result(int istep, MacroSolver *nsprocess)
{
    if (istep <= 2000 && istep%100 == 0) {nsprocess->nsout2dat(istep, nsprocess->Qf,nsprocess->Nl,nsprocess->Nr);}
    if (istep > 2000 && istep <= 10000 && (istep)%1000==0) {nsprocess->nsout2dat(istep, nsprocess->Qf,nsprocess->Nl,nsprocess->Nr);}
    if (istep > 10000 && istep <= 100000 && (istep)%5000==0) {nsprocess->nsout2dat(istep, nsprocess->Qf,nsprocess->Nl,nsprocess->Nr);}
    if (istep > 100000 && istep%50000==0) {nsprocess->nsout2dat(istep, nsprocess->Qf,nsprocess->Nl,nsprocess->Nr);}
}


void statistic_Result(int istep, ProcessDSMC *process)
{
    bool shouldOutputMacro = false;
    if (istep <= 2000 && istep%100 == 0) shouldOutputMacro = true;
    else if ((istep > 2000 && istep<= 10000)&& (istep)%1000==0) shouldOutputMacro = true;
    else if ((istep > 10000 && istep <= 50000)&&(istep)%2000==0) shouldOutputMacro = true;
    else if ((istep)%50000==0) shouldOutputMacro = true;

    if (shouldOutputMacro)
    {
        process->out2dat(istep);
    }

    const bool shouldOutputBoundary =
        EnableBoundaryStressHeatStatistic &&
        (istep > NSS) &&
        (NBoundaryStressHeatOutputEvery > 0) &&
        ((istep - NSS) % NBoundaryStressHeatOutputEvery == 0);
    if (shouldOutputBoundary)
    {
        process->outBoundaryStressHeat(istep);
    }
}
