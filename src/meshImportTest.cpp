/*
 * MPI driver for mesh setup, DSMC stepping, GSIS acceleration, and cleanup.
 */

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

/*
 * reportStepTimingMax: performs one solver support operation.
 * Params: mpiCtx, istep, duration_preprocess, duration_advection, duration_cache2chain, duration_collision, duration_statisticmacro; returns: none.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 */
static void reportStepTimingMax(const MpiContext& mpiCtx,
                                int istep,
                                double& duration_preprocess,
                                double& duration_advection,
                                double& duration_cache2chain,
                                double& duration_collision,
                                double& duration_statisticmacro)
{
    // Only produce timing output at the configured display interval.
    if (istep % NDSIPLAY != 0)
        return;

    // Collect the local timing counters in the same order used for output.
    double localTime[5] = {
        duration_preprocess,
        duration_advection,
        duration_cache2chain,
        duration_collision,
        duration_statisticmacro
    };

    // maxTime receives the per-stage maximum across active MPI ranks.
    double maxTime[5] = {0.0, 0.0, 0.0, 0.0, 0.0};

    // Inactive ranks are excluded because they do not own DSMC work.
    if (mpiCtx.active())
    {
        // The maximum value exposes the slowest rank, which limits progress.
        MPI_Reduce(localTime, maxTime, 5, MPI_DOUBLE, MPI_MAX, 0, mpiCtx.calGroup);

        // Only one active rank prints to keep the log readable.
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

    // Start a fresh accumulation window after each timing report.
    duration_preprocess = 0.0;
    duration_advection = 0.0;
    duration_cache2chain = 0.0;
    duration_collision = 0.0;
    duration_statisticmacro = 0.0;
}

/*
 * main: runs the full parallel DSMC/GSIS workflow.
 * Params: argc, argv; returns: index, count, owner, or status value.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 */
int main(int argc, char **argv)
{
    // Rank metadata and global wall-clock markers are initialized first.
    int rank, size;
    double startTime, endTime;

    // MPI_COMM_WORLD is kept as the global communicator for all ranks.
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // compute_comm contains only the ranks that participate in DSMC work.
    MPI_Comm compute_comm = MPI_COMM_NULL;
    const bool active = MpiContext::activeFromWorldRank(rank);
    int color = active ? 0 : MPI_UNDEFINED;  
    MPI_Comm_split(comm, color, rank, &compute_comm);

    // MpiContext centralizes root/leader tests and compute-rank numbering.
    MpiContext mpiCtx(comm, compute_comm);

    // MessagePassing owns helper routines for creating MPI message types.
    MessagePassing *mpass = new MessagePassing();

    // The mesh object exists on the root rank before broadcast; other ranks
    // use the compact meshMessage to construct their downstream state.
    meshImport *mesh = NULL;
    meshMessage mess;
    int c_rank = mpiCtx.c_rank; 
    int c_size = mpiCtx.c_size;
    int istep = 0;

    // Root imports the mesh file and publishes global mesh parameters.
    if (mpiCtx.root())
    {
        // argv[1] allows batch scripts to override the default Apollo mesh.
        const char *filePath = (argc > 1) ? argv[1] : "./mesh/3dapollo372500.cas";
        mesh = new meshImport(filePath, size - 1, 1);  
        mesh->rank = rank; mesh->c_size = size - 1;

        // Set flow control parameters before deriving the meshMessage.
        mesh->setMa_Kn_CFL(5, 0.01, 1e2, 1e3, 5);
        mesh->setMeshMessage();
        mess = mesh->mess;
    }

    // Broadcast meshMessage through the custom datatype understood by all ranks.
    MPI_Datatype myMessage;
    mpass->commitMyMesssge(myMessage);
    MPI_Bcast(&mess, 1, myMessage, 0, comm);
    MPI_Type_free(&myMessage);

    // Solver pointers are declared before allocation so cleanup can be explicit.
    dynamicDSMC *dyprocess = NULL;
    MacroSolver *nsprocess = NULL;
    MeshparticalInitial *partinit = NULL;
    ProcessDSMC *process = NULL;
    ProcessGSIS *progsis = NULL;

    // Build the macroscopic solver and mesh-particle partition initializer.
    nsprocess = new MacroSolver(mesh, mess,mpass,mpiCtx);
    partinit = new MeshparticalInitial(mesh, mess, mpass, mpiCtx);

    // Abort early if the partition files or ownership tables are incomplete.
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

    // Construct the DSMC, dynamic load-balance, and GSIS coupling modules.
    process = new ProcessDSMC(mesh,mess,partinit,mpiCtx);
    dyprocess = new dynamicDSMC(mesh, mpass, mess, partinit, process,mpiCtx);
    progsis = new ProcessGSIS(mesh, process, partinit, nsprocess,mpiCtx);

    // Initialize the particle distribution from a macroscopic G13 solve.
    nsprocess->nsProcessG13(1e5, 1e-5, false,istep);
    progsis->NS2DSMC();
    progsis->molecular_velocity_change();

    // Prepare and output the initial macroscopic DSMC statistics.
    process->statistic_macroPre();
    process->out2dat(istep);

    // Perform initial load distribution and register GSIS state variables.
    dyprocess->dynamic_rankload_distribute(c_size);
    progsis->regsisvariables();

    // Per-stage timers measure each major section of the DSMC loop.
    double preprocess_start, preprocess_end;
    double advection_start, advection_end;
    double cache2chain_start, cache2chain_end;
    double collision_start, collision_end;
    double statisticmacro_start, statisticmacro_end;
    double duration_preprocess=0.0, duration_advection=0.0, duration_cache2chain=0.0, duration_collision=0.0, duration_statisticmacro=0.0;
    double dynamic_start,dynamic_end;
    double dsmc_step_start, dsmc_step_end;

    // Synchronize before starting the main runtime clock.
    MPI_Barrier(comm);
    if(mpiCtx.root()){cout << "DSMC_begin "<< endl;}
    startTime = MPI_Wtime();

    // Main DSMC time-marching loop.
    for (istep = 1; istep <= NTOTAL; istep++)
    {   
        // Record the post-startup timing origin when the sampling step is reached.
        if(istep == NSS)
        {
            dsmc_step_start = MPI_Wtime();
        }

        // Active ranks execute particle-level DSMC operations.
        if (mpiCtx.active()) 
        {
            // Preprocessing prepares cell data and efficient quadrature state.
            preprocess_start=MPI_Wtime();
            process->preprocesseffquad(istep); 
            preprocess_end=MPI_Wtime();  
            duration_preprocess += (preprocess_end - preprocess_start);

            // Synchronize compute ranks before moving particles.
            MPI_Barrier(compute_comm);

            // Advection advances particle positions for the current time step.
            advection_start=MPI_Wtime(); 
            process->advection(istep);
            advection_end=MPI_Wtime(); 
            duration_advection += (advection_end - advection_start);

            // Rebuild linked-cell storage from temporary particle caches.
            cache2chain_start=MPI_Wtime();
            process->cache2chain();
            cache2chain_end=MPI_Wtime(); 
            duration_cache2chain += (cache2chain_end - cache2chain_start);

            // Apply DSMC collision sampling and molecular velocity updates.
            collision_start=MPI_Wtime();   
            process->collisionDSMC();
            collision_end=MPI_Wtime(); 
            duration_collision += (collision_end - collision_start);

            // Accumulate macroscopic moments from the updated particle state.
            preprocess_start=MPI_Wtime();
            process->statistic_macro(istep);
            preprocess_end=MPI_Wtime(); 
            duration_preprocess += (preprocess_end - preprocess_start);
        }

        // All ranks synchronize before shared output decisions are evaluated.
        MPI_Barrier(comm);

        // Scheduled field and boundary-statistic output is timed separately.
        statisticmacro_start=MPI_Wtime(); 
        statistic_Result(istep,process);
        statisticmacro_end=MPI_Wtime(); 
        duration_statisticmacro += (statisticmacro_end - statisticmacro_start);

        // Print the slowest-rank timing profile at the display interval.
        reportStepTimingMax(mpiCtx,
            istep,
            duration_preprocess,
            duration_advection,
            duration_cache2chain,
            duration_collision,
            duration_statisticmacro);

        // Optional GSIS acceleration periodically solves macroscopic equations.
        if (ifgsis == 1 && istep%NGSIS==0 && istep > NSCHEME)
        {
            if (c_rank ==0) {cout << "The macroscopic equations are solving after " << istep << " DSMC steps" << endl;}
            if (compute_comm != MPI_COMM_NULL)
            {
                // Coupled macro iteration drives DSMC toward the target tolerance.
                progsis->macro_iter_process(maxError,istep);
            }
            MPI_Barrier(comm);

            // Write macroscopic solver results on its own output schedule.
            ns_result(istep, nsprocess);
            if (c_rank ==0) {cout << "The macroscopic acceleration is done after " << istep << " DSMC steps" << endl;}
        }

        // Early iterations rebalance more often while the flow field settles.
        if(istep < 1000)
        {
            if(istep%200== 0)
            {
                MPI_Barrier(comm);
                dynamic_start = MPI_Wtime();

                // Repartition dynamic work and refresh GSIS variable ownership.
                dyprocess->dynamic_rankload_distribute(c_size);
                progsis->regsisvariables();
                dynamic_end = MPI_Wtime();

                // The compute leader reports the cost of dynamic redistribution.
                if(c_rank == 0)
                {
                    cout<< "dynamic_times   "<< dynamic_end - dynamic_start <<endl;
                }
            }
        }
        else
        {
            // Later iterations rebalance less frequently to reduce overhead.
            if(istep%1000 == 0)
            {
                MPI_Barrier(comm);
                dynamic_start = MPI_Wtime();

                // Keep rank load and GSIS bookkeeping consistent after migration.
                dyprocess->dynamic_rankload_distribute(c_size);
                progsis->regsisvariables();
                dynamic_end = MPI_Wtime();

                // The compute leader reports the cost of dynamic redistribution.
                if(c_rank == 0)
                {
                    cout<< "dynamic_times   "<< dynamic_end - dynamic_start <<endl;
                }
            }
        }
    }

    // Record final timing after the DSMC loop has completed.
    dsmc_step_end = MPI_Wtime();
    MPI_Barrier(comm);
    endTime = MPI_Wtime();

    // Root prints total wall time and the post-NSS DSMC timing segment.
    if(mpiCtx.root()){
        printf("Total running time = %lf\n", (endTime - startTime));
        printf("Total time pre = %lf\n", (dsmc_step_end - dsmc_step_start));
    }

    // The active leader prints a visual separator for the run log.
    if(mpiCtx.activeLeader()){
        printf("----------------------------\n");
    }

    // Delete heap objects in the reverse order of broad dependency use.
    delete mesh;
    delete mpass;
    delete dyprocess;
    delete partinit;
    delete process;
    delete progsis;
    delete nsprocess;

    // Free the split communicator only on ranks that actually created it.
    if (mpiCtx.active()) MPI_Comm_free(&compute_comm);

    // MPI_Finalize must be called by every rank before process exit.
    MPI_Finalize();
    return 0; 
}

/*
 * ns_result: performs one solver support operation.
 * Params: istep, nsprocess; returns: none.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 */
void ns_result(int istep, MacroSolver *nsprocess)
{
    // Dense early output captures the rapid initial macroscopic evolution.
    if (istep <= 2000 && istep%100 == 0) {nsprocess->nsout2dat(istep, nsprocess->Qf,nsprocess->Nl,nsprocess->Nr);}

    // Mid-range output reduces file count once the solution changes slower.
    if (istep > 2000 && istep <= 10000 && (istep)%1000==0) {nsprocess->nsout2dat(istep, nsprocess->Qf,nsprocess->Nl,nsprocess->Nr);}

    // Long transient output keeps periodic checkpoints without heavy I/O.
    if (istep > 10000 && istep <= 100000 && (istep)%5000==0) {nsprocess->nsout2dat(istep, nsprocess->Qf,nsprocess->Nl,nsprocess->Nr);}

    // Very late output uses the coarsest interval for production-length runs.
    if (istep > 100000 && istep%50000==0) {nsprocess->nsout2dat(istep, nsprocess->Qf,nsprocess->Nl,nsprocess->Nr);}
}

/*
 * statistic_Result: performs one solver support operation.
 * Params: istep, process; returns: none.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 */
void statistic_Result(int istep, ProcessDSMC *process)
{
    // Build the macroscopic field-output decision before doing any I/O.
    bool shouldOutputMacro = false;

    // The output cadence becomes progressively coarser as istep grows.
    if (istep <= 2000 && istep%100 == 0) shouldOutputMacro = true;
    else if ((istep > 2000 && istep<= 10000)&& (istep)%1000==0) shouldOutputMacro = true;
    else if ((istep > 10000 && istep <= 50000)&&(istep)%2000==0) shouldOutputMacro = true;
    else if ((istep)%50000==0) shouldOutputMacro = true;

    // Write DSMC macroscopic fields only on selected cadence steps.
    if (shouldOutputMacro)
    {
        process->out2dat(istep);
    }

    // Boundary stress and heat statistics have their own runtime switch.
    const bool shouldOutputBoundary =
        EnableBoundaryStressHeatStatistic &&
        (istep > NSS) &&
        (NBoundaryStressHeatOutputEvery > 0) &&
        ((istep - NSS) % NBoundaryStressHeatOutputEvery == 0);

    // Boundary output is independent from the macroscopic field cadence.
    if (shouldOutputBoundary)
    {
        process->outBoundaryStressHeat(istep);
    }
}
