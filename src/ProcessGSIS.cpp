/*
 * DSMC/NS exchange, filtering, reconstruction, and particle adjustment.
 */

# include "ProcessGSIS.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <random>
using namespace std;

namespace
{
/*
 * parseEnvBool: performs one solver support operation.
 * Params: value, defaultValue; returns: success or decision flag.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
static bool parseEnvBool(const char* value, bool defaultValue)
{
    if (value == nullptr || value[0] == '\0') return defaultValue;
    switch (value[0])
    {
        case '0':
        case 'n':
        case 'N':
        case 'f':
        case 'F':
            return false;
        case '1':
        case 'y':
        case 'Y':
        case 't':
        case 'T':
            return true;
        default:
            return defaultValue;
    }
}
/*
 * parseEnvDouble: performs one solver support operation.
 * Params: value, defaultValue; returns: computed scalar.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
static double parseEnvDouble(const char* value, double defaultValue)
{
    if (value == nullptr || value[0] == '\0') return defaultValue;
    char* end = nullptr;
    const double parsed = std::strtod(value, &end);
    if (end == value || !std::isfinite(parsed)) return defaultValue;
    return parsed;
}
/*
 * parseEnvInt: performs one solver support operation.
 * Params: value, defaultValue; returns: index, count, owner, or status value.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
static int parseEnvInt(const char* value, int defaultValue)
{
    if (value == nullptr || value[0] == '\0') return defaultValue;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value) return defaultValue;
    return (int)parsed;
}
/*
 * clampLower: performs one solver support operation.
 * Params: value, lowerBound; returns: computed scalar.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
static double clampLower(double value, double lowerBound)
{
    return (value < lowerBound) ? lowerBound : value;
}
/*
 * loadDsmc2NsCouplingConfig: updates partition ownership or load data.
 * Params: config; returns: ProcessGSIS::Dsmc2NsCouplingConfig.
 * Flow:
 *   - gather load/owner data.
 *   - build the new mapping.
 *   - refresh dependent local state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
static ProcessGSIS::Dsmc2NsCouplingConfig loadDsmc2NsCouplingConfig(
    ProcessGSIS::Dsmc2NsCouplingConfig config)
{
    config.sampleFilterEnabled =
        parseEnvBool(std::getenv("DSMC2NS_SAMPLE_FILTER"),
                     config.sampleFilterEnabled);
    config.macroMinSampleCount =
        clampLower(parseEnvDouble(std::getenv("DSMC2NS_MACRO_MIN_SAMPLES"),
                                  config.macroMinSampleCount), 0.0);
    config.highOrderSampleFilterEnabled =
        parseEnvBool(std::getenv("DSMC2NS_HIGH_ORDER_SAMPLE_FILTER"),
                     config.highOrderSampleFilterEnabled);
    config.highOrderMinSampleCount =
        clampLower(parseEnvDouble(std::getenv("DSMC2NS_HIGH_ORDER_MIN_SAMPLES"),
                                  config.highOrderMinSampleCount), 0.0);
    config.macroLowerBoundsEnabled =
        parseEnvBool(std::getenv("DSMC2NS_MACRO_LOWER_BOUNDS"),
                     parseEnvBool(std::getenv("DSMC2NS_MACRO_BOUNDS"),
                                  config.macroLowerBoundsEnabled));
    config.macroUpperBoundsEnabled =
        parseEnvBool(std::getenv("DSMC2NS_MACRO_UPPER_BOUNDS"),
                     config.macroUpperBoundsEnabled);
    config.rhoMin = parseEnvDouble(std::getenv("DSMC2NS_RHO_MIN"), config.rhoMin);
    config.rhoMax = parseEnvDouble(std::getenv("DSMC2NS_RHO_MAX"), config.rhoMax);
    config.tMin = parseEnvDouble(std::getenv("DSMC2NS_T_MIN"), config.tMin);
    config.tMax = parseEnvDouble(std::getenv("DSMC2NS_T_MAX"), config.tMax);
    config.trMin = parseEnvDouble(std::getenv("DSMC2NS_TR_MIN"), config.trMin);
    config.trMax = parseEnvDouble(std::getenv("DSMC2NS_TR_MAX"), config.trMax);
    config.maMin = parseEnvDouble(std::getenv("DSMC2NS_MA_MIN"), config.maMin);
    config.maMax = parseEnvDouble(std::getenv("DSMC2NS_MA_MAX"), config.maMax);
    config.knCellUpperBoundEnabled =
        parseEnvBool(std::getenv("DSMC2NS_KN_CELL_UPPER_BOUND"),
                     parseEnvBool(std::getenv("DSMC2NS_KN_CELL_FILTER"),
                                  config.knCellUpperBoundEnabled));
    config.knCellMax =
        clampLower(parseEnvDouble(std::getenv("DSMC2NS_KN_CELL_MAX"),
                                  config.knCellMax), 0.0);
    config.highOrderLowerBoundsEnabled =
        parseEnvBool(std::getenv("DSMC2NS_HIGH_ORDER_LOWER_BOUNDS"),
                     parseEnvBool(std::getenv("DSMC2NS_HIGH_ORDER_BOUNDS"),
                                  config.highOrderLowerBoundsEnabled));
    config.highOrderUpperBoundsEnabled =
        parseEnvBool(std::getenv("DSMC2NS_HIGH_ORDER_UPPER_BOUNDS"),
                     config.highOrderUpperBoundsEnabled);
    config.stressRatioMin =
        parseEnvDouble(std::getenv("DSMC2NS_STRESS_RATIO_MIN"),
                       config.stressRatioMin);
    config.stressRatioMax =
        parseEnvDouble(std::getenv("DSMC2NS_STRESS_RATIO_MAX"),
                       config.stressRatioMax);
    config.heatRatioMin =
        parseEnvDouble(std::getenv("DSMC2NS_HEAT_RATIO_MIN"),
                       config.heatRatioMin);
    config.heatRatioMax =
        parseEnvDouble(std::getenv("DSMC2NS_HEAT_RATIO_MAX"),
                       config.heatRatioMax);
    config.rotHeatRatioMin =
        parseEnvDouble(std::getenv("DSMC2NS_ROT_HEAT_RATIO_MIN"),
                       config.rotHeatRatioMin);
    config.rotHeatRatioMax =
        parseEnvDouble(std::getenv("DSMC2NS_ROT_HEAT_RATIO_MAX"),
                       config.rotHeatRatioMax);
    config.lowOrderHarmonicEnabled =
        parseEnvBool(std::getenv("DSMC2NS_LOW_ORDER_HARMONIC"),
                     config.lowOrderHarmonicEnabled);
    config.lowOrderHarmonicIterations =
        std::max(0, parseEnvInt(std::getenv("DSMC2NS_LOW_ORDER_HARMONIC_ITER"),
                                config.lowOrderHarmonicIterations));
    config.lowOrderHarmonicOmega =
        clampLower(parseEnvDouble(std::getenv("DSMC2NS_LOW_ORDER_HARMONIC_OMEGA"),
                                  config.lowOrderHarmonicOmega), 0.0);
    config.highOrderDampingCoeff =
        clampLower(parseEnvDouble(std::getenv("DSMC2NS_HIGH_ORDER_DAMPING"),
                                  config.highOrderDampingCoeff), 0.0);
    config.logFilterSummary =
        parseEnvBool(std::getenv("DSMC2NS_FILTER_LOG"),
                     config.logFilterSummary);
    if (config.rhoMax < config.rhoMin) std::swap(config.rhoMax, config.rhoMin);
    if (config.tMax < config.tMin) std::swap(config.tMax, config.tMin);
    if (config.trMax < config.trMin) std::swap(config.trMax, config.trMin);
    if (config.maMax < config.maMin) std::swap(config.maMax, config.maMin);
    if (config.stressRatioMax < config.stressRatioMin)
        std::swap(config.stressRatioMax, config.stressRatioMin);
    if (config.heatRatioMax < config.heatRatioMin)
        std::swap(config.heatRatioMax, config.heatRatioMin);
    if (config.rotHeatRatioMax < config.rotHeatRatioMin)
        std::swap(config.rotHeatRatioMax, config.rotHeatRatioMin);
    return config;
}
/*
 * positiveUniform: performs one solver support operation.
 * Params: dis, gen; returns: computed scalar.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
static double positiveUniform(std::uniform_real_distribution<double>& dis,
                              std::mt19937& gen)
{
    const double r = dis(gen);
    return (r > 0.0) ? r : 1.0e-300;
}
/*
 * finiteAtIndices: performs one solver support operation.
 * Params: values, indices, count; returns: success or decision flag.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
static bool finiteAtIndices(const double* values, const int* indices, int count)
{
    for (int i = 0; i < count; ++i)
    {
        if (!std::isfinite(values[indices[i]]))
            return false;
    }
    return true;
}
/*
 * dsmc2nsConfiguredSampleAccepted: updates particles or particle-derived state.
 * Params: sampleCount, filterEnabled, minSampleCount; returns: success or decision flag.
 * Flow:
 *   - map ownership.
 *   - filter or reconstruct fields.
 *   - write accepted coupled state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
static bool dsmc2nsConfiguredSampleAccepted(double sampleCount,
                                            bool filterEnabled,
                                            double minSampleCount)
{
    if (!std::isfinite(sampleCount)) return false;
    return !filterEnabled || sampleCount > minSampleCount;
}
/*
 * macroLowOrderFinite: couples DSMC data with macroscopic fields.
 * Params: values; returns: success or decision flag.
 * Flow:
 *   - map ownership.
 *   - filter or reconstruct fields.
 *   - write accepted coupled state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
static bool macroLowOrderFinite(const double* values)
{
    const int lowOrderIndices[] = {0, 1, 2, 3, 4, 14};
    return finiteAtIndices(values, lowOrderIndices,
                           (int)(sizeof(lowOrderIndices) / sizeof(lowOrderIndices[0])));
}
/*
 * macroHighOrderFinite: couples DSMC data with macroscopic fields.
 * Params: values; returns: success or decision flag.
 * Flow:
 *   - map ownership.
 *   - filter or reconstruct fields.
 *   - write accepted coupled state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
static bool macroHighOrderFinite(const double* values)
{
    const int highOrderIndices[] = {5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17};
    return finiteAtIndices(values, highOrderIndices,
                           (int)(sizeof(highOrderIndices) / sizeof(highOrderIndices[0])));
}
struct Dsmc2NsMetrics
{
    double rho = 0.0;
    double ux = 0.0;
    double uy = 0.0;
    double uz = 0.0;
    double T = 0.0;
    double Tr = 0.0;
    double sampleCount = 0.0;
    double Ma = 1.0e300;
    double stressRatio = 1.0e300;
    double heatRatio = 1.0e300;
    double rotHeatRatio = 1.0e300;
    double knCell = 1.0e300;
};
/*
 * buildDsmc2NsMetrics: works with mesh topology or geometric intersections.
 * Params: in, sampleCount, ns, cellVolume; returns: Dsmc2NsMetrics.
 * Flow:
 *   - map ownership.
 *   - filter or reconstruct fields.
 *   - write accepted coupled state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
static Dsmc2NsMetrics buildDsmc2NsMetrics(const double* in,
                                          double sampleCount,
                                          const MacroSolver* ns,
                                          double cellVolume)
{
    const double sqrt2 = std::sqrt(2.0);
    Dsmc2NsMetrics metrics;
    metrics.rho = in[0];
    metrics.ux = sqrt2 * in[1];
    metrics.uy = sqrt2 * in[2];
    metrics.uz = sqrt2 * in[3];
    metrics.T = in[4];
    metrics.Tr = in[14];
    metrics.sampleCount = sampleCount;
    const double u2 = metrics.ux * metrics.ux +
                      metrics.uy * metrics.uy +
                      metrics.uz * metrics.uz;
    if (metrics.T > 0.0 && std::isfinite(metrics.T) &&
/*
 * isfinite: performs one solver support operation.
 * Params: u2; returns: none.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
        std::isfinite(u2) && ns != nullptr)
    {
        metrics.Ma = std::sqrt(u2) / std::sqrt(ns->mess.gamma * metrics.T);
    }
    if (metrics.rho > 0.0 && metrics.T > 0.0 && cellVolume > 0.0 &&
        std::isfinite(metrics.rho) && std::isfinite(metrics.T) &&
/*
 * isfinite: performs one solver support operation.
 * Params: cellVolume; returns: none.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
        std::isfinite(cellVolume) && ns != nullptr)
    {
        const double pi = std::acos(-1.0);
        const double sphereDiameter =
            2.0 * std::pow(3.0 * cellVolume / (4.0 * pi), 1.0 / 3.0);
        if (sphereDiameter > 0.0 && std::isfinite(sphereDiameter))
        {
            const double meanFreePath =
                ns->mess.Kn * std::pow(metrics.T, ns->mess.omega - 0.5) /
                metrics.rho;
            if (std::isfinite(meanFreePath))
                metrics.knCell = meanFreePath / sphereDiameter;
        }
    }
    const double pressure = metrics.rho * metrics.T;
    const double stressMax = std::max(
        std::max(std::max(std::fabs(in[5]), std::fabs(in[6])),
                 std::max(std::fabs(in[7]), std::fabs(in[8]))),
        std::max(std::fabs(in[9]), std::fabs(in[10])));
    const double heatMax = sqrt2 * std::max(std::fabs(in[11]),
                                            std::max(std::fabs(in[12]),
                                                     std::fabs(in[13])));
    const double rotHeatMax = sqrt2 * std::max(std::fabs(in[15]),
                                               std::max(std::fabs(in[16]),
                                                        std::fabs(in[17])));
    const double heatDenom =
        (pressure > 0.0 && metrics.T > 0.0 &&
         std::isfinite(pressure) && std::isfinite(metrics.T))
            ? pressure * std::sqrt(metrics.T)
            : 0.0;
    if (pressure > 0.0 && std::isfinite(pressure))
        metrics.stressRatio = stressMax / pressure;
    if (heatDenom > 0.0 && std::isfinite(heatDenom))
    {
        metrics.heatRatio = heatMax / heatDenom;
        metrics.rotHeatRatio = rotHeatMax / heatDenom;
    }
    return metrics;
}
struct Dsmc2NsDecision
{
    bool lowSampleMacro = false;
    bool lowSampleHighOrder = false;
    bool macroFiniteReject = false;
    bool macroLowerBoundsReject = false;
    bool macroUpperBoundsReject = false;
    bool knCellUpperBoundsReject = false;
    bool highOrderFiniteReject = false;
    bool highOrderLowerBoundsReject = false;
    bool highOrderUpperBoundsReject = false;
    bool macroReliable = false;
    bool highOrderReliable = false;
};
/*
 * decideDsmc2NsCoupling: couples DSMC data with macroscopic fields.
 * Params: config, in, metrics; returns: Dsmc2NsDecision.
 * Flow:
 *   - map ownership.
 *   - filter or reconstruct fields.
 *   - write accepted coupled state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
static Dsmc2NsDecision decideDsmc2NsCoupling(
    const ProcessGSIS::Dsmc2NsCouplingConfig& config,
    const double* in,
    const Dsmc2NsMetrics& metrics)
{
    Dsmc2NsDecision decision;
    decision.lowSampleMacro =
        !dsmc2nsConfiguredSampleAccepted(metrics.sampleCount,
                                         config.sampleFilterEnabled,
                                         config.macroMinSampleCount);
    decision.macroFiniteReject = !macroLowOrderFinite(in);
    decision.macroLowerBoundsReject =
        config.macroLowerBoundsEnabled &&
        (!std::isfinite(metrics.rho) || metrics.rho < config.rhoMin ||
         !std::isfinite(metrics.T) || metrics.T < config.tMin ||
         !std::isfinite(metrics.Tr) || metrics.Tr < config.trMin ||
         !std::isfinite(metrics.Ma) || metrics.Ma < config.maMin);
    decision.macroUpperBoundsReject =
        config.macroUpperBoundsEnabled &&
        (!std::isfinite(metrics.rho) || metrics.rho > config.rhoMax ||
         !std::isfinite(metrics.T) || metrics.T > config.tMax ||
         !std::isfinite(metrics.Tr) || metrics.Tr > config.trMax ||
         !std::isfinite(metrics.Ma) || metrics.Ma > config.maMax);
    decision.knCellUpperBoundsReject =
        config.knCellUpperBoundEnabled &&
        (!std::isfinite(metrics.knCell) || metrics.knCell > config.knCellMax);
    decision.macroReliable =
        !decision.lowSampleMacro &&
        !decision.macroFiniteReject &&
        !decision.macroLowerBoundsReject &&
        !decision.macroUpperBoundsReject &&
        !decision.knCellUpperBoundsReject;
    decision.lowSampleHighOrder =
        config.highOrderSampleFilterEnabled &&
        !dsmc2nsConfiguredSampleAccepted(metrics.sampleCount,
                                         config.sampleFilterEnabled,
                                         config.highOrderMinSampleCount);
    decision.highOrderFiniteReject = !macroHighOrderFinite(in);
    decision.highOrderLowerBoundsReject =
        config.highOrderLowerBoundsEnabled &&
        (!std::isfinite(metrics.stressRatio) ||
         metrics.stressRatio < config.stressRatioMin ||
         !std::isfinite(metrics.heatRatio) ||
         metrics.heatRatio < config.heatRatioMin ||
         !std::isfinite(metrics.rotHeatRatio) ||
         metrics.rotHeatRatio < config.rotHeatRatioMin);
    decision.highOrderUpperBoundsReject =
        config.highOrderUpperBoundsEnabled &&
        (!std::isfinite(metrics.stressRatio) ||
         metrics.stressRatio > config.stressRatioMax ||
         !std::isfinite(metrics.heatRatio) ||
         metrics.heatRatio > config.heatRatioMax ||
         !std::isfinite(metrics.rotHeatRatio) ||
         metrics.rotHeatRatio > config.rotHeatRatioMax);
    decision.highOrderReliable =
        decision.macroReliable &&
        !decision.lowSampleHighOrder &&
        !decision.highOrderFiniteReject &&
        !decision.highOrderLowerBoundsReject &&
        !decision.highOrderUpperBoundsReject;
    return decision;
}
/*
 * randomBarycentric4: works with mesh topology or geometric intersections.
 * Params: dis, gen, a, b, c, d; returns: none.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
static void randomBarycentric4(std::uniform_real_distribution<double>& dis,
                               std::mt19937& gen,
                               double& a,
                               double& b,
                               double& c,
                               double& d)
{
    const double r0 = positiveUniform(dis, gen);
    const double r1 = positiveUniform(dis, gen);
    const double r2 = positiveUniform(dis, gen);
    const double r3 = positiveUniform(dis, gen);
    const double sum = r0 + r1 + r2 + r3;
    if (!(sum > 0.0) || !std::isfinite(sum))
    {
        a = b = c = d = 0.25;
        return;
    }
    a = r0 / sum;
    b = r1 / sum;
    c = r2 / sum;
    d = r3 / sum;
}
struct GsisPacketExchange
{
/*
 * prefixCounts: performs one solver support operation.
 * Params: counts, displs, total; returns: none.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
    static void prefixCounts(const vector<int> &counts, vector<int> &displs, int &total)
    {
        displs.assign(counts.size(), 0);
        total = 0;
        for (size_t i = 0; i < counts.size(); ++i)
        {
            displs[i] = total;
            total += counts[i];
        }
    }
/*
 * exchange: moves structured data through MPI.
 * Params: comm, rankCount, valueWidth, sendCounts, sendGids, sendValues, recvGids, recvValues; returns: success or decision flag.
 * Flow:
 *   - prepare counts or datatypes.
 *   - perform MPI transfer.
 *   - store received data.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
    static bool exchange(MPI_Comm comm,
                         int rankCount,
                         int valueWidth,
                         const vector<int> &sendCounts,
                         const vector<int> &sendGids,
                         const vector<double> &sendValues,
                         vector<int> &recvGids,
                         vector<double> &recvValues)
    {
        if (rankCount <= 0 || valueWidth <= 0 || (int)sendCounts.size() != rankCount)
            return false;
        int expectedSend = 0;
        bool localOk = true;
        for (int count : sendCounts)
        {
            if (count < 0) localOk = false;
            expectedSend += count;
        }
        if (expectedSend < 0 ||
            expectedSend != (int)sendGids.size() ||
            expectedSend * valueWidth != (int)sendValues.size())
            localOk = false;
        int localOkInt = localOk ? 1 : 0;
        vector<int> allOk((size_t)rankCount, 0);
        MPI_Allgather(&localOkInt, 1, MPI_INT, allOk.data(), 1, MPI_INT, comm);
        if (find(allOk.begin(), allOk.end(), 0) != allOk.end())
            return false;
        vector<int> recvCounts((size_t)rankCount, 0);
        MPI_Alltoall(const_cast<int*>(sendCounts.data()), 1, MPI_INT,
                     recvCounts.data(), 1, MPI_INT, comm);
        vector<int> sendDispls, recvDispls;
        int sendTotal = 0, recvTotal = 0;
        prefixCounts(sendCounts, sendDispls, sendTotal);
        prefixCounts(recvCounts, recvDispls, recvTotal);
        vector<int> sendValueCounts((size_t)rankCount, 0), recvValueCounts((size_t)rankCount, 0);
        vector<int> sendValueDispls((size_t)rankCount, 0), recvValueDispls((size_t)rankCount, 0);
        for (int r = 0; r < rankCount; ++r)
        {
            sendValueCounts[(size_t)r] = sendCounts[(size_t)r] * valueWidth;
            recvValueCounts[(size_t)r] = recvCounts[(size_t)r] * valueWidth;
            sendValueDispls[(size_t)r] = sendDispls[(size_t)r] * valueWidth;
            recvValueDispls[(size_t)r] = recvDispls[(size_t)r] * valueWidth;
        }
        recvGids.assign((size_t)recvTotal, -1);
        recvValues.assign((size_t)recvTotal * (size_t)valueWidth, 0.0);
        MPI_Alltoallv(sendGids.empty() ? nullptr : const_cast<int*>(sendGids.data()),
                      const_cast<int*>(sendCounts.data()),
                      sendDispls.empty() ? nullptr : sendDispls.data(),
                      MPI_INT,
                      recvGids.empty() ? nullptr : recvGids.data(),
                      recvCounts.data(),
                      recvDispls.empty() ? nullptr : recvDispls.data(),
                      MPI_INT,
                      comm);
        MPI_Alltoallv(sendValues.empty() ? nullptr : const_cast<double*>(sendValues.data()),
                      sendValueCounts.data(),
                      sendValueDispls.data(),
                      MPI_DOUBLE,
                      recvValues.empty() ? nullptr : recvValues.data(),
                      recvValueCounts.data(),
                      recvValueDispls.data(),
                      MPI_DOUBLE,
                      comm);
        return sendTotal == (int)sendGids.size() &&
               sendTotal * valueWidth == (int)sendValues.size();
    }
};
struct InitTetLocal
{
    int node[4] = {-1, -1, -1, -1};
    double volume = 0.0;
};
/*
 * validLocalNodeIndex: works with mesh topology or geometric intersections.
 * Params: idx, xyz; returns: success or decision flag.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
static bool validLocalNodeIndex(int idx, const vector<double>& xyz)
{
    return idx >= 0 && (size_t)idx < xyz.size() / 3u;
}
/*
 * tetAbsVolume: works with mesh topology or geometric intersections.
 * Params: xyz, n0, n1, n2, n3; returns: computed scalar.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
static double tetAbsVolume(const vector<double>& xyz,
                           int n0, int n1, int n2, int n3)
{
    const double ax = xyz[3 * n1 + 0] - xyz[3 * n0 + 0];
    const double ay = xyz[3 * n1 + 1] - xyz[3 * n0 + 1];
    const double az = xyz[3 * n1 + 2] - xyz[3 * n0 + 2];
    const double bx = xyz[3 * n2 + 0] - xyz[3 * n0 + 0];
    const double by = xyz[3 * n2 + 1] - xyz[3 * n0 + 1];
    const double bz = xyz[3 * n2 + 2] - xyz[3 * n0 + 2];
    const double cx = xyz[3 * n3 + 0] - xyz[3 * n0 + 0];
    const double cy = xyz[3 * n3 + 1] - xyz[3 * n0 + 1];
    const double cz = xyz[3 * n3 + 2] - xyz[3 * n0 + 2];
    const double crx = by * cz - bz * cy;
    const double cry = bz * cx - bx * cz;
    const double crz = bx * cy - by * cx;
    return fabs(ax * crx + ay * cry + az * crz) / 6.0;
}
/*
 * appendVertexTet: works with mesh topology or geometric intersections.
 * Params: process, n0, n1, n2, n3, tets; returns: success or decision flag.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
static bool appendVertexTet(const ProcessDSMC& process,
                            int n0, int n1, int n2, int n3,
                            vector<InitTetLocal>& tets)
{
    const int nodes[4] = {n0, n1, n2, n3};
    for (int k = 0; k < 4; ++k)
        if (!validLocalNodeIndex(nodes[k], process.localPointXY)) return false;
    InitTetLocal tet;
    for (int k = 0; k < 4; ++k) tet.node[k] = nodes[k];
    tet.volume = tetAbsVolume(process.localPointXY, n0, n1, n2, n3);
    if (!(tet.volume > 0.0) || !isfinite(tet.volume)) return false;
    tets.push_back(tet);
    return true;
}
/*
 * collectCellUniqueNodes: works with mesh topology or geometric intersections.
 * Params: process, cell, uniqueNodes; returns: success or decision flag.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
static bool collectCellUniqueNodes(const ProcessDSMC& process,
                                   const DsmcCell& cell,
                                   vector<int>& uniqueNodes)
{
    uniqueNodes.clear();
    for (int m = 0; m < cell.num && m < NN; ++m)
    {
        const int faceLocal = cell.cell2face[m];
        if (faceLocal < 0 || faceLocal >= (int)process.edges.size()) return false;
        const DsmcEdge& face = process.edges[(size_t)faceLocal];
        if (face.faceType != 3 && face.faceType != 4) return false;
        for (int k = 0; k < face.faceType; ++k)
        {
            const int node = face.faceMap[k];
            if (!validLocalNodeIndex(node, process.localPointXY)) return false;
            if (find(uniqueNodes.begin(), uniqueNodes.end(), node) == uniqueNodes.end())
                uniqueNodes.push_back(node);
        }
    }
    return true;
}
/*
 * appendTetCellTets: works with mesh topology or geometric intersections.
 * Params: process, cell, tets; returns: success or decision flag.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
static bool appendTetCellTets(const ProcessDSMC& process,
                              const DsmcCell& cell,
                              vector<InitTetLocal>& tets)
{
    vector<int> uniqueNodes;
    if (!collectCellUniqueNodes(process, cell, uniqueNodes)) return false;
    if (uniqueNodes.size() != 4u) return false;
    return appendVertexTet(process, uniqueNodes[0], uniqueNodes[1],
                           uniqueNodes[2], uniqueNodes[3], tets);
}
/*
 * appendPyramidCellTets: works with mesh topology or geometric intersections.
 * Params: process, cell, tets; returns: success or decision flag.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
static bool appendPyramidCellTets(const ProcessDSMC& process,
                                  const DsmcCell& cell,
                                  vector<InitTetLocal>& tets)
{
    int quadFace = -1;
    for (int m = 0; m < cell.num && m < NN; ++m)
    {
        const int faceLocal = cell.cell2face[m];
        if (faceLocal < 0 || faceLocal >= (int)process.edges.size()) return false;
        const DsmcEdge& face = process.edges[(size_t)faceLocal];
        if (face.faceType == 4)
        {
            if (quadFace >= 0) return false;
            quadFace = faceLocal;
        }
    }
    if (quadFace < 0) return false;
    vector<int> uniqueNodes;
    if (!collectCellUniqueNodes(process, cell, uniqueNodes)) return false;
    if (uniqueNodes.size() != 5u) return false;
    const DsmcEdge& base = process.edges[(size_t)quadFace];
    int apex = -1;
    for (int node : uniqueNodes)
    {
        bool onBase = false;
        for (int k = 0; k < 4; ++k)
        {
            if (base.faceMap[k] == node)
            {
                onBase = true;
                break;
            }
        }
        if (!onBase)
        {
            apex = node;
            break;
        }
    }
    if (apex < 0) return false;
    unsigned char tag = (quadFace >= 0 && quadFace < (int)process.faceSplitTag.size())
        ? process.faceSplitTag[(size_t)quadFace]
        : meshImport::FACE_SPLIT_02;
    int tri0[3], tri1[3];
    meshImport::decode_quad_split_tag(tag, tri0, tri1);
    if (tri0[0] < 0 || tri1[0] < 0)
        meshImport::decode_quad_split_tag(meshImport::FACE_SPLIT_02, tri0, tri1);
    if (tri0[0] < 0 || tri1[0] < 0) return false;
    return appendVertexTet(process,
                           base.faceMap[tri0[0]],
                           base.faceMap[tri0[1]],
                           base.faceMap[tri0[2]],
                           apex,
                           tets) &&
           appendVertexTet(process,
                           base.faceMap[tri1[0]],
                           base.faceMap[tri1[1]],
                           base.faceMap[tri1[2]],
                           apex,
                           tets);
}
/*
 * appendCenterFaceTet: works with mesh topology or geometric intersections.
 * Params: process, cellLocal, faceLocal, tri, tets; returns: success or decision flag.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
static bool appendCenterFaceTet(const ProcessDSMC& process,
                                int cellLocal,
                                int faceLocal,
                                const int tri[3],
                                vector<InitTetLocal>& tets)
{
    if (faceLocal < 0 || faceLocal >= (int)process.edges.size()) return false;
    const DsmcEdge& face = process.edges[(size_t)faceLocal];
    const double* center = process.cells[(size_t)cellLocal].cellXY;
    InitTetLocal tet;
    for (int k = 0; k < 3; ++k)
    {
        const int nodeSlot = tri[k];
        if (nodeSlot < 0 || nodeSlot >= face.faceType) return false;
        tet.node[k] = face.faceMap[nodeSlot];
        if (!validLocalNodeIndex(tet.node[k], process.localPointXY)) return false;
    }
    tet.node[3] = -1;
    const int n0 = tet.node[0];
    const int n1 = tet.node[1];
    const int n2 = tet.node[2];
    const double ax = process.localPointXY[3 * n0 + 0] - center[0];
    const double ay = process.localPointXY[3 * n0 + 1] - center[1];
    const double az = process.localPointXY[3 * n0 + 2] - center[2];
    const double bx = process.localPointXY[3 * n1 + 0] - center[0];
    const double by = process.localPointXY[3 * n1 + 1] - center[1];
    const double bz = process.localPointXY[3 * n1 + 2] - center[2];
    const double cx = process.localPointXY[3 * n2 + 0] - center[0];
    const double cy = process.localPointXY[3 * n2 + 1] - center[1];
    const double cz = process.localPointXY[3 * n2 + 2] - center[2];
    const double crx = by * cz - bz * cy;
    const double cry = bz * cx - bx * cz;
    const double crz = bx * cy - by * cx;
    tet.volume = fabs(ax * crx + ay * cry + az * crz) / 6.0;
    if (!(tet.volume > 0.0) || !isfinite(tet.volume)) return false;
    tets.push_back(tet);
    return true;
}
/*
 * appendCenterFaceTets: works with mesh topology or geometric intersections.
 * Params: process, cellLocal, cell, tets; returns: success or decision flag.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
static bool appendCenterFaceTets(const ProcessDSMC& process,
                                 int cellLocal,
                                 const DsmcCell& cell,
                                 vector<InitTetLocal>& tets)
{
    for (int m = 0; m < cell.num && m < NN; ++m)
    {
        const int faceLocal = cell.cell2face[m];
        if (faceLocal < 0 || faceLocal >= (int)process.edges.size()) return false;
        const DsmcEdge& face = process.edges[(size_t)faceLocal];
        if (face.faceType == 3)
        {
            const int tri[3] = {0, 1, 2};
            if (!appendCenterFaceTet(process, cellLocal, faceLocal, tri, tets)) return false;
        }
        else if (face.faceType == 4)
        {
            unsigned char tag = (faceLocal >= 0 && faceLocal < (int)process.faceSplitTag.size())
                ? process.faceSplitTag[(size_t)faceLocal]
                : meshImport::FACE_SPLIT_02;
            int tri0[3], tri1[3];
            meshImport::decode_quad_split_tag(tag, tri0, tri1);
            if (tri0[0] < 0 || tri1[0] < 0)
                meshImport::decode_quad_split_tag(meshImport::FACE_SPLIT_02, tri0, tri1);
            if (!appendCenterFaceTet(process, cellLocal, faceLocal, tri0, tets)) return false;
            if (!appendCenterFaceTet(process, cellLocal, faceLocal, tri1, tets)) return false;
        }
        else
        {
            return false;
        }
    }
    return !tets.empty();
}
/*
 * buildCellInitTets: works with mesh topology or geometric intersections.
 * Params: process, cellLocal, tets; returns: success or decision flag.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
static bool buildCellInitTets(const ProcessDSMC& process,
                              int cellLocal,
                              vector<InitTetLocal>& tets)
{
    tets.clear();
    if (cellLocal < 0 || cellLocal >= (int)process.cells.size()) return false;
    const DsmcCell& cell = process.cells[(size_t)cellLocal];
    if (appendTetCellTets(process, cell, tets))
        return true;
    tets.clear();
    if (appendPyramidCellTets(process, cell, tets))
        return true;
    tets.clear();
    return appendCenterFaceTets(process, cellLocal, cell, tets);
}
/*
 * sampleCellLocation: works with mesh topology or geometric intersections.
 * Params: process, cellLocal, dis, gen, out; returns: success or decision flag.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
static bool sampleCellLocation(const ProcessDSMC& process,
                               int cellLocal,
                               std::uniform_real_distribution<double>& dis,
                               std::mt19937& gen,
                               double out[DIM])
{
    vector<InitTetLocal> tets;
    if (!buildCellInitTets(process, cellLocal, tets)) return false;
    double totalVolume = 0.0;
    for (const InitTetLocal& tet : tets)
        totalVolume += tet.volume;
    if (!(totalVolume > 0.0) || !isfinite(totalVolume)) return false;
    double pick = dis(gen) * totalVolume;
    const InitTetLocal* chosen = &tets.back();
    for (const InitTetLocal& tet : tets)
    {
        if (pick <= tet.volume)
        {
            chosen = &tet;
            break;
        }
        pick -= tet.volume;
    }
    double a, b, c, d;
    randomBarycentric4(dis, gen, a, b, c, d);
    if (chosen->node[3] >= 0)
    {
        const int n0 = chosen->node[0];
        const int n1 = chosen->node[1];
        const int n2 = chosen->node[2];
        const int n3 = chosen->node[3];
        for (int dim = 0; dim < DIM; ++dim)
        {
            out[dim] = a * process.localPointXY[3 * n0 + dim] +
                       b * process.localPointXY[3 * n1 + dim] +
                       c * process.localPointXY[3 * n2 + dim] +
                       d * process.localPointXY[3 * n3 + dim];
        }
        return true;
    }
    const double* center = process.cells[(size_t)cellLocal].cellXY;
    const int n0 = chosen->node[0];
    const int n1 = chosen->node[1];
    const int n2 = chosen->node[2];
    for (int dim = 0; dim < DIM; ++dim)
    {
        out[dim] = a * center[dim] +
                   b * process.localPointXY[3 * n0 + dim] +
                   c * process.localPointXY[3 * n1 + dim] +
                   d * process.localPointXY[3 * n2 + dim];
    }
    return true;
}
}

/*
 * ProcessGSIS: initializes ProcessGSIS state.
 * Params: mesh, process, partinit, nsprocess, mpiCtx; returns: none.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
ProcessGSIS::ProcessGSIS(meshImport *mesh, ProcessDSMC *process, MeshparticalInitial *partinit, MacroSolver *nsprocess, const MpiContext& mpiCtx)
{
    this->mpi = &mpiCtx;
    this->comm = mpiCtx.comm;
    this->calGroup = mpiCtx.calGroup;
    if (!this->mpi->active()) return;
    this->mesh = mesh;
    this->process = process;
    this->partinit = partinit;
    this->pnsSolver = nsprocess;
    this->c_rank = mpiCtx.c_rank;
    this->c_size = mpiCtx.c_size;
    this->icell = this->process->iNcell;
    this->mess = this->process->mess;
    synchronizeBoundaryTables();
    this->use_exp_weighted_dsmc2ns =
        parseEnvBool(std::getenv("DSMC2NS_EXP_WEIGHT"),
                     this->use_exp_weighted_dsmc2ns);
    this->dsmc2nsCoupling =
        loadDsmc2NsCouplingConfig(this->dsmc2nsCoupling);
    variablesetup();
}

/*
 * synchronizeBoundaryTables: applies boundary-condition behavior.
 * Params: none; returns: none.
 * Flow:
 *   - select boundary faces.
 *   - apply the physical model.
 *   - store flux or particle effects.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
void ProcessGSIS::synchronizeBoundaryTables()
{
    if (this->process == nullptr || this->pnsSolver == nullptr) return;
    this->process->boundaryTable = this->pnsSolver->boundaryTable;
    this->process->rebuildBoundaryDerivedState();
}

/*
 * variablesetup: prepares derived solver state.
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
void ProcessGSIS::variablesetup()
{
    this->nsresult.assign((size_t)this->icell * 6u, 0.0);
    this->dsmc2ns_macro_accepted.assign((size_t)this->icell, 1);
    if (this->pnsSolver != nullptr)
    {
        this->ns_dsmc2ns_macro_accepted.assign((size_t)this->pnsSolver->iNcell, 0);
        this->ns_dsmc2ns_high_order_accepted.assign((size_t)this->pnsSolver->iNcell, 0);
        this->ns_dsmc2ns_macro_source.assign((size_t)this->pnsSolver->Ncell, MACRO_REJECTED);
        this->ns_loworder_reconstruct_level.assign((size_t)this->pnsSolver->iNcell, -1);
        this->ns_recon_index.assign((size_t)this->pnsSolver->iNcell, -1);
    }
    this->ns_recon_cells.clear();
    this->ns_recon_phi.clear();
    this->ns_recon_phi_new.clear();
    this->dsmc2ns_acceptance_ready = false;
    refreshNsOwnerRanges(true);
}

/*
 * ~ProcessGSIS: releases owned buffers and MPI helper state.
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
ProcessGSIS::~ProcessGSIS()
{
}

/*
 * refreshNsOwnerRanges: updates partition ownership or load data.
 * Params: force; returns: none.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
void ProcessGSIS::refreshNsOwnerRanges(bool force)
{
    if (!this->mpi || !this->mpi->active() || this->pnsSolver == nullptr || this->c_size <= 0)
        return;
    if (!force &&
        (int)this->nsStartByRank.size() == this->c_size &&
        (int)this->nsEndByRank.size() == this->c_size)
        return;
    int myRange[2] = {this->pnsSolver->Nl, this->pnsSolver->Nr};
    vector<int> allRanges((size_t)this->c_size * 2u, 0);
    MPI_Allgather(myRange, 2, MPI_INT, allRanges.data(), 2, MPI_INT, this->calGroup);
    this->nsStartByRank.assign((size_t)this->c_size, 0);
    this->nsEndByRank.assign((size_t)this->c_size, 0);
    for (int r = 0; r < this->c_size; ++r)
    {
        this->nsStartByRank[(size_t)r] = allRanges[(size_t)r * 2u + 0u];
        this->nsEndByRank[(size_t)r] = allRanges[(size_t)r * 2u + 1u];
    }
}

/*
 * nsOwnerOfGlobalCell: updates partition ownership or load data.
 * Params: globalCell; returns: index, count, owner, or status value.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
int ProcessGSIS::nsOwnerOfGlobalCell(int globalCell) const
{
    if (this->pnsSolver == nullptr || this->pnsSolver->NsreMeIndex == nullptr)
        return -1;
    if (globalCell < 0 || globalCell >= this->mess.Ncell)
        return -1;
    const int nsIndex = this->pnsSolver->NsreMeIndex[globalCell];
    for (int r = 0; r < (int)this->nsStartByRank.size() && r < (int)this->nsEndByRank.size(); ++r)
    {
        if (nsIndex >= this->nsStartByRank[(size_t)r] && nsIndex < this->nsEndByRank[(size_t)r])
            return r;
    }
    return -1;
}

/*
 * nsLocalOfGlobalCell: works with mesh topology or geometric intersections.
 * Params: globalCell; returns: index, count, owner, or status value.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
int ProcessGSIS::nsLocalOfGlobalCell(int globalCell) const
{
    if (this->pnsSolver == nullptr || this->pnsSolver->NsreMeIndex == nullptr)
        return -1;
    if (globalCell < 0 || globalCell >= this->mess.Ncell)
        return -1;
    const int nsIndex = this->pnsSolver->NsreMeIndex[globalCell];
    if (nsIndex < this->pnsSolver->Nl || nsIndex >= this->pnsSolver->Nr)
        return -1;
    return nsIndex - this->pnsSolver->Nl;
}

/*
 * dsmcOwnerOfGlobalCell: updates partition ownership or load data.
 * Params: globalCell; returns: index, count, owner, or status value.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
int ProcessGSIS::dsmcOwnerOfGlobalCell(int globalCell) const
{
    if (this->process == nullptr || globalCell < 0 || globalCell >= this->mess.Ncell)
        return -1;
    int owner = this->process->partitionState.ownerOf(globalCell);
    if (owner >= 0 && owner < this->c_size)
        return owner;
    if (globalCell < (int)this->process->rank_cell_all.size())
    {
        owner = this->process->rank_cell_all[(size_t)globalCell];
        if (owner >= 0 && owner < this->c_size)
            return owner;
    }
    return -1;
}

/*
 * dsmcLocalOfGlobalCell: works with mesh topology or geometric intersections.
 * Params: globalCell; returns: index, count, owner, or status value.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
int ProcessGSIS::dsmcLocalOfGlobalCell(int globalCell) const
{
    if (this->process == nullptr || globalCell < 0 || globalCell >= this->mess.Ncell)
        return -1;
    const int local = this->process->localOfGlobalCell(globalCell);
    if (local < 0 || local >= this->process->iNcell)
        return -1;
    return local;
}

/*
 * regsisvariables: couples DSMC data with macroscopic fields.
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
void ProcessGSIS::regsisvariables()
{
    if (!this->mpi->active()) return;
    this->icell = this->process->iNcell;
    this->nsresult.assign((size_t)this->icell * 6u, 0.0);
    this->dsmc2ns_macro_accepted.assign((size_t)this->icell, 1);
    if (this->pnsSolver != nullptr)
    {
        this->ns_dsmc2ns_macro_accepted.assign((size_t)this->pnsSolver->iNcell, 0);
        this->ns_dsmc2ns_high_order_accepted.assign((size_t)this->pnsSolver->iNcell, 0);
        this->ns_dsmc2ns_macro_source.assign((size_t)this->pnsSolver->Ncell, MACRO_REJECTED);
        this->ns_loworder_reconstruct_level.assign((size_t)this->pnsSolver->iNcell, -1);
        this->ns_recon_index.assign((size_t)this->pnsSolver->iNcell, -1);
    }
    this->ns_recon_cells.clear();
    this->ns_recon_phi.clear();
    this->ns_recon_phi_new.clear();
    this->dsmc2ns_acceptance_ready = false;
    refreshNsOwnerRanges(true);
}

/*
 * macro_iter_process: couples DSMC data with macroscopic fields.
 * Params: maxError, istep; returns: none.
 * Flow:
 *   - map ownership.
 *   - filter or reconstruct fields.
 *   - write accepted coupled state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
void ProcessGSIS::macro_iter_process(double maxError,int istep)
{
    if (!this->mpi->active()) return;
    MacroSolver *ns = this->pnsSolver;
    this->DSMC2NS();
    this->reconstructLowOrderForNS();
    MPI_Barrier(calGroup);
    ns->calcCellHot(istep);
    this->dampRejectedHighOrderForNS();
    ns->nsProcess(1000,maxError,true,istep);
    this->NS2DSMC();
    if(ifvary == 1){this->molecular_velocity_change();}
}

/*
 * NS2DSMC: couples DSMC data with macroscopic fields.
 * Params: none; returns: none.
 * Flow:
 *   - map ownership.
 *   - filter or reconstruct fields.
 *   - write accepted coupled state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
void ProcessGSIS::NS2DSMC()
{
    if (!this->mpi->active()) {return;}
    refreshNsOwnerRanges();
    this->icell = this->process->iNcell;
    this->nsresult.assign((size_t)this->icell * 6u, 0.0);
    this->dsmc2ns_macro_accepted.assign(
        (size_t)this->icell, this->dsmc2ns_acceptance_ready ? 0 : 1);
    const int resultWidth = 6;
    const int acceptedIndex = resultWidth;
    const int width = resultWidth + 1;
    vector<int> sendCounts((size_t)this->c_size, 0);
    for (int i = 0; i < this->pnsSolver->iNcell; ++i)
    {
        int global_index = this->pnsSolver->NsreMeIndex2[this->pnsSolver->Nl + i]; 
        const int dst = dsmcOwnerOfGlobalCell(global_index);
        if (dst >= 0 && dst < this->c_size)
            ++sendCounts[(size_t)dst];
    }
    vector<int> sendDispls((size_t)this->c_size, 0);
    int sendTotal = 0;
    for (int r = 0; r < this->c_size; ++r)
    {
        sendDispls[(size_t)r] = sendTotal;
        sendTotal += sendCounts[(size_t)r];
    }
    vector<int> cursor = sendDispls;
    vector<int> sendGids((size_t)sendTotal, -1);
    vector<double> sendValues((size_t)sendTotal * (size_t)width, 0.0);
    for (int i = 0; i < this->pnsSolver->iNcell; ++i)
    {
        const int global_index = this->pnsSolver->NsreMeIndex2[this->pnsSolver->Nl + i];
        const int dst = dsmcOwnerOfGlobalCell(global_index);
        if (dst < 0 || dst >= this->c_size)
            continue;
        const int pos = cursor[(size_t)dst]++;
        sendGids[(size_t)pos] = global_index;
        double *out = &sendValues[(size_t)pos * (size_t)width];
        out[0] = this->pnsSolver->Qf[i][0];
        out[1] = this->pnsSolver->Qf[i][1] / sqrt(2.0);
        out[2] = this->pnsSolver->Qf[i][2] / sqrt(2.0);
        out[3] = this->pnsSolver->Qf[i][3] / sqrt(2.0);
        out[4] = this->pnsSolver->Qf[i][4];
        out[5] = this->pnsSolver->Qf[i][5];
        out[acceptedIndex] =
            (!this->dsmc2ns_acceptance_ready ||
             ((size_t)i < this->ns_dsmc2ns_macro_accepted.size() &&
              this->ns_dsmc2ns_macro_accepted[(size_t)i] != 0))
                ? 1.0
                : 0.0;
    }
    vector<int> recvGids;
    vector<double> recvValues;
    if (!GsisPacketExchange::exchange(this->calGroup, this->c_size, width,
                                      sendCounts, sendGids, sendValues,
                                      recvGids, recvValues))
    {
        if (this->mpi->activeLeader())
            cout << "GSIS_NS2DSMC_PACKET_FAIL" << endl;
        return;
    }
    for (int p = 0; p < (int)recvGids.size(); ++p)
    {
        const int local = dsmcLocalOfGlobalCell(recvGids[(size_t)p]);
        if (local < 0)
            continue;
        const double *in = &recvValues[(size_t)p * (size_t)width];
        if (this->dsmc2ns_acceptance_ready)
        {
            this->dsmc2ns_macro_accepted[(size_t)local] =
                (in[acceptedIndex] > 0.5) ? 1 : 0;
        }
        for (int k = 0; k < resultWidth; ++k)
            this->nsresult[(size_t)local * (size_t)resultWidth + (size_t)k] =
                recvValues[(size_t)p * (size_t)width + (size_t)k];
    }
    if (this->dsmc2ns_acceptance_ready && this->dsmc2nsCoupling.logFilterSummary)
    {
        long long localAccept[3] = {0, 0, 0};
        for (int i = 0; i < this->icell; ++i)
        {
            ++localAccept[0];
            if ((size_t)i < this->dsmc2ns_macro_accepted.size() &&
                this->dsmc2ns_macro_accepted[(size_t)i] != 0)
                ++localAccept[1];
            else
                ++localAccept[2];
        }
        long long globalAccept[3] = {0, 0, 0};
        MPI_Allreduce(localAccept, globalAccept, 3, MPI_LONG_LONG, MPI_SUM, this->calGroup);
        if (this->mpi->activeLeader())
        {
            cout << "GSIS_NS2DSMC_ACCEPT_SUM"
                 << " ready=1"
                 << " total=" << globalAccept[0]
                 << " accepted=" << globalAccept[1]
                 << " rejected=" << globalAccept[2]
                 << endl;
        }
    }
}

/*
 * isNsMpiHaloNeighborForLowOrder: performs one solver support operation.
 * Params: ns, cell, faceSlot, neighbor; returns: success or decision flag.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
bool ProcessGSIS::isNsMpiHaloNeighborForLowOrder(
    const MacroSolver* ns, int cell, int faceSlot, int neighbor) const
{
    if (ns == nullptr || ns->cells == nullptr)
        return false;
    if (cell < 0 || cell >= ns->iNcell)
        return false;
    if (faceSlot < 0 || faceSlot >= NN)
        return false;
    if (neighbor < ns->iNcell || neighbor >= ns->Ncell)
        return false;
    const int face = ns->cells[cell].cell2face[faceSlot];
    if (face < 0 || face >= ns->Nface)
        return false;
    return face < ns->eNface;
}

/*
 * syncNsHaloLowOrderState: performs one solver support operation.
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
void ProcessGSIS::syncNsHaloLowOrderState()
{
    if (!this->mpi->active()) return;
    MacroSolver *ns = this->pnsSolver;
    if (ns == nullptr || ns->Qf == nullptr || ns->Qc == nullptr ||
        ns->request == nullptr || ns->status == nullptr ||
        ns->sendBuf == nullptr || ns->recBuf == nullptr ||
        ns->sendCount == nullptr || ns->recCount == nullptr)
        return;
    if ((int)this->ns_dsmc2ns_macro_source.size() < ns->Ncell)
        this->ns_dsmc2ns_macro_source.resize((size_t)ns->Ncell, MACRO_REJECTED);
    ns->packQfQc();
    MPI_Startall(2 * ns->iSize, &ns->request[0]);
    MPI_Waitall(2 * ns->iSize, &ns->request[0], &ns->status[0]);
    ns->unPackQfQc();
    const int size = ns->iSize;
    vector<int> sendCounts((size_t)size, 0);
    vector<int> recvCounts((size_t)size, 0);
    vector<int> sendDispls((size_t)size, 0);
    vector<int> recvDispls((size_t)size, 0);
    for (int r = 0; r < size; ++r)
    {
        sendCounts[(size_t)r] = ns->sendCount[r + 1] - ns->sendCount[r];
        recvCounts[(size_t)r] = ns->recCount[r + 1] - ns->recCount[r];
        sendDispls[(size_t)r] = ns->sendCount[r];
        recvDispls[(size_t)r] = ns->recCount[r];
    }
    vector<unsigned char> sendSource(ns->sendCell.size(), MACRO_REJECTED);
    vector<unsigned char> recvSource(ns->recCell.size(), MACRO_REJECTED);
    for (size_t i = 0; i < ns->sendCell.size(); ++i)
    {
        const int cell = ns->sendCell[i];
        if (cell >= 0 && cell < ns->iNcell &&
            (size_t)cell < this->ns_dsmc2ns_macro_source.size())
            sendSource[i] = this->ns_dsmc2ns_macro_source[(size_t)cell];
    }
    MPI_Alltoallv(sendSource.empty() ? nullptr : sendSource.data(),
                  sendCounts.data(), sendDispls.data(),
                  MPI_UNSIGNED_CHAR,
                  recvSource.empty() ? nullptr : recvSource.data(),
                  recvCounts.data(), recvDispls.data(),
                  MPI_UNSIGNED_CHAR, ns->myGroup);
    for (size_t i = 0; i < ns->recCell.size(); ++i)
    {
        const int cell = ns->recCell[i];
        if (cell >= ns->iNcell && cell < ns->Ncell &&
            (size_t)cell < this->ns_dsmc2ns_macro_source.size())
            this->ns_dsmc2ns_macro_source[(size_t)cell] = recvSource[i];
    }
}

/*
 * DSMC2NS: couples DSMC data with macroscopic fields.
 * Params: none; returns: none.
 * Flow:
 *   - map ownership.
 *   - filter or reconstruct fields.
 *   - write accepted coupled state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
void ProcessGSIS::DSMC2NS()
{
    if (!this->mpi->active()) return;
    MacroSolver *ns = this->pnsSolver;
    refreshNsOwnerRanges();
    this->dsmc2ns_acceptance_ready = false;
    this->dsmc2ns_macro_accepted.assign((size_t)this->icell, 1);
    this->ns_dsmc2ns_macro_accepted.assign((size_t)ns->iNcell, 0);
    this->ns_dsmc2ns_high_order_accepted.assign((size_t)ns->iNcell, 0);
    this->ns_dsmc2ns_macro_source.assign((size_t)ns->Ncell, MACRO_REJECTED);
    this->ns_loworder_reconstruct_level.assign((size_t)ns->iNcell, -1);
    this->ns_recon_index.assign((size_t)ns->iNcell, -1);
    this->ns_recon_cells.clear();
    this->ns_recon_phi.clear();
    this->ns_recon_phi_new.clear();
    if ((int)this->process->dsmc2ns_sparse_state.size() != this->icell)
        this->process->dsmc2ns_sparse_state.assign(
            (size_t)this->icell, ProcessDSMC::DSMC2NS_SPARSE_NORMAL);
    if ((int)this->process->dsmc2ns_sparse_accum_steps.size() != this->icell)
        this->process->dsmc2ns_sparse_accum_steps.assign((size_t)this->icell, 0);
    const int macroWidth = this->process->Madata;
    const int macroAcceptedIndex = macroWidth;
    const int highOrderAcceptedIndex = macroWidth + 1;
    const int macroSourceIndex = macroWidth + 2;
    const int width = macroWidth + 3;
    const bool useFinalRecord =
        EnableDsmc2NsFinalRecord &&
        (this->process->istep > NSS) &&
        (this->process->istep >= NSS + NFINAL_RECORD_DELAY);
    const vector<double>* macroSource = &this->process->local;
    const char* macroSourceName = "local";
    if (useFinalRecord)
    {
        macroSource = &this->process->final_record;
        macroSourceName = "final_record";
    }
    else if (!this->use_exp_weighted_dsmc2ns)
    {
        macroSource = &this->process->record;
        macroSourceName = "record";
    }
    vector<int> sendCounts((size_t)this->c_size, 0);
    for (int i = 0; i < this->icell; ++i)
    {
        const int global_index = this->process->local_cells[i];
        const int dst = nsOwnerOfGlobalCell(global_index);
        if (dst >= 0 && dst < this->c_size)
            ++sendCounts[(size_t)dst];
    }
    vector<int> sendDispls((size_t)this->c_size, 0);
    int sendTotal = 0;
    for (int r = 0; r < this->c_size; ++r)
    {
        sendDispls[(size_t)r] = sendTotal;
        sendTotal += sendCounts[(size_t)r];
    }
    vector<int> cursor = sendDispls;
    vector<int> sendGids((size_t)sendTotal, -1);
    vector<double> sendValues((size_t)sendTotal * (size_t)width, 0.0);
    long long localFilter[11] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < this->icell; ++i)
    {
        const int global_index = this->process->local_cells[i];
        const int dst = nsOwnerOfGlobalCell(global_index);
        if (dst < 0 || dst >= this->c_size)
            continue;
        const int pos = cursor[(size_t)dst]++;
        sendGids[(size_t)pos] = global_index;
        double *out = &sendValues[(size_t)pos * (size_t)width];
        unsigned char &sparseState =
            this->process->dsmc2ns_sparse_state[(size_t)i];
        bool sparseCell =
            (sparseState == ProcessDSMC::DSMC2NS_SPARSE_ACCUMULATING ||
             sparseState == ProcessDSMC::DSMC2NS_SPARSE_RELEASED);
        const vector<double>* cellMacroSource =
            sparseCell ? &this->process->record : macroSource;
        for (int k = 0; k < macroWidth; ++k)
            out[k] = (*cellMacroSource)[(size_t)i * (size_t)macroWidth + (size_t)k];
        double sampleCount = 0.0;
        bool sampleCountValid = false;
        if (sparseCell)
        {
            if ((size_t)i < this->process->stepinter_rho.size())
            {
                sampleCount = this->process->stepinter_rho[(size_t)i];
                sampleCountValid = std::isfinite(sampleCount);
            }
        }
        else if (useFinalRecord)
        {
            if ((size_t)i < this->process->steady_rho.size())
            {
                sampleCount = this->process->steady_rho[(size_t)i];
                sampleCountValid = std::isfinite(sampleCount);
            }
        }
        else if ((size_t)i < this->process->dsmc2ns_window_samples.size())
        {
            sampleCount = this->process->dsmc2ns_window_samples[(size_t)i];
            sampleCountValid = std::isfinite(sampleCount);
            if (this->dsmc2nsCoupling.sampleFilterEnabled &&
                (size_t)i < this->process->dsmc2ns_window_valid.size() &&
                this->process->dsmc2ns_window_valid[(size_t)i] == 0)
                sampleCountValid = false;
        }
        double previewSampleCount = sampleCountValid ? sampleCount : 0.0;
        if (!std::isfinite(previewSampleCount))
            previewSampleCount = 0.0;
        double cellVolume = 0.0;
        if ((size_t)i < this->process->cells.size())
            cellVolume = this->process->cells[(size_t)i].area;
        const Dsmc2NsMetrics metrics =
            buildDsmc2NsMetrics(out, previewSampleCount, ns, cellVolume);
        const Dsmc2NsDecision decision =
            decideDsmc2NsCoupling(this->dsmc2nsCoupling, out, metrics);
        bool macroReliable = decision.macroReliable;
        bool highOrderReliable = decision.highOrderReliable;
        if (!sparseCell &&
            (decision.lowSampleMacro || decision.knCellUpperBoundsReject))
        {
            sparseState = ProcessDSMC::DSMC2NS_SPARSE_ACCUMULATING;
            sparseCell = true;
            macroReliable = false;
            highOrderReliable = false;
        }
        else if (sparseState == ProcessDSMC::DSMC2NS_SPARSE_ACCUMULATING &&
                 macroReliable)
        {
            sparseState = ProcessDSMC::DSMC2NS_SPARSE_RELEASED;
            sparseCell = true;
        }
        out[macroAcceptedIndex] = macroReliable ? 1.0 : 0.0;
        out[highOrderAcceptedIndex] = highOrderReliable ? 1.0 : 0.0;
        out[macroSourceIndex] =
            sparseCell ? (double)MACRO_ACCUMULATED_DSMC : (double)MACRO_DIRECT_DSMC;
        localFilter[0] += macroReliable ? 0 : 1;
        localFilter[1] += highOrderReliable ? 0 : 1;
        localFilter[2] += decision.lowSampleMacro ? 1 : 0;
        localFilter[3] += decision.lowSampleHighOrder ? 1 : 0;
        localFilter[4] += decision.macroFiniteReject ? 1 : 0;
        localFilter[5] += decision.macroLowerBoundsReject ? 1 : 0;
        localFilter[6] += decision.macroUpperBoundsReject ? 1 : 0;
        localFilter[7] += decision.knCellUpperBoundsReject ? 1 : 0;
        localFilter[8] += decision.highOrderFiniteReject ? 1 : 0;
        localFilter[9] += decision.highOrderLowerBoundsReject ? 1 : 0;
        localFilter[10] += decision.highOrderUpperBoundsReject ? 1 : 0;
    }
    vector<int> recvGids;
    vector<double> recvValues;
    if (!GsisPacketExchange::exchange(this->calGroup, this->c_size, width,
                                      sendCounts, sendGids, sendValues,
                                      recvGids, recvValues))
    {
        if (this->mpi->activeLeader())
            cout << "GSIS_DSMC2NS_PACKET_FAIL" << endl;
        return;
    }
    const double sqrt2 = sqrt(2.0);
    for (int p = 0; p < (int)recvGids.size(); ++p)
    {
        const int nsLocal = nsLocalOfGlobalCell(recvGids[(size_t)p]);
        if (nsLocal < 0 || nsLocal >= ns->iNcell)
            continue;
        const double *in = &recvValues[(size_t)p * (size_t)width];
        const bool macroAccepted = (in[macroAcceptedIndex] > 0.5) &&
                                   macroLowOrderFinite(in);
        this->ns_dsmc2ns_macro_accepted[(size_t)nsLocal] =
            macroAccepted ? 1 : 0;
        if (!macroAccepted)
        {
            this->ns_dsmc2ns_macro_source[(size_t)nsLocal] = MACRO_REJECTED;
            this->ns_loworder_reconstruct_level[(size_t)nsLocal] = -1;
            if (this->ns_recon_index[(size_t)nsLocal] < 0)
            {
                this->ns_recon_index[(size_t)nsLocal] =
                    (int)this->ns_recon_cells.size();
                this->ns_recon_cells.push_back(nsLocal);
            }
            continue;
        }
        ns->Qf[nsLocal][0] = in[0];
        ns->Qf[nsLocal][1] = sqrt2 * in[1];
        ns->Qf[nsLocal][2] = sqrt2 * in[2];
        ns->Qf[nsLocal][3] = sqrt2 * in[3];
        ns->Qf[nsLocal][4] = in[4];
        ns->Qf[nsLocal][5] = in[14];
        const int sourceValue = (int)(in[macroSourceIndex] + 0.5);
        this->ns_dsmc2ns_macro_source[(size_t)nsLocal] =
            (sourceValue == MACRO_ACCUMULATED_DSMC)
                ? MACRO_ACCUMULATED_DSMC
                : MACRO_DIRECT_DSMC;
        this->ns_loworder_reconstruct_level[(size_t)nsLocal] = 0;
        const bool highOrderAccepted = (in[highOrderAcceptedIndex] > 0.5) &&
                                       macroHighOrderFinite(in);
        this->ns_dsmc2ns_high_order_accepted[(size_t)nsLocal] =
            highOrderAccepted ? 1 : 0;
        if (!highOrderAccepted)
        {
            continue;
        }
        ns->d_sigmaq[nsLocal][0] = -sqrt2 * in[11];
        ns->d_sigmaq[nsLocal][1] = -sqrt2 * in[12];
        ns->d_sigmaq[nsLocal][2] = -sqrt2 * in[13];
        ns->d_sigmaq[nsLocal][3] = -sqrt2 * in[15];
        ns->d_sigmaq[nsLocal][4] = -sqrt2 * in[16];
        ns->d_sigmaq[nsLocal][5] = -sqrt2 * in[17];
        ns->d_sigmaq[nsLocal][6] = -in[5];
        ns->d_sigmaq[nsLocal][7] = -in[6];
        ns->d_sigmaq[nsLocal][8] = -in[10];
        ns->d_sigmaq[nsLocal][9] = -in[7];
        ns->d_sigmaq[nsLocal][10] = -in[8];
        ns->d_sigmaq[nsLocal][11] = -in[9];
    }
    this->dsmc2ns_acceptance_ready = true;
    if (this->dsmc2nsCoupling.logFilterSummary)
    {
        long long globalFilter[11] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        MPI_Allreduce(localFilter, globalFilter, 11, MPI_LONG_LONG, MPI_SUM, this->calGroup);
        if (this->mpi->activeLeader())
        {
            cout << "GSIS_COUPLE_FILTER_SUM"
                 << " mode=macro_active"
                 << " highOrderMode=active"
                 << " step=" << ((this->process != nullptr) ? this->process->istep : -1)
                 << " source=" << macroSourceName
                 << " skippedMacro=" << globalFilter[0]
                 << " zeroHighOrder=" << globalFilter[1]
                 << " lowSampleMacro=" << globalFilter[2]
                 << " lowSampleHighOrder=" << globalFilter[3]
                 << " macroFiniteReject=" << globalFilter[4]
                 << " macroLowerBoundsReject=" << globalFilter[5]
                 << " macroUpperBoundsReject=" << globalFilter[6]
                 << " knCellUpperBoundsReject=" << globalFilter[7]
                 << " highOrderFiniteReject=" << globalFilter[8]
                 << " highOrderLowerBoundsReject=" << globalFilter[9]
                 << " highOrderUpperBoundsReject=" << globalFilter[10]
                 << " sampleFilter=" << (this->dsmc2nsCoupling.sampleFilterEnabled ? 1 : 0)
                 << " macroMinSamples=" << this->dsmc2nsCoupling.macroMinSampleCount
                 << " highOrderSampleFilter=" << (this->dsmc2nsCoupling.highOrderSampleFilterEnabled ? 1 : 0)
                 << " highOrderMinSamples=" << this->dsmc2nsCoupling.highOrderMinSampleCount
                 << " macroLowerBounds=" << (this->dsmc2nsCoupling.macroLowerBoundsEnabled ? 1 : 0)
                 << " macroUpperBounds=" << (this->dsmc2nsCoupling.macroUpperBoundsEnabled ? 1 : 0)
                 << " rhoMin=" << this->dsmc2nsCoupling.rhoMin
                 << " rhoMax=" << this->dsmc2nsCoupling.rhoMax
                 << " Tmin=" << this->dsmc2nsCoupling.tMin
                 << " Tmax=" << this->dsmc2nsCoupling.tMax
                 << " Trmin=" << this->dsmc2nsCoupling.trMin
                 << " Trmax=" << this->dsmc2nsCoupling.trMax
                 << " MaMin=" << this->dsmc2nsCoupling.maMin
                 << " MaMax=" << this->dsmc2nsCoupling.maMax
                 << " knCellUpperBound=" << (this->dsmc2nsCoupling.knCellUpperBoundEnabled ? 1 : 0)
                 << " knCellMax=" << this->dsmc2nsCoupling.knCellMax
                 << " highOrderLowerBounds=" << (this->dsmc2nsCoupling.highOrderLowerBoundsEnabled ? 1 : 0)
                 << " highOrderUpperBounds=" << (this->dsmc2nsCoupling.highOrderUpperBoundsEnabled ? 1 : 0)
                 << " stressRatioMin=" << this->dsmc2nsCoupling.stressRatioMin
                 << " stressRatioMax=" << this->dsmc2nsCoupling.stressRatioMax
                 << " heatRatioMin=" << this->dsmc2nsCoupling.heatRatioMin
                 << " heatRatioMax=" << this->dsmc2nsCoupling.heatRatioMax
                 << " rotHeatRatioMin=" << this->dsmc2nsCoupling.rotHeatRatioMin
                 << " rotHeatRatioMax=" << this->dsmc2nsCoupling.rotHeatRatioMax
                 << endl;
        }
    }
    ns->origin2Conservation();
}

/*
 * reconstructLowOrderForNS: couples DSMC data with macroscopic fields.
 * Params: none; returns: none.
 * Flow:
 *   - map ownership.
 *   - filter or reconstruct fields.
 *   - write accepted coupled state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
void ProcessGSIS::reconstructLowOrderForNS()
{
    if (!this->mpi->active()) return;
    if (!this->dsmc2nsCoupling.lowOrderHarmonicEnabled) return;
    MacroSolver *ns = this->pnsSolver;
    if (ns == nullptr || ns->Qf == nullptr || ns->cells == nullptr) return;
    syncNsHaloLowOrderState();
    const size_t reconCount = this->ns_recon_cells.size();
    this->ns_recon_phi.assign(reconCount, array<double, 6>());
    this->ns_recon_phi_new.assign(reconCount, array<double, 6>());
    vector<char> hasReliableNeighbor(reconCount, 0);
    vector<char> hasReliableNeighborNew(reconCount, 0);
    vector<int> firstValidLevel(reconCount, -1);
    const double rhoMin = std::max(this->dsmc2nsCoupling.rhoMin, 1.0e-300);
    const double tMin = std::max(this->dsmc2nsCoupling.tMin, 1.0e-300);
    const double trMin = std::max(this->dsmc2nsCoupling.trMin, 1.0e-300);
    const double omega = this->dsmc2nsCoupling.lowOrderHarmonicOmega;
    const int iterations = std::max(0, this->dsmc2nsCoupling.lowOrderHarmonicIterations);
    long long localHaloUsed = 0;
    long long localHaloSkipped = 0;
    for (size_t idx = 0; idx < reconCount; ++idx)
    {
        const int c = this->ns_recon_cells[idx];
        array<double, 6> &phi = this->ns_recon_phi[idx];
        if (c < 0 || c >= ns->iNcell)
            continue;
        phi[0] = std::log(std::max(ns->Qf[c][0], rhoMin));
        phi[1] = ns->Qf[c][1];
        phi[2] = ns->Qf[c][2];
        phi[3] = ns->Qf[c][3];
        phi[4] = std::log(std::max(ns->Qf[c][4], tMin));
        phi[5] = std::log(std::max(ns->Qf[c][5], trMin));
    }
    for (int iter = 0; iter < iterations; ++iter)
    {
        std::fill(hasReliableNeighborNew.begin(), hasReliableNeighborNew.end(), 0);
        for (size_t idx = 0; idx < reconCount; ++idx)
        {
            const int c = this->ns_recon_cells[idx];
            if (c < 0 || c >= ns->iNcell)
            {
                this->ns_recon_phi_new[idx] = this->ns_recon_phi[idx];
                hasReliableNeighborNew[idx] = hasReliableNeighbor[idx];
                continue;
            }
            array<double, 6> sum = array<double, 6>();
            double sumW = 0.0;
            bool reliableNeighbor = false;
            for (int f = 0; f < NN; ++f)
            {
                const int nb = ns->cells[c].cell2cell[f];
                if (nb < 0 || nb >= ns->Ncell)
                    continue;
                array<double, 6> nbPhi = array<double, 6>();
                bool useNeighbor = false;
                if (nb < ns->iNcell)
                {
                    const unsigned char source =
                        ((size_t)nb < this->ns_dsmc2ns_macro_source.size())
                            ? this->ns_dsmc2ns_macro_source[(size_t)nb]
                            : MACRO_REJECTED;
                    if (source == MACRO_DIRECT_DSMC ||
                        source == MACRO_ACCUMULATED_DSMC ||
                        source == MACRO_INTERPOLATED)
                    {
                        nbPhi[0] = std::log(std::max(ns->Qf[nb][0], rhoMin));
                        nbPhi[1] = ns->Qf[nb][1];
                        nbPhi[2] = ns->Qf[nb][2];
                        nbPhi[3] = ns->Qf[nb][3];
                        nbPhi[4] = std::log(std::max(ns->Qf[nb][4], tMin));
                        nbPhi[5] = std::log(std::max(ns->Qf[nb][5], trMin));
                        useNeighbor = true;
                        reliableNeighbor = true;
                    }
                    else if ((size_t)nb < this->ns_recon_index.size())
                    {
                        const int nbRecon = this->ns_recon_index[(size_t)nb];
                        if (nbRecon >= 0 && (size_t)nbRecon < reconCount)
                        {
                            nbPhi = this->ns_recon_phi[(size_t)nbRecon];
                            useNeighbor = true;
                            reliableNeighbor =
                                reliableNeighbor ||
                                hasReliableNeighbor[(size_t)nbRecon] != 0;
                        }
                    }
                }
                else if (isNsMpiHaloNeighborForLowOrder(ns, c, f, nb))
                {
                    const unsigned char source =
                        ((size_t)nb < this->ns_dsmc2ns_macro_source.size())
                            ? this->ns_dsmc2ns_macro_source[(size_t)nb]
                            : MACRO_REJECTED;
                    if (source == MACRO_DIRECT_DSMC ||
                        source == MACRO_ACCUMULATED_DSMC ||
                        source == MACRO_INTERPOLATED)
                    {
                        nbPhi[0] = std::log(std::max(ns->Qf[nb][0], rhoMin));
                        nbPhi[1] = ns->Qf[nb][1];
                        nbPhi[2] = ns->Qf[nb][2];
                        nbPhi[3] = ns->Qf[nb][3];
                        nbPhi[4] = std::log(std::max(ns->Qf[nb][4], tMin));
                        nbPhi[5] = std::log(std::max(ns->Qf[nb][5], trMin));
                        bool finitePhi = true;
                        for (int k = 0; k < 6; ++k)
                            finitePhi = finitePhi && std::isfinite(nbPhi[(size_t)k]);
                        if (finitePhi)
                        {
                            useNeighbor = true;
                            reliableNeighbor = true;
                            ++localHaloUsed;
                        }
                        else
                        {
                            ++localHaloSkipped;
                        }
                    }
                    else
                    {
                        ++localHaloSkipped;
                    }
                }
                else
                {
                    ++localHaloSkipped;
                }
                if (!useNeighbor)
                    continue;
                for (int k = 0; k < 6; ++k)
                    sum[(size_t)k] += nbPhi[(size_t)k];
                sumW += 1.0;
            }
            if (sumW > 0.0)
            {
                for (int k = 0; k < 6; ++k)
                {
                    const double avg = sum[(size_t)k] / sumW;
                    this->ns_recon_phi_new[idx][(size_t)k] =
                        (1.0 - omega) * this->ns_recon_phi[idx][(size_t)k] +
                        omega * avg;
                }
                hasReliableNeighborNew[idx] =
                    (hasReliableNeighbor[idx] != 0 || reliableNeighbor) ? 1 : 0;
                if (reliableNeighbor && firstValidLevel[idx] < 0)
                    firstValidLevel[idx] = iter + 1;
            }
            else
            {
                this->ns_recon_phi_new[idx] = this->ns_recon_phi[idx];
                hasReliableNeighborNew[idx] = hasReliableNeighbor[idx];
            }
        }
        this->ns_recon_phi.swap(this->ns_recon_phi_new);
        hasReliableNeighbor.swap(hasReliableNeighborNew);
    }
    long long localSourceCount[5] = {0, 0, 0, 0, 0};
    long long localInterpolated = 0;
    long long localOldNs = 0;
    bool wroteLowOrder = false;
    for (size_t idx = 0; idx < reconCount; ++idx)
    {
        const int c = this->ns_recon_cells[idx];
        if (c < 0 || c >= ns->iNcell)
            continue;
        if (hasReliableNeighbor[idx] == 0)
        {
            this->ns_dsmc2ns_macro_source[(size_t)c] = MACRO_OLD_NS;
            ++localOldNs;
            continue;
        }
        const array<double, 6> &phi = this->ns_recon_phi[idx];
        double q[6];
        q[0] = std::max(std::exp(phi[0]), rhoMin);
        q[1] = phi[1];
        q[2] = phi[2];
        q[3] = phi[3];
        q[4] = std::max(std::exp(phi[4]), tMin);
        q[5] = std::max(std::exp(phi[5]), trMin);
        bool finite = true;
        for (int k = 0; k < 6; ++k)
            finite = finite && std::isfinite(q[k]);
        if (!finite)
        {
            this->ns_dsmc2ns_macro_source[(size_t)c] = MACRO_OLD_NS;
            ++localOldNs;
            continue;
        }
        for (int k = 0; k < 6; ++k)
            ns->Qf[c][k] = q[k];
        this->ns_dsmc2ns_macro_source[(size_t)c] = MACRO_INTERPOLATED;
        this->ns_dsmc2ns_macro_accepted[(size_t)c] = 1;
        this->ns_loworder_reconstruct_level[(size_t)c] =
            std::max(1, firstValidLevel[idx]);
        ++localInterpolated;
        wroteLowOrder = true;
    }
    for (int i = 0; i < ns->iNcell; ++i)
    {
        unsigned char source = MACRO_REJECTED;
        if ((size_t)i < this->ns_dsmc2ns_macro_source.size())
            source = this->ns_dsmc2ns_macro_source[(size_t)i];
        if (source <= MACRO_OLD_NS)
            ++localSourceCount[source];
    }
    if (wroteLowOrder)
        ns->origin2Conservation();
    if (this->dsmc2nsCoupling.logFilterSummary)
    {
        long long globalSourceCount[5] = {0, 0, 0, 0, 0};
        long long globalExtra[4] = {0, 0, 0, 0};
        long long localExtra[4] = {
            localInterpolated,
            localOldNs,
            localHaloUsed,
            localHaloSkipped
        };
        MPI_Allreduce(localSourceCount, globalSourceCount, 5, MPI_LONG_LONG, MPI_SUM, this->calGroup);
        MPI_Allreduce(localExtra, globalExtra, 4, MPI_LONG_LONG, MPI_SUM, this->calGroup);
        if (this->mpi->activeLeader())
        {
            cout << "LOW_ORDER_HARMONIC"
                 << " direct=" << globalSourceCount[MACRO_DIRECT_DSMC]
                 << " accumulated=" << globalSourceCount[MACRO_ACCUMULATED_DSMC]
                 << " rejected=" << globalSourceCount[MACRO_REJECTED]
                 << " interpolated=" << globalSourceCount[MACRO_INTERPOLATED]
                 << " old_ns=" << globalSourceCount[MACRO_OLD_NS]
                 << " iter=" << iterations
                 << " omega=" << omega
                 << " newInterpolated=" << globalExtra[0]
                 << " newOldNs=" << globalExtra[1]
                 << " haloUsed=" << globalExtra[2]
                 << " haloSkipped=" << globalExtra[3]
                 << endl;
        }
    }
}

/*
 * dampRejectedHighOrderForNS: couples DSMC data with macroscopic fields.
 * Params: none; returns: none.
 * Flow:
 *   - map ownership.
 *   - filter or reconstruct fields.
 *   - write accepted coupled state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
void ProcessGSIS::dampRejectedHighOrderForNS()
{
    if (!this->mpi->active()) return;
    MacroSolver *ns = this->pnsSolver;
    if (ns == nullptr || ns->d_sigmaq == nullptr) return;
    long long localStats[3] = {0, 0, 0};
    for (int i = 0; i < ns->iNcell; ++i)
    {
        const bool highAccepted =
            (size_t)i < this->ns_dsmc2ns_high_order_accepted.size() &&
            this->ns_dsmc2ns_high_order_accepted[(size_t)i] != 0;
        const unsigned char source =
            ((size_t)i < this->ns_dsmc2ns_macro_source.size())
                ? this->ns_dsmc2ns_macro_source[(size_t)i]
                : MACRO_REJECTED;
        const bool lowDirect =
            source == MACRO_DIRECT_DSMC ||
            source == MACRO_ACCUMULATED_DSMC;
        if (highAccepted && lowDirect)
        {
            ++localStats[0];
            continue;
        }
        double alpha = this->dsmc2nsCoupling.highOrderDampingCoeff;
        if (source == MACRO_INTERPOLATED)
        {
            int level = 1;
            if ((size_t)i < this->ns_loworder_reconstruct_level.size())
                level = std::max(1, this->ns_loworder_reconstruct_level[(size_t)i]);
            alpha = alpha / (double)level;
        }
        if (source == MACRO_OLD_NS || source == MACRO_REJECTED)
            alpha = 0.0;
        if (alpha == 0.0)
            ++localStats[2];
        else
            ++localStats[1];
        for (int k = 0; k < VAR2; ++k)
        {
            if (std::isfinite(ns->d_sigmaq[i][k]))
                ns->d_sigmaq[i][k] *= alpha;
            else
                ns->d_sigmaq[i][k] = 0.0;
        }
    }
    if (this->dsmc2nsCoupling.logFilterSummary)
    {
        long long globalStats[3] = {0, 0, 0};
        MPI_Allreduce(localStats, globalStats, 3, MPI_LONG_LONG, MPI_SUM, this->calGroup);
        if (this->mpi->activeLeader())
        {
            cout << "HIGH_ORDER_DAMP"
                 << " accepted=" << globalStats[0]
                 << " damped=" << globalStats[1]
                 << " zero=" << globalStats[2]
                 << " alpha=" << this->dsmc2nsCoupling.highOrderDampingCoeff
                 << endl;
        }
    }
}

/*
 * molecular_velocity_change: updates particles or particle-derived state.
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
void ProcessGSIS::molecular_velocity_change()
{
    if (!this->mpi->active()) return;
    MeshparticalInitial *partinit = this->partinit;
    double Neff = this->mess.Neff;
    int ns_ifnan = 0;
    const bool useAcceptanceGate = this->dsmc2ns_acceptance_ready;
    long long skippedByDsmc2NsReject = 0;
    for (int i = 0; i < this->icell; i++)
    {
        if (useAcceptanceGate &&
            ((size_t)i >= this->dsmc2ns_macro_accepted.size() ||
             this->dsmc2ns_macro_accepted[(size_t)i] == 0))
        {
            ++skippedByDsmc2NsReject;
            continue;
        }
        int meshindex = this->process->local_cells[i];
        ParticleBucketSoA &bucket = this->process->currParticles(meshindex);
        double n_dsmc, n_ns, n_change = 0.0,area;
        double u_ns[3], u_change[3] = {0,0,0};
        double T_ns, T_change = 0.0;
        double Tr_ns,  Tr_change = 0.0;
        double k=1.0, b_list[3]={0.0,0.0,0.0};
        double kr = 1.0;
        if (isnan(this->nsresult[i*6 + 0]) || isnan(this->nsresult[i*6 + 1]) || isnan(this->nsresult[i*6 + 2]) || isnan(this->nsresult[i*6 + 3]))
        {
            ns_ifnan = 1;break;
        }
        area = this->process->cells[i].area;
        n_dsmc = static_cast<double>(bucket.size());
        n_dsmc = static_cast<double>(n_dsmc);
        n_ns = this->nsresult[i * 6 + 0]*(area*this->mess.n_ref)/Neff;
        n_ns = static_cast<double>(partinit->Iround(n_ns));
        u_ns[0] = this->nsresult[i * 6 + 1];u_ns[1] = this->nsresult[i * 6 + 2];u_ns[2] = this->nsresult[i * 6 + 3]; 
        T_ns = this->nsresult[i * 6 + 4]; Tr_ns = this->nsresult[i * 6 + 5];
        if (n_ns > n_dsmc)
        {   
            int np_modify = static_cast<int>(n_ns - n_dsmc); 
            replication(i, np_modify, u_ns, T_ns, Tr_ns);    
        }
        else if(n_ns < n_dsmc) 
        {
            int np_modify = static_cast<int>(n_dsmc - n_ns);
            deletion(i,np_modify);      
        }
        const size_t bucketSize = bucket.size();
        for (size_t pi = 0; pi < bucketSize; ++pi)
        {
            n_change += 1.0;
            u_change[0] += bucket.vx[pi];
            u_change[1] += bucket.vy[pi];
            u_change[2] += bucket.vz[pi];
            T_change += (bucket.vx[pi]*bucket.vx[pi]+bucket.vy[pi]*bucket.vy[pi]+bucket.vz[pi]*bucket.vz[pi]);
            Tr_change += bucket.p_Ir[pi];
        }
        double Np = n_change;
        if (static_cast<int>(Np) != 0)
        {
            n_change = Neff/area/this->mess.n_ref*Np;
            u_change[0] = u_change[0]/Np;u_change[1] = u_change[1]/Np;u_change[2] = u_change[2]/Np;
            double u2 = u_change[0]*u_change[0]+u_change[1]*u_change[1]+u_change[2]*u_change[2];
            T_change = 2.0/(3.0*Np)*(T_change-Np*u2);
            Tr_change = 4.0/this->mess.dr/Np*Tr_change; 
        }
        if (T_change>0 && T_ns > 0 && Tr_change > 0)
        {
            k = sqrt(T_ns/T_change);
            b_list[0] = u_ns[0] - k * u_change[0];
            b_list[1] = u_ns[1] - k * u_change[1];
            b_list[2] = u_ns[2] - k * u_change[2];
            kr = Tr_ns/Tr_change;
            if(isnan(k) || isnan(b_list[0]) || isnan(b_list[1]) || isnan(b_list[2]))
            {
                cout << "T_ns = " << T_ns << "  T_change = " << T_change << endl;
            }
        }
        else
        {
            k = 1.0;
            kr = 1.0;
            b_list[0] = 0.0;b_list[1] = 0.0;b_list[2] = 0.0;
        }
        for (size_t pi = 0; pi < bucket.size(); ++pi)
        {
            bucket.vx[pi] = k*bucket.vx[pi]+b_list[0];
            bucket.vy[pi] = k*bucket.vy[pi]+b_list[1];
            bucket.vz[pi] = k*bucket.vz[pi]+b_list[2];
            if (isnan(bucket.vx[pi]) ||  isnan(bucket.vy[pi]) || isnan(bucket.vz[pi]))
            {
                cout << "particle velocity nan!" << endl;
            }
            bucket.p_Ir[pi] = kr * bucket.p_Ir[pi];
            bucket.p_mesh_serial[pi] = i;
            bucket.p_rank_serial[pi] = this->c_rank;
            bucket.p_serial[pi] = (int)pi;
        }
    }
    long long globalSkippedByDsmc2NsReject = 0;
    if (useAcceptanceGate)
    {
        MPI_Allreduce(&skippedByDsmc2NsReject, &globalSkippedByDsmc2NsReject,
                      1, MPI_LONG_LONG, MPI_SUM, this->calGroup);
    }
    if (ns_ifnan == 0 && this->mpi->activeLeader())
    {
        cout << "The molecule velocities have been shifted";
        if (useAcceptanceGate)
        {
            cout << " (acceptanceGate=on skipped="
                 << globalSkippedByDsmc2NsReject << ")";
        }
        cout << endl;
    }
    else {if (this->mpi->activeLeader()){cout << "The solution from ns solver is nan, particle velocities have not been shifted!" << endl;}}
}

/*
 * replication: performs one solver support operation.
 * Params: i, np_modify, u_ns, T_ns, Tr_ns; returns: success or decision flag.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
bool ProcessGSIS::replication(int i, int np_modify, double *u_ns, double T_ns, double Tr_ns)
{
    auto& gen = partinit->thread_rng();
    double u[3], T, Tr;
    u[0] = u_ns[0];u[1] = u_ns[1]; u[2] = u_ns[2]; T = T_ns; Tr = Tr_ns;
    int meshindex = this->process->local_cells[i];
    ParticleBucketSoA &bucket = this->process->currParticles(meshindex);
    int Np_cell = (int)bucket.size();
    if (Np_cell == 0)
    {
        if (T_ns < 0) {return false;}
        else{
            for (int j = 0; j< np_modify;j++)
            {
                particle newpart;
                newpart = MaxwellianSample(newpart, u, T, Tr, i);
                newpart.p_serial = (int)bucket.size();
                newpart.p_mesh_serial = i;
                newpart.p_rank_serial = this->c_rank;
                bucket.push_back(newpart);
            }
        }
    }
    else
    {
        for (int j = 0; j<np_modify;j++)
        {
            particle newpart;
            if (Np_cell == 1)
            {
                newpart = bucket.get(0);
            }
            else
            {
                uniform_int_distribution<int> dist(0, Np_cell - 1); 
                int kp1 = dist(gen);
                newpart = bucket.get((size_t)kp1);
            }
            newpart = randomlocation(newpart, i);
            newpart.p_serial = (int)bucket.size();
            newpart.p_mesh_serial = i;
            newpart.p_rank_serial = this->c_rank;
            bucket.push_back(newpart);
            Np_cell = (int)bucket.size();
        }
    }
    return true;
}

/*
 * MaxwellianSample: updates particles or particle-derived state.
 * Params: part, u, T, Tr, i; returns: updated particle.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
particle ProcessGSIS::MaxwellianSample(particle part, double *u, double T, double Tr, int i)
{
    auto& gen = partinit->thread_rng();
    auto& dis = partinit->get_uniform();
    double r, theta;
    part.p_serial = 0;
    r = dis(gen);
    theta = 2*M_PI*dis(gen);
    part.p_velocity[0] = sqrt(T)*sqrt(-log(r))*sin(theta)+u[0];
    r = dis(gen);
    theta = 2*M_PI*dis(gen);
    part.p_velocity[1] = sqrt(T)*sqrt(-log(r))*sin(theta)+u[1];
    r = dis(gen);
    theta = 2*M_PI*dis(gen);
    part.p_velocity[2] = sqrt(T)*sqrt(-log(r))*sin(theta)+u[2];
    r = dis(gen);
    part.p_Ir = -log(r) * 0.5 * Tr;
    part.dt_left = 0;
    part = randomlocation(part, i);
    return part;
}

/*
 * randomlocation: updates particles or particle-derived state.
 * Params: part, i; returns: updated particle.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
particle ProcessGSIS::randomlocation(particle part, int i)
{
    auto& gen = partinit->thread_rng();
    auto& dis = partinit->get_uniform();
    if (!sampleCellLocation(*this->process, i, dis, gen, part.p_location))
    {
        const double* center = this->process->cells[(std::size_t)i].cellXY;
        for (int dim = 0; dim < DIM; ++dim)
        {
            part.p_location[dim] = center[dim];
        }
    }
    return part;
}

/*
 * deletion: performs one solver support operation.
 * Params: i, np_modify; returns: success or decision flag.
 * Flow:
 *   - check inputs.
 *   - process local arrays.
 *   - update outputs or member state.
 * Side effects are kept in object fields, buffers, files, or MPI-visible data.
 * Callers are expected to provide valid local/global indices and initialized solver state.
 * Uses the current local/global ownership maps prepared by initialization.
 * Keeps the existing array layout and updates only the documented outputs.
 */
bool ProcessGSIS::deletion(int i, int np_modify)
{
    auto& gen = partinit->thread_rng();
    int meshindex = this->process->local_cells[i];
    ParticleBucketSoA &bucket = this->process->currParticles(meshindex);
    int Np_cell = (int)bucket.size();
    int Np_local = Np_cell;
    if (np_modify > Np_cell || bucket.empty()) {cout << "ERROR: Deletion: The partilce numbers here should not be zero!" << endl;return false;}
    for (int j = 0; j < np_modify; j ++)
    {
        uniform_int_distribution<int> dist(0, Np_local-1);
        const int kp1 = dist(gen);
        const size_t deleteIndex = (size_t)kp1;
        const size_t lastIndex = bucket.size() - 1u;
        if (deleteIndex != lastIndex)
        {
            bucket.p_serial[deleteIndex] = bucket.p_serial[lastIndex];
            bucket.p_rank_serial[deleteIndex] = bucket.p_rank_serial[lastIndex];
            bucket.p_mesh_serial[deleteIndex] = bucket.p_mesh_serial[lastIndex];
            bucket.vx[deleteIndex] = bucket.vx[lastIndex];
            bucket.vy[deleteIndex] = bucket.vy[lastIndex];
            bucket.vz[deleteIndex] = bucket.vz[lastIndex];
            bucket.px[deleteIndex] = bucket.px[lastIndex];
            bucket.py[deleteIndex] = bucket.py[lastIndex];
            bucket.pz[deleteIndex] = bucket.pz[lastIndex];
            bucket.p_Ir[deleteIndex] = bucket.p_Ir[lastIndex];
            bucket.dt_left[deleteIndex] = bucket.dt_left[lastIndex];
        }
        bucket.pop_back();
        Np_local -= 1;
    }
    for (size_t pi = 0; pi < bucket.size(); ++pi)
    {
        bucket.p_serial[pi] = (int)pi;
        bucket.p_rank_serial[pi] = this->c_rank;
        bucket.p_mesh_serial[pi] = i;
    }
    return true;
}
