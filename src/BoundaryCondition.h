/*
 * Boundary roles and default tag-to-model mappings used by NS and DSMC.
 */

#pragma once

#include <vector>

// BoundaryRole stores state used by this module.
enum class BoundaryRole : unsigned char
{
    // No boundary behavior has been assigned.
    None,
    // Coupling interface between solver regions.
    Interface,
    // Flow enters the computational domain.
    Inlet,
    // Flow leaves the computational domain.
    Outlet,
    // Solid surface boundary.
    Wall,
    // Mirror/symmetry plane boundary.
    Symmetry
};

// BoundaryModel stores state used by this module.
enum class BoundaryModel : unsigned char
{
    // Model-free placeholder for inactive boundaries.
    None,
    // Prescribed freestream state at an inlet.
    FreestreamInlet,
    // Inlet controlled by pressure data.
    PressureInlet,
    // Outlet that allows flow to leave naturally.
    OpenOutlet,
    // Outlet controlled by pressure data.
    PressureOutlet,
    // Wall with fixed temperature.
    IsothermalWall,
    // DSMC diffuse reflection wall.
    DiffuseWall,
    // DSMC specular reflection wall.
    SpecularWall
};

// BoundaryCondition stores state used by this module.
struct BoundaryCondition
{
    // Fluent boundary tag used as the lookup key.
    int tag = -1;
    // Macroscopic solver role and model for this tag.
    BoundaryRole nsRole = BoundaryRole::None;
    BoundaryModel nsModel = BoundaryModel::None;
    // DSMC role and model for the same physical boundary.
    BoundaryRole dsmcRole = BoundaryRole::None;
    BoundaryModel dsmcModel = BoundaryModel::None;
    // Reservoir values used by inlet/outlet DSMC particle treatment.
    bool injectParticles = false;
    double pressure = 0.0;
    double rho = 1.0;
    double ux = 0.0;
    double uy = 0.0;
    double uz = 0.0;
    double temperature = 1.0;
    double freestreamUxScale = 1.0;
};

// BoundaryConditionTable stores state used by this module.
class BoundaryConditionTable
{
public:
// BoundaryConditionTable: installs the built-in Apollo boundary map.
    BoundaryConditionTable()
    {
        loadDefaults();
    }
// loadDefaults: defines interface, inlet, outlet, and wall defaults.
    void loadDefaults()
    {
        conditions.clear();
        // Tag 3 is the NS/DSMC interface and does not inject particles.
        BoundaryCondition interfaceBc;
        interfaceBc.tag = 3;
        interfaceBc.nsRole = BoundaryRole::Interface;
        interfaceBc.nsModel = BoundaryModel::None;
        interfaceBc.dsmcRole = BoundaryRole::Interface;
        interfaceBc.dsmcModel = BoundaryModel::None;
        addOrReplace(interfaceBc);
        // Tag 4 is a freestream inlet for both solvers.
        BoundaryCondition inletBc;
        inletBc.tag = 4;
        inletBc.nsRole = BoundaryRole::Inlet;
        inletBc.nsModel = BoundaryModel::FreestreamInlet;
        inletBc.dsmcRole = BoundaryRole::Inlet;
        inletBc.dsmcModel = BoundaryModel::FreestreamInlet;
        inletBc.injectParticles = true;
        addOrReplace(inletBc);
        // Tag 5 is an open outlet without particle injection.
        BoundaryCondition outletBc;
        outletBc.tag = 5;
        outletBc.nsRole = BoundaryRole::Outlet;
        outletBc.nsModel = BoundaryModel::OpenOutlet;
        outletBc.dsmcRole = BoundaryRole::Outlet;
        outletBc.dsmcModel = BoundaryModel::OpenOutlet;
        outletBc.injectParticles = false;
        addOrReplace(outletBc);
        // Tag 6 is an isothermal diffuse wall.
        BoundaryCondition wallBc;
        wallBc.tag = 6;
        wallBc.nsRole = BoundaryRole::Wall;
        wallBc.nsModel = BoundaryModel::IsothermalWall;
        wallBc.dsmcRole = BoundaryRole::Wall;
        wallBc.dsmcModel = BoundaryModel::DiffuseWall;
        wallBc.temperature = 1.0;
        addOrReplace(wallBc);
    }
// addOrReplace (bc): updates an existing tag or appends a new one.
    void addOrReplace(const BoundaryCondition& bc)
    {
        for (BoundaryCondition& item : conditions)
        {
            if (item.tag == bc.tag)
            {
                item = bc;
                return;
            }
        }
        conditions.push_back(bc);
    }
// byTag (tag): returns the configured boundary or the fallback entry.
    const BoundaryCondition& byTag(int tag) const
    {
        for (const BoundaryCondition& item : conditions)
        {
            if (item.tag == tag)
                return item;
        }
        return fallback;
    }
private:
    std::vector<BoundaryCondition> conditions;
    BoundaryCondition fallback;
};
