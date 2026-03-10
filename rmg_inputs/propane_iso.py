# RMG input file for propane pyrolysis isotopologue simulation.
#
# Conditions from the RMG-Py isotope supplement:
#   T = 850°C (1123 K), P = 2 bar, C3H8 = 1.0 mol fraction
#   Residence time: 85 ms
#
# Run with:
#   python vendor/RMG-Py/scripts/isotopes.py rmg_inputs/propane_iso.py \
#       --output iso_propane \
#       --maximumIsotopicAtoms 1 \
#       --useOriginalReactions

database(
    thermoLibraries=['primaryThermoLibrary'],
    reactionLibraries=[],
    seedMechanisms=[],
    kineticsDepositories=['training'],
    kineticsFamilies='default',
    kineticsEstimator='rate rules',
)

species(
    label='C3H8',
    reactive=True,
    structure=SMILES("CCC"),
)
species(
    label='CH3',
    reactive=True,
    structure=SMILES("[CH3]"),
)
species(
    label='C2H5',
    reactive=True,
    structure=SMILES("C[CH2]"),
)
species(
    label='CH4',
    reactive=True,
    structure=SMILES("C"),
)
species(
    label='C2H4',
    reactive=True,
    structure=SMILES("C=C"),
)
species(
    label='C2H6',
    reactive=True,
    structure=SMILES("CC"),
)
species(
    label='H',
    reactive=True,
    structure=SMILES("[H]"),
)
species(
    label='H2',
    reactive=True,
    structure=SMILES("[H][H]"),
)

simpleReactor(
    temperature=(1123, 'K'),
    pressure=(2.0, 'bar'),
    initialMoleFractions={
        'C3H8': 1.0,
    },
    terminationTime=(0.085, 's'),
)

simulator(
    atol=1e-16,
    rtol=1e-8,
)

model(
    toleranceKeepInEdge=0.0,
    toleranceMoveToCore=0.01,
    toleranceInterruptSimulation=0.1,
    maximumEdgeSpecies=100000,
)

options(
    units='si',
    generateOutputHTML=False,
    generatePlots=False,
    saveEdgeSpecies=False,
    saveSimulationProfiles=False,
)
