"""
Replication of Goldman et al. (2019) paper figures.

Adapted from code/create_paper_figures.py for:
  - Cantera 3.x  (.yaml mechanism files, reactor.phase API)
  - pandas 2.0+  (no DataFrame.append, no iteritems)

Run from the repository root:
    uv run python rmg_inputs/goldmanm-RMG_isotopes_paper_data-234bd52/replicate_figures.py

Output figures saved to:
    rmg_inputs/goldmanm-RMG_isotopes_paper_data-234bd52/results/
"""

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
import matplotlib
matplotlib.use("Agg")   # non-interactive backend – safe for scripts
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.lines import Line2D
import cantera as ct

# ── paths ─────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
MECH_DIR  = os.path.join(HERE, "mechanisms")
EXP_DIR   = os.path.join(HERE, "exp_data")
IMAGE_DIR = os.path.join(HERE, "results")
os.makedirs(IMAGE_DIR, exist_ok=True)

# ── plot style (matches original) ─────────────────────────────────────────────
sns.set_palette("colorblind", n_colors=4)
sns.set_style("white")
sns.set_context("paper", font_scale=1.5)
sns.set_style("ticks", {"ytick.direction": "in", "xtick.direction": "in"})
mpl.rcParams["xtick.top"] = True
mpl.rcParams["ytick.right"] = True

# ── Cantera helpers (Cantera 3.x + pandas 2.0 compatible) ─────────────────────

def _reactor_phase(reactor):
    """Return the Solution (phase) object from a reactor, supporting both
    old (reactor.kinetics) and new (reactor.phase) Cantera 3.x API."""
    return getattr(reactor, "phase", None) or reactor.kinetics


def _get_conditions(simulator, reactor, solution):
    return pd.Series({
        "time (s)": simulator.time,
        "temperature (K)": solution.T,
        "pressure (Pa)": solution.P,
        "density (kmol/m3)": solution.density_mole,
        "volume (m3)": reactor.volume,
        "enthalpy (J/kg)": solution.enthalpy_mass,
        "internal energy (J/kg)": solution.int_energy_mass,
    })


def _get_species(solution):
    mfrac = solution.mole_fraction_dict()
    rho = solution.density_mole
    return pd.Series({name: mfrac.get(name, 0.0) * rho
                      for name in solution.species_names})


def run_simulation(solution, times, conditions=None,
                   condition_type="constant-temperature-and-pressure",
                   output_species=True, output_reactions=False,
                   atol=1e-15, rtol=1e-9):
    """Run a Cantera simulation at fixed time points."""
    if conditions is not None:
        solution.TPX = conditions
    if condition_type == "constant-temperature-and-pressure":
        reactor = ct.IdealGasConstPressureReactor(solution, energy="off")
    elif condition_type == "adiabatic-constant-volume":
        reactor = ct.IdealGasReactor(solution)
    elif condition_type == "constant-temperature-and-volume":
        reactor = ct.IdealGasReactor(solution, energy="off")
    else:
        raise NotImplementedError(condition_type)

    simulator = ct.ReactorNet([reactor])
    sol = _reactor_phase(reactor)
    simulator.atol = atol
    simulator.rtol = rtol

    cond_rows, spec_rows = [], []
    for t in times:
        simulator.advance(t)
        cond_rows.append(_get_conditions(simulator, reactor, sol))
        if output_species:
            spec_rows.append(_get_species(sol))

    cond_df = pd.DataFrame(cond_rows)
    time_vec = cond_df["time (s)"]
    cond_df.index = time_vec
    out = {"conditions": cond_df}
    if output_species:
        spec_df = pd.DataFrame(spec_rows)
        spec_df.index = time_vec
        out["species"] = spec_df
    return out


def run_simulation_till_conversion(solution, species, conversion,
                                   conditions=None,
                                   condition_type="constant-temperature-and-pressure",
                                   output_species=True, output_reactions=False,
                                   skip_data=150, atol=1e-15, rtol=1e-9):
    """Run a Cantera simulation until a target conversion is reached."""
    if conditions is not None:
        solution.TPX = conditions
    if condition_type == "constant-temperature-and-pressure":
        reactor = ct.IdealGasConstPressureReactor(solution, energy="off")
    elif condition_type == "adiabatic-constant-volume":
        reactor = ct.IdealGasReactor(solution)
    else:
        raise NotImplementedError(condition_type)

    simulator = ct.ReactorNet([reactor])
    sol = _reactor_phase(reactor)
    simulator.atol = atol
    simulator.rtol = rtol

    if isinstance(species, str):
        target_idx = [sol.species_index(species)]
    else:
        target_idx = [sol.species_index(s) for s in species]
    start_conc = sum(sol.concentrations[i] for i in target_idx)

    cond_rows, spec_rows = [], []
    skip_count = skip_data + 1
    reached = False
    while not reached:
        simulator.step()
        new_conv = 1.0 - sum(sol.concentrations[i] for i in target_idx) / start_conc
        if new_conv >= conversion:
            reached = True
        if skip_count >= skip_data or reached:
            skip_count = 0
            cond_rows.append(_get_conditions(simulator, reactor, sol))
            if output_species:
                spec_rows.append(_get_species(sol))
        skip_count += 1

    cond_df = pd.DataFrame(cond_rows)
    time_vec = cond_df["time (s)"]
    cond_df.index = time_vec
    out = {"conditions": cond_df}
    if output_species:
        spec_df = pd.DataFrame(spec_rows)
        spec_df.index = time_vec
        out["species"] = spec_df
    return out


# ── isotope analysis helpers ──────────────────────────────────────────────────

REFERENCE_RATIO = 0.011115  # NGS-2 Propane, Hut et al. 1987 (used in Gilbert 2016)


def delta_from_fraction(desired_delta, reference_ratio=REFERENCE_RATIO):
    ratio = (desired_delta / 1000.0 + 1) * reference_ratio
    return ratio / (1.0 + ratio)


def isotope_ratio(conc, labeled, unlabeled):
    """Ratio of labeled to unlabeled atoms across all isotopologues."""
    num = (conc * labeled).sum()
    den = (conc * unlabeled).sum()
    return num / den


def get_delta(conc, labeled, unlabeled, reference_ratio=REFERENCE_RATIO):
    return (isotope_ratio(conc, labeled, unlabeled) / reference_ratio - 1.0) * 1000.0


def get_psie(conc, info, type_1, type_2, reference_ratio=REFERENCE_RATIO):
    d1 = get_delta(conc, info[f"{type_1}_enriched"], info[f"{type_1}_unenriched"],
                   reference_ratio)
    d2 = get_delta(conc, info[f"{type_2}_enriched"], info[f"{type_2}_unenriched"],
                   reference_ratio)
    return d1 - d2


# ── model configuration (same as original) ────────────────────────────────────

MODEL_CLUSTERS = {
    "full": {"propane": 26, "ethyl": 25, "methyl": 24, "ethene": 22, "H-atom": 23,
             "n-propyl": 20, "methane": 19, "ethane": 18, "ethenyl": 17,
             "ethyne": 16, "propene": 14},
    "drg":  {"propane": 8,  "ethyl": 6,  "methyl": 5,  "ethene": 2,  "H-atom": 4,
             "n-propyl": 3, "methane": 1, "ethane": 7},
    "3rxn": {"propane": 5,  "ethyl": 3,  "methyl": 2,  "ethene": 1,
             "methane": 0, "ethane": 4},
    "6rxn": {"propane": 7,  "ethyl": 5,  "methyl": 4,  "ethene": 2,  "H-atom": 3,
             "methane": 1, "ethane": 6},
}

MAIN_PATHS = [
    ("full", os.path.join(MECH_DIR, "full_model")),
    ("drg",  os.path.join(MECH_DIR, "drg_model")),
    ("3rxn", os.path.join(MECH_DIR, "three_reaction_model")),
    ("6rxn", os.path.join(MECH_DIR, "six_reaction_model")),
]

# initial propane isotopologue concentrations (same as original)
delta_total = -28
psia = 5.4
edge_delta   = delta_total + psia / 3.0
center_delta = delta_total - 2.0 * psia / 3.0
edge_frac   = delta_from_fraction(edge_delta)
center_frac = delta_from_fraction(center_delta)
f_prop = 0.0049

INIT_MOLE_FRACS = {
    "CCC":   f_prop * (1-center_frac) * (1-edge_frac)**2,
    "CCC-2": f_prop * center_frac * edge_frac**2,
    "CCC-3": f_prop * edge_frac**2 * (1-center_frac),
    "CCC-4": f_prop * 2*edge_frac * (1-edge_frac) * center_frac,
    "CCC-5": f_prop * 2*edge_frac * (1-center_frac) * (1-edge_frac),
    "CCC-6": f_prop * center_frac * (1-edge_frac)**2,
    "[He]":  1 - f_prop,
}


# ── Figures 2 & 3 ─────────────────────────────────────────────────────────────
print("Creating figures 2 and 3 ...")

enrichment_results = []
concentrations = {}
ethyl_psie_all = pd.DataFrame()

for name, path in MAIN_PATHS:
    cluster_info = pd.read_csv(os.path.join(path, "isotopomer_cluster_info.csv"),
                               index_col="name")
    mol_clusters = MODEL_CLUSTERS[name]
    temp = 850 + 273
    times = np.linspace(1e-4, 95.0 / temp, 100)

    solution = ct.Solution(os.path.join(path, "chem.yaml"))
    output = run_simulation(solution, times,
                            conditions=(temp, 2e5, INIT_MOLE_FRACS),
                            condition_type="constant-temperature-and-pressure",
                            output_species=True, output_reactions=False)
    species = output["species"]

    delta_enrichments = pd.DataFrame(columns=list(mol_clusters.keys()), index=times,
                                     dtype=float)
    concentration_data = pd.DataFrame(columns=list(mol_clusters.keys()), index=times,
                                      dtype=float)
    ethyl_psie = pd.Series(dtype=float)

    for t in times:
        for molecule, cnum in mol_clusters.items():
            labels = cluster_info[cluster_info.cluster_number == cnum].index
            conc_row = species.loc[t, labels]
            concentration_data.loc[t, molecule] = conc_row.sum()
            if molecule != "H-atom" and not np.isclose(conc_row.sum(), 0, atol=1e-40):
                delta_enrichments.loc[t, molecule] = get_delta(
                    conc_row,
                    cluster_info.loc[labels, "enriched_atoms"],
                    cluster_info.loc[labels, "unenriched_atoms"],
                )
            if molecule == "ethyl":
                ethyl_psie[t] = get_psie(conc_row, cluster_info.loc[labels, :],
                                         "r", "not_r")

    concentrations[name] = concentration_data
    ethyl_psie_all[name] = ethyl_psie
    enrichment_results.append((name, delta_enrichments))

# Figure 2 – ethene enrichment over time
f, ax = plt.subplots()
for identifier, enrichments in enrichment_results:
    enrichments.plot(y="ethene", ax=ax)
ax.set_ylabel("ethene\n$^{13}\\delta$C\n(‰)", rotation="horizontal", labelpad=30)
ax.set_xlabel("time (s)")
ax.annotate("6 reaction model", (.22, .4), xycoords="axes fraction")
ax.annotate("full model", (.7, .7), xycoords="axes fraction", rotation=3.5)
ax.annotate("3 reaction model", (.2, .88), xycoords="axes fraction", rotation=3.5)
ax.annotate("DRG model", (0.7, .82), xycoords="axes fraction", rotation=3)
ax.legend([])
plt.savefig(os.path.join(IMAGE_DIR, "figure2_ethene_enrichment.pdf"), bbox_inches="tight")
plt.close()
print("  Saved figure2_ethene_enrichment.pdf")

# Figure 3 – ethyl PSIA over time
f, ax = plt.subplots()
for col in ethyl_psie_all.columns:
    ethyl_psie_all[col].plot(ax=ax)
ax.set_ylabel("ethyl\nPSIA\n(‰)", rotation="horizontal", labelpad=20)
ax.set_xlabel("time (s)")
ax.annotate("6 reaction model", (.15, .95), (.2, .7), xycoords="axes fraction",
            textcoords="axes fraction", arrowprops={"arrowstyle": "-"})
ax.annotate("full model", (.4, .91), (.4, .8), xycoords="axes fraction",
            textcoords="axes fraction", arrowprops={"arrowstyle": "-"})
ax.annotate("DRG model", (.65, .89), (.6, .7), xycoords="axes fraction",
            textcoords="axes fraction", arrowprops={"arrowstyle": "-"})
ax.annotate("3 reaction model", (.2, .5), xycoords="axes fraction")
ax.set_ylim(-21, 1)
ax.legend([])
plt.savefig(os.path.join(IMAGE_DIR, "figure3_ethyl_psie.pdf"), bbox_inches="tight")
plt.close()
print("  Saved figure3_ethyl_psie.pdf")


# ── Figure 4 – mole fractions vs temperature ──────────────────────────────────
print("Creating figure 4 ...")

model_name, model_path = MAIN_PATHS[0]   # full model
mol_clusters = MODEL_CLUSTERS[model_name]
isotopomer_info = pd.read_csv(os.path.join(model_path, "isotopomer_cluster_info.csv"),
                               index_col="name")

mole_fractions = pd.DataFrame(index=list(mol_clusters.keys()))
for temp in (750, 800, 850, 900, 950):
    solution = ct.Solution(os.path.join(model_path, "chem.yaml"))
    t_final = 95.0 / temp
    output = run_simulation(solution, [0, t_final],
                            conditions=(temp + 273, 2e5, INIT_MOLE_FRACS),
                            condition_type="constant-temperature-and-pressure",
                            output_species=True, output_reactions=False)
    species_row = output["species"].iloc[-1]
    isotopomer_info["conc"] = species_row

    spec_conc = {}
    for molecule, cnum in mol_clusters.items():
        labels = isotopomer_info[isotopomer_info.cluster_number == cnum].index
        n_atoms = (isotopomer_info.loc[labels[0], "enriched_atoms"] +
                   isotopomer_info.loc[labels[0], "unenriched_atoms"])
        spec_conc[molecule] = isotopomer_info.loc[labels, "conc"].sum() * n_atoms

    total = sum(
        isotopomer_info.loc[idx, "conc"] * (
            isotopomer_info.loc[idx, "enriched_atoms"] +
            isotopomer_info.loc[idx, "unenriched_atoms"]
        )
        for idx in isotopomer_info.index
    )
    mole_fractions[temp] = pd.Series({m: c / total for m, c in spec_conc.items()})

# experimental data
fig1A_a = pd.read_csv(os.path.join(EXP_DIR, "Gilbert_fig1A_from_engauge_try_2a.csv"),
                      index_col="Temperature (C)")
fig1A_b = pd.read_csv(os.path.join(EXP_DIR, "Gilbert_fig1A_from_engauge_try_2b.csv"),
                      index_col="Temperature (C)")
fig1A_c = pd.read_csv(os.path.join(EXP_DIR, "Gilbert_fig1A_from_engauge_try_2c.csv"),
                      index_col="Temperature (C)")
fig1A_original = (fig1A_a + fig1A_b + fig1A_c) / 3

mf_plot = mole_fractions.T * 100
mf_plot = mf_plot.divide(100 - mf_plot["propane"], axis="index") * 100
col_order_model = ["methane", "ethane", "ethene", "propene", "ethyne"]
col_order_exp   = ["methane", "ethane", "ethene", "propene"]

fig1A_plot = fig1A_original[col_order_exp]
fig1A_plot = fig1A_plot.divide(fig1A_plot.sum(axis="columns"), axis="index") * 100
mf_plot = mf_plot[col_order_model]

f, ax = plt.subplots()
fig1A_plot.plot.area(ax=ax, linewidth=0, stacked=True, alpha=0.3)
mf_plot.plot(ax=ax, linestyle="-", linewidth=2, markersize=0, stacked=True)
ax.set_ylabel("fraction carbon from propane (%)")
ax.set_xlabel("T (°C)")
ax.set_xticks([750, 800, 850, 900, 950])
ax.set_xlim(750, 950)
ax.set_ylim(1, 110)
ax.legend([])
ax.annotate("methane", (800, 18), (760, 5), arrowprops={"arrowstyle": "-"})
ax.annotate("ethane",  (800, 31), (760, 22), arrowprops={"arrowstyle": "-"})
ax.annotate("ethene",  (800, 91), (760, 55), arrowprops={"arrowstyle": "-"})
ax.annotate("propene", (820, 97), (760, 92), arrowprops={"arrowstyle": "-"})
ax.annotate("ethyne",  (900, 100), (860, 103), arrowprops={"arrowstyle": "-"})
plt.savefig(os.path.join(IMAGE_DIR, "figure4_mole_fractions.pdf"), bbox_inches="tight")
plt.close()
print("  Saved figure4_mole_fractions.pdf")


# ── Figure 5 – enrichments by conversion ──────────────────────────────────────
print("Creating figure 5 ...")

styles = ["o", "^", "s", "v", "D"]
line_styles = [(0, (1, 2)), (0, (5, 10)), (0, (1, 5)), (0, (3, 5, 1, 5))]

f, ax = plt.subplots()
fig1B_data = pd.read_csv(os.path.join(EXP_DIR, "Gilbert_fig1B_engauge.csv"),
                          index_col="Temperature (C)")
del fig1B_data["propene"]
fig1B_data.plot(ax=ax, linestyle="", linewidth=0.5,
                style=styles, markersize=6, markeredgewidth=1)

conversions_gilbert = 1 - fig1A_original["propane"] / 100

for mi in range(2):
    m_name, m_path = MAIN_PATHS[mi]
    mol_cl = MODEL_CLUSTERS[m_name]
    iso_info = pd.read_csv(os.path.join(m_path, "isotopomer_cluster_info.csv"),
                            index_col="name")
    enrich_by_conv = pd.DataFrame(dtype=float)
    for temp in [750, 800, 850, 900, 950]:
        conv = conversions_gilbert[temp]
        solution = ct.Solution(os.path.join(m_path, "chem.yaml"))
        output = run_simulation_till_conversion(
            solution, species="CCC", conversion=conv,
            conditions=(temp + 273, 2e5, INIT_MOLE_FRACS),
            condition_type="constant-temperature-and-pressure",
            output_species=True, output_reactions=False)
        sp = output["species"].iloc[-1]
        iso_info["conc"] = sp
        for molecule, cnum in mol_cl.items():
            labels = iso_info[iso_info.cluster_number == cnum].index
            if molecule != "H-atom":
                enrich_by_conv.loc[temp, molecule] = get_delta(
                    sp[labels],
                    iso_info.loc[labels, "enriched_atoms"],
                    iso_info.loc[labels, "unenriched_atoms"],
                )
    enrich_by_conv.plot(y=list(fig1B_data.columns), ax=ax,
                        linestyle=line_styles[mi], linewidth=1,
                        style=styles, markersize=2, markeredgewidth=1,
                        markerfacecolor="None")

ax.set_ylabel("$\\delta^{13}C$\n$(\\perthousand)$", rotation="horizontal",
              va="center", ha="right")
ax.set_xlabel("T (°C)")
items_, entries_ = ax.get_legend_handles_labels()
items_ = items_[:len(fig1B_data.columns)]
legend_items = [
    Line2D([], [], linestyle="none", marker=it.get_marker(),
           markersize=it.get_markersize(), markeredgewidth=it.get_markeredgewidth(),
           markerfacecolor=it.get_markerfacecolor(),
           markeredgecolor=it.get_markeredgecolor())
    for it in items_
]
legend_items.append(Line2D([], [], linestyle="none"))
legend_items.append(Line2D([], [], linestyle="", color="black", linewidth=0.5,
                            marker="d", markerfacecolor="black",
                            markeredgecolor="black", markersize=6, markeredgewidth=1))
for ls in line_styles[:2]:
    legend_items.append(Line2D([], [], linestyle=ls, color="black", linewidth=1,
                                marker="d", markerfacecolor="none",
                                markeredgecolor="black", markersize=2, markeredgewidth=1))
entries_ = list(entries_[:len(fig1B_data.columns)]) + ["", "experiment"] + \
           [f"{n} model" for n, _ in MAIN_PATHS[:2]]
ax.legend(legend_items, entries_,
          bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
ax.set_xticks([750, 800, 850, 900, 950])
ax.set_xlim(740, 960)
ax.set_yticks([-40, -30, -20, -10, 0])
plt.savefig(os.path.join(IMAGE_DIR, "figure5_enrichments_by_conversion.pdf"),
            bbox_inches="tight")
plt.close()
print("  Saved figure5_enrichments_by_conversion.pdf")


# ── Figure 6 & Table 2 – slopes of enrichment ─────────────────────────────────
print("Creating figure 6 and table 2 ...")

model_to_slopes = {}
model_to_temp_enrichments = {}

for mi in range(4):
    m_name, m_path = MAIN_PATHS[mi]
    mol_cl = MODEL_CLUSTERS[m_name]
    iso_info = pd.read_csv(os.path.join(m_path, "isotopomer_cluster_info.csv"),
                            index_col="name")
    slopes_found = pd.DataFrame(index=["dC2H4 = f(dCH4)", "dC2H6 = f(dCH4)",
                                        "dC2H6 = f(dC2H4)", "dBulk = f(dCH4)"])
    temp_enrichments = {}

    for temp in (800, 850, 900, 950):
        delta_enrich = pd.DataFrame()
        for psia_val in np.linspace(-10, 20, 5):
            c_delta = -28
            e_delta = -28 + psia_val
            e_frac = delta_from_fraction(e_delta)
            c_frac = delta_from_fraction(c_delta)
            init_mf = {
                "CCC":   f_prop * (1-c_frac) * (1-e_frac)**2,
                "CCC-2": f_prop * c_frac * e_frac**2,
                "CCC-3": f_prop * e_frac**2 * (1-c_frac),
                "CCC-4": f_prop * 2*e_frac * (1-e_frac) * c_frac,
                "CCC-5": f_prop * 2*e_frac * (1-c_frac) * (1-e_frac),
                "CCC-6": f_prop * c_frac * (1-e_frac)**2,
                "[He]":  1 - f_prop,
            }
            conditions = (temp + 273, 2e5, init_mf)
            t_final = 95.0 / temp
            solution = ct.Solution(os.path.join(m_path, "chem.yaml"))
            output = run_simulation(solution, [0, t_final], conditions,
                                    condition_type="constant-temperature-and-pressure",
                                    output_species=True, output_reactions=False)
            sp_final = output["species"].iloc[-1]
            sp_init  = output["species"].iloc[0]
            iso_info["conc"] = sp_final

            row = {}
            for molecule, cnum in mol_cl.items():
                labels = iso_info[iso_info.cluster_number == cnum].index
                if molecule != "H-atom":
                    row[molecule] = get_delta(sp_final[labels],
                                              iso_info.loc[labels, "enriched_atoms"],
                                              iso_info.loc[labels, "unenriched_atoms"])
            # bulk taken from initial conditions (propane)
            prop_labels = iso_info[iso_info.cluster_number == mol_cl["propane"]].index
            iso_info["conc"] = sp_init
            row["bulk"] = get_delta(sp_init[prop_labels],
                                    iso_info.loc[prop_labels, "enriched_atoms"],
                                    iso_info.loc[prop_labels, "unenriched_atoms"])
            delta_enrich[psia_val] = pd.Series(row)

        delta_enrich = delta_enrich.T
        temp_enrichments[temp] = delta_enrich

        axes_x = ["methane", "methane", "ethene", "methane"]
        axes_y = ["ethene",  "ethane",  "ethane", "bulk"]
        slopes = []
        for ax_x, ax_y in zip(axes_x, axes_y):
            fit = sm.ols(formula=f"{ax_y} ~ {ax_x}", data=delta_enrich).fit()
            slopes.append(fit.params.iat[1])
        slopes_found[temp] = slopes

    model_to_slopes[m_name] = slopes_found
    model_to_temp_enrichments[m_name] = temp_enrichments

# plot figure 6
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.25)
delta_enrichments = model_to_temp_enrichments["full"][850]
fig6_axes_x = ["ethene", "methane", "methane", "methane"]
fig6_axes_y = ["ethane", "ethane",  "ethene",  "bulk"]
axes_x_ticks = [[-34, -28, -22, -16, -10]] * 4
axes_y_ticks = [np.linspace(-48, -48+32, 5), np.linspace(-48, -48+32, 5),
                np.linspace(-40, -40+28, 5), np.linspace(-40, -40+28, 5)]
axes_y_lim = [(t.min(), t.max()) for t in axes_y_ticks]
axes_x_lim = [(min(t), max(t)) for t in axes_x_ticks]
index2section = {2: "A", 1: "B", 0: "C", 3: "D"}

f, ax_grid = plt.subplots(ncols=2, nrows=2, sharex=False, sharey=False)
f.subplots_adjust(hspace=0.3, wspace=0.5)
for idx, axis in enumerate(ax_grid.flatten()):
    gilbert_data = pd.read_csv(
        os.path.join(EXP_DIR, f"Gilbert_fig2{index2section[idx]}_engauge.csv"))
    gilbert_data.plot(x=gilbert_data.columns[0], y=gilbert_data.columns[1],
                      ax=axis, style="o")
    delta_enrichments.plot(x=fig6_axes_x[idx], y=fig6_axes_y[idx],
                           ax=axis, style="-")
    axis.set_ylabel(
        f"$\\delta^{{13}}C_{{ {fig6_axes_y[idx]} }}$\n$(\\perthousand)$",
        rotation="horizontal", labelpad=25)
    axis.set_xlabel(
        f"$\\delta^{{13}}C_{{ {fig6_axes_x[idx]} }}(\\perthousand)$")
    axis.set_xlim(axes_x_lim[idx])
    axis.set_ylim(axes_y_lim[idx])
    axis.set_xticks(axes_x_ticks[idx])
    if idx in [0, 1]:
        axis.set_xticklabels([""] * 5)
    axis.set_yticks(axes_y_ticks[idx])
    if idx in [1, 3]:
        axis.set_yticklabels([""] * 5)
    axis.legend([])

plt.savefig(os.path.join(IMAGE_DIR, "figure6_enrichment_slopes.pdf"),
            bbox_inches="tight")
plt.close()
print("  Saved figure6_enrichment_slopes.pdf")

# Table 2
gilbert_vals  = pd.read_csv(os.path.join(EXP_DIR, "Gilbert_table2_values.csv"),
                             index_col="relationship")
gilbert_vals.columns = [np.int64(c) for c in gilbert_vals.columns]
gilbert_unc   = pd.read_csv(os.path.join(EXP_DIR, "Gilbert_table2_uncertainty.csv"),
                             index_col="relationship")

scaled_diffs = pd.DataFrame()
for model, sf in model_to_slopes.items():
    diff = gilbert_vals - sf
    scaled_diffs[model] = (diff.values.flatten() /
                            gilbert_unc.values.flatten())
scaled_diffs["species_slopes"] = [v for v in gilbert_vals.index for _ in range(4)]
scaled_diffs["temperature"]    = [v for _ in range(4) for v in gilbert_vals.columns]

error_table = pd.DataFrame()
error_table["all"] = scaled_diffs.std(axis="index", numeric_only=True)
for t in gilbert_vals.columns:
    subset = scaled_diffs[scaled_diffs.temperature == t]
    error_table[t] = subset.std(axis="index", numeric_only=True)
error_table = error_table.T
del error_table["temperature"]
error_table = error_table[["full", "drg", "6rxn", "3rxn"]]
print("\n########### Table 2 ##############")
print(error_table.round(decimals=1).to_latex())

print(f"\nAll figures saved to: {IMAGE_DIR}")
print("Done.")
