from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from semopy import Model, calc_stats


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
STUDY1_FILE = DATA_DIR / "study1_data.xlsx"
STUDY2_FILE = DATA_DIR / "study2_data.xlsx"
BOOTSTRAP_DRAWS = 5000
RANDOM_SEED = 20260905

S1_ORDER = ["Unrelated", "Same-product", "Similar-product", "Superior-alternative"]
S2_ORDER = ["NoAck+Alternative", "Ack+Alternative", "NoAck+Support", "Ack+Support"]

S1_FACTORS = {
    "PAL": ["PAL1", "PAL2", "PAL3", "PAL4"],
    "Trust": ["TRU1", "TRU2", "TRU3", "TRU4"],
    "Closure": ["CLO1", "CLO2", "CLO3", "CLO4", "CLO5"],
    "Counterfactual": ["CFT1", "CFT2", "CFT3", "CFT4"],
    "Regret": ["REG1", "REG2", "REG3", "REG4"],
    "Return": ["RET1", "RET2", "RET3", "RET4", "RET5"],
    "Avoidance": ["AVO1", "AVO2", "AVO3", "AVO4", "AVO5"],
    "Irritation": ["IRR1", "IRR2", "IRR3", "IRR4", "IRR5"],
}

S2_FACTORS = {
    "PAL": ["PAL1", "PAL2", "PAL3", "PAL4"],
    "ManipIntent": ["MAN1", "MAN2", "MAN3", "MAN4", "MAN5"],
    "Trust": ["TRU1", "TRU2", "TRU3", "TRU4"],
    "Closure": ["CLO1", "CLO2", "CLO3", "CLO4", "CLO5"],
    "Counterfactual": ["CFT1", "CFT2", "CFT3", "CFT4"],
    "Regret": ["REG1", "REG2", "REG3", "REG4"],
    "Irritation": ["IRR1", "IRR2", "IRR3", "IRR4", "IRR5"],
}


def save(table: pd.DataFrame, name: str) -> None:
    table.to_csv(RESULTS_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")


def p_label(p: float) -> str:
    return "< .001" if p < .001 else f"= {p:.3f}".replace("0.", ".")


def cronbach_alpha(frame: pd.DataFrame) -> float:
    x = frame.to_numpy(float)
    k = x.shape[1]
    return float(k / (k - 1) * (1 - x.var(axis=0, ddof=1).sum() / x.sum(axis=1).var(ddof=1)))


def prepare_composites(data: pd.DataFrame, factor_map: dict[str, list[str]], study: int) -> pd.DataFrame:
    data = data.copy()
    for factor, items in factor_map.items():
        missing = sorted(set(items) - set(data.columns))
        if missing:
            raise ValueError(f"Study {study} is missing columns: {missing}")
        scoring = data[items].copy()
        if study == 1 and factor == "Return":
            scoring["RET5"] = 8 - scoring["RET5"]
        data[factor] = scoring.mean(axis=1)
    if study == 1:
        data["same_vs_rest"] = np.where(data["condition"].eq("Same-product"), 1.0, -1 / 3)
        data["superior_vs_rest"] = np.where(data["condition"].eq("Superior-alternative"), 1.0, -1 / 3)
    else:
        data["interaction"] = data["acknowledge_purchase"] * data["postpurchase_support"]
    return data


def validate_data(data: pd.DataFrame, study: int, order: list[str]) -> pd.DataFrame:
    if data["ID"].duplicated().any():
        raise ValueError(f"Study {study} contains duplicate IDs")
    counts = data["condition"].value_counts().reindex(order)
    if counts.isna().any():
        raise ValueError(f"Study {study} does not contain every expected condition")
    item_prefixes = ("IDN", "SIM", "ADV", "REP", "PAL", "TRU", "CLO", "CFT", "REG", "RET", "AVO", "IRR", "ACC", "FIT", "MAN")
    item_cols = [c for c in data if c.startswith(item_prefixes) and any(ch.isdigit() for ch in c)]
    if not data[item_cols].apply(lambda s: s.between(1, 7).all()).all():
        raise ValueError(f"Study {study} has item responses outside 1-7")
    report = pd.DataFrame({"condition": order, "n": counts.to_numpy(int)})
    save(report, f"study{study}_data_validation")
    return report


def fit_cfa(data: pd.DataFrame, factor_map: dict[str, list[str]]) -> tuple[Model, pd.DataFrame]:
    syntax = "\n".join(f"{factor} =~ {' + '.join(items)}" for factor, items in factor_map.items())
    model = Model(syntax)
    columns = [item for items in factor_map.values() for item in items]
    model.fit(data[columns], obj="MLW")
    fit = calc_stats(model).loc["Value"]
    observed_order = model.vars["observed"]
    observed_corr = np.corrcoef(data[observed_order].to_numpy(float), rowvar=False)
    sigma = model.calc_sigma()[0]
    scale = np.sqrt(np.diag(sigma))
    implied_corr = sigma / np.outer(scale, scale)
    upper = np.triu_indices_from(observed_corr, k=1)
    srmr = float(np.sqrt(np.mean((observed_corr[upper] - implied_corr[upper]) ** 2)))
    table = pd.DataFrame([{
        "chi_square": float(fit["chi2"]), "df": int(fit["DoF"]),
        "CFI": float(fit["CFI"]), "TLI": float(fit["TLI"]),
        "RMSEA": float(fit["RMSEA"]), "SRMR": srmr,
    }])
    return model, table


def measurement_quality(data: pd.DataFrame, factor_map: dict[str, list[str]], model: Model, study: int) -> pd.DataFrame:
    estimates = model.inspect(std_est=True)
    standardized_column = "Est. Std" if "Est. Std" in estimates.columns else "Estimate"
    rows = []
    for factor, items in factor_map.items():
        scoring = data[items].copy()
        loadings = estimates.loc[
            (estimates["op"] == "~") & (estimates["rval"] == factor) & estimates["lval"].isin(items),
            standardized_column,
        ].astype(float).to_numpy()
        cr = loadings.sum() ** 2 / (loadings.sum() ** 2 + np.sum(1 - loadings ** 2))
        ave = np.mean(loadings ** 2)
        rows.append({
            "construct": factor, "items": len(items),
            "loading_min": loadings.min(), "loading_max": loadings.max(),
            "cronbach_alpha": cronbach_alpha(scoring), "CR": cr, "AVE": ave,
        })
    return pd.DataFrame(rows)


def descriptives(data: pd.DataFrame, outcomes: list[str], order: list[str]) -> pd.DataFrame:
    rows = []
    for outcome in outcomes:
        for condition in order:
            x = data.loc[data["condition"].eq(condition), outcome]
            rows.append({"variable": outcome, "condition": condition, "n": len(x), "M": x.mean(), "SD": x.std(ddof=1)})
    return pd.DataFrame(rows)


def one_way_table(data: pd.DataFrame, outcomes: list[str]) -> pd.DataFrame:
    rows = []
    for outcome in outcomes:
        model = smf.ols(f"{outcome} ~ C(condition)", data=data).fit()
        row = anova_lm(model, typ=2).iloc[0]
        ss_total = ((data[outcome] - data[outcome].mean()) ** 2).sum()
        rows.append({
            "variable": outcome, "df1": int(row["df"]), "df2": int(model.df_resid),
            "F": row["F"], "p": row["PR(>F)"], "eta_squared": row["sum_sq"] / ss_total,
        })
    return pd.DataFrame(rows)


def independent_comparison(data: pd.DataFrame, outcome: str, group1: str, group2: str) -> dict:
    x = data.loc[data["condition"].eq(group1), outcome].to_numpy(float)
    y = data.loc[data["condition"].eq(group2), outcome].to_numpy(float)
    t_value, p = stats.ttest_ind(x, y, equal_var=True)
    pooled_sd = math.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2))
    difference = x.mean() - y.mean()
    se = pooled_sd * math.sqrt(1 / len(x) + 1 / len(y))
    critical = stats.t.ppf(.975, len(x) + len(y) - 2)
    return {
        "variable": outcome, "group_1": group1, "group_2": group2,
        "M1": x.mean(), "SD1": x.std(ddof=1), "M2": y.mean(), "SD2": y.std(ddof=1),
        "mean_difference": difference, "SE_difference": se, "t": t_value,
        "df": len(x) + len(y) - 2, "p": p, "CI_low": difference - critical * se,
        "CI_high": difference + critical * se, "cohens_d": difference / pooled_sd,
    }


def standardized_mechanism_regressions(data: pd.DataFrame) -> pd.DataFrame:
    z = data.copy()
    variables = ["PAL", "Trust", "Irritation", "Avoidance", "Closure", "Counterfactual", "Regret", "Return"]
    for variable in variables:
        z[variable] = (z[variable] - z[variable].mean()) / z[variable].std(ddof=0)
    specifications = [
        ("Avoidance", "Avoidance ~ PAL + Trust + Irritation + C(condition)", ["PAL", "Trust", "Irritation"]),
        ("Return", "Return ~ Closure + Counterfactual + Regret + Irritation + C(condition)", ["Closure", "Counterfactual", "Regret", "Irritation"]),
    ]
    rows = []
    for outcome, formula, terms in specifications:
        model = smf.ols(formula, data=z).fit()
        ci = model.conf_int()
        for term in terms:
            rows.append({
                "outcome": outcome, "predictor": term, "beta": model.params[term],
                "SE_beta": model.bse[term], "t": model.tvalues[term], "p": model.pvalues[term],
                "CI_low": ci.loc[term, 0], "CI_high": ci.loc[term, 1], "R_squared": model.rsquared,
            })
    return pd.DataFrame(rows)


def serial_indirect(
    data: pd.DataFrame, x: str, chain: list[str], rng: np.random.Generator, full_serial_model: bool
) -> dict:
    columns = [x] + chain
    z = data[columns].apply(lambda s: (s - s.mean()) / s.std(ddof=0)).to_numpy(float)

    def path_product(sample: np.ndarray) -> float:
        n = len(sample)
        first = np.linalg.lstsq(np.c_[np.ones(n), sample[:, 0]], sample[:, 1], rcond=None)[0][1]
        product = first
        for index in range(2, sample.shape[1]):
            predictors = sample[:, [0, *range(1, index)]] if full_serial_model else sample[:, [0, index - 1]]
            coefficient = np.linalg.lstsq(
                np.c_[np.ones(n), predictors], sample[:, index], rcond=None
            )[0][-1]
            product *= coefficient
        return float(product)

    point = path_product(z)
    values = np.empty(BOOTSTRAP_DRAWS)
    for draw in range(BOOTSTRAP_DRAWS):
        values[draw] = path_product(z[rng.integers(0, len(z), len(z))])
    return {
        "specification": "full_serial_model" if full_serial_model else "adjacent_path_model_used_in_current_manuscript",
        "path": f"{x} -> " + " -> ".join(chain), "standardized_indirect_effect": point,
        "boot_SE": values.std(ddof=1), "CI_low": np.quantile(values, .025),
        "CI_high": np.quantile(values, .975), "bootstrap_draws": BOOTSTRAP_DRAWS,
    }


def linear_combination(model, terms: list[str], weights: list[float]) -> tuple[float, float, float, float, float, float]:
    names = list(model.params.index)
    vector = np.zeros(len(names))
    for term, weight in zip(terms, weights):
        vector[names.index(term)] = weight
    estimate = float(vector @ model.params.to_numpy())
    se = float(np.sqrt(vector @ model.cov_params().to_numpy() @ vector))
    t_value = estimate / se
    p = float(2 * stats.t.sf(abs(t_value), model.df_resid))
    critical = stats.t.ppf(.975, model.df_resid)
    return estimate, se, t_value, p, estimate - critical * se, estimate + critical * se


def study2_factorial(data: pd.DataFrame, outcomes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    factorial_rows, simple_rows = [], []
    terms = ["acknowledge_purchase", "postpurchase_support", "acknowledge_purchase:postpurchase_support"]
    for outcome in outcomes:
        model = smf.ols(
            f"{outcome} ~ acknowledge_purchase * postpurchase_support + initial_confidence + initial_satisfaction",
            data=data,
        ).fit()
        table = anova_lm(model, typ=2)
        error_ss = table.loc["Residual", "sum_sq"]
        for term in terms:
            row = table.loc[term]
            factorial_rows.append({
                "outcome": outcome, "effect": term, "df1": int(row["df"]), "df2": int(model.df_resid),
                "F": row["F"], "p": row["PR(>F)"],
                "partial_eta_squared": row["sum_sq"] / (row["sum_sq"] + error_ss), "R_squared": model.rsquared,
            })
        for label, effect_terms, weights in [
            ("Acknowledgment within superior-alternative ads", [terms[0]], [1]),
            ("Acknowledgment within post-purchase support", [terms[0], terms[2]], [1, 1]),
        ]:
            estimate, se, t_value, p, low, high = linear_combination(model, effect_terms, weights)
            simple_rows.append({
                "outcome": outcome, "simple_effect": label, "B": estimate, "SE": se,
                "t": t_value, "df": int(model.df_resid), "p": p, "CI_low": low, "CI_high": high,
            })
    return pd.DataFrame(factorial_rows), pd.DataFrame(simple_rows)


def study2_behavior(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rates = data.groupby("condition", sort=False).agg(
        n=("ID", "size"), reversals=("choice_reversal", "sum"),
        reversal_rate=("choice_reversal", "mean"),
        exchange_count=("behavior_choice", lambda s: (s == "Exchange").sum()),
        cancel_count=("behavior_choice", lambda s: (s == "Cancel").sum()),
        view_policy_rate=("view_policy", "mean"), hide_ad_rate=("hide_ad", "mean"),
    ).reindex(S2_ORDER).reset_index()
    model = smf.logit(
        "choice_reversal ~ acknowledge_purchase * postpurchase_support + initial_confidence + initial_satisfaction",
        data=data,
    ).fit(disp=0)
    ci = model.conf_int()
    selected = ["acknowledge_purchase", "postpurchase_support", "acknowledge_purchase:postpurchase_support"]
    logistic = pd.DataFrame([{
        "predictor": term, "B": model.params[term], "SE": model.bse[term], "z": model.tvalues[term],
        "p": model.pvalues[term], "OR": np.exp(model.params[term]),
        "OR_CI_low": np.exp(ci.loc[term, 0]), "OR_CI_high": np.exp(ci.loc[term, 1]),
    } for term in selected])

    def proportion_test(group1: str, group2: str) -> dict:
        x = data.loc[data["condition"].eq(group1), "choice_reversal"]
        y = data.loc[data["condition"].eq(group2), "choice_reversal"]
        pooled = (x.sum() + y.sum()) / (len(x) + len(y))
        se_null = math.sqrt(pooled * (1 - pooled) * (1 / len(x) + 1 / len(y)))
        difference = x.mean() - y.mean()
        z_value = difference / se_null
        se = math.sqrt(x.mean() * (1 - x.mean()) / len(x) + y.mean() * (1 - y.mean()) / len(y))
        return {
            "comparison": f"{group1} vs {group2}", "rate_1": x.mean(), "rate_2": y.mean(),
            "difference": difference, "z": z_value, "p": 2 * stats.norm.sf(abs(z_value)),
            "CI_low": difference - 1.96 * se, "CI_high": difference + 1.96 * se,
        }

    comparisons = pd.DataFrame([
        proportion_test("Ack+Alternative", "NoAck+Alternative"),
        proportion_test("Ack+Support", "NoAck+Support"),
    ])
    return rates, logistic, comparisons


def study1_analysis(data: pd.DataFrame) -> None:
    outcomes = ["PAL", "Trust", "Irritation", "Avoidance", "Closure", "Counterfactual", "Regret", "Return"]
    cfa_data = data.copy()
    cfa_data["RET5"] = 8 - cfa_data["RET5"]
    model, fit = fit_cfa(cfa_data, S1_FACTORS)
    save(fit, "study1_cfa_fit")
    save(measurement_quality(cfa_data, S1_FACTORS, model, 1), "study1_measurement_quality")
    save(descriptives(data, outcomes, S1_ORDER), "study1_descriptives")
    save(one_way_table(data, outcomes), "study1_omnibus_anova")

    manipulation_variables = {"same_product_recognition": ["IDN1", "IDN2", "IDN3"],
                              "alternative_superiority": ["ADV1", "ADV2", "ADV3"],
                              "advertising_repetition": ["REP1", "REP2", "REP3"]}
    checks = data.copy()
    for name, items in manipulation_variables.items():
        checks[name] = checks[items].mean(axis=1)
    save(descriptives(checks, list(manipulation_variables), S1_ORDER), "study1_manipulation_descriptives")
    save(one_way_table(checks, list(manipulation_variables)), "study1_manipulation_anova")

    comparisons = []
    comparisons.append(independent_comparison(data, "PAL", "Same-product", "Unrelated"))
    for outcome in ["Closure", "Counterfactual", "Regret", "Return"]:
        comparisons.append(independent_comparison(data, outcome, "Superior-alternative", "Similar-product"))
    save(pd.DataFrame(comparisons), "study1_planned_comparisons")
    save(standardized_mechanism_regressions(data), "study1_mechanism_regressions")
    rng = np.random.default_rng(RANDOM_SEED)
    indirect = pd.DataFrame([
        serial_indirect(data, "same_vs_rest", ["PAL", "Trust", "Avoidance"], rng, False),
        serial_indirect(data, "same_vs_rest", ["PAL", "Trust", "Avoidance"], rng, True),
        serial_indirect(data, "superior_vs_rest", ["Closure", "Counterfactual", "Regret", "Return"], rng, False),
        serial_indirect(data, "superior_vs_rest", ["Closure", "Counterfactual", "Regret", "Return"], rng, True),
    ])
    save(indirect, "study1_serial_indirect_effects")


def study2_analysis(data: pd.DataFrame) -> None:
    outcomes = ["PAL", "ManipIntent", "Trust", "Closure", "Counterfactual", "Regret", "Irritation"]
    model, fit = fit_cfa(data, S2_FACTORS)
    save(fit, "study2_cfa_fit")
    save(measurement_quality(data, S2_FACTORS, model, 2), "study2_measurement_quality")
    save(descriptives(data, outcomes, S2_ORDER), "study2_descriptives")
    factorial, simple = study2_factorial(data, outcomes)
    save(factorial, "study2_factorial_ancova")
    save(simple, "study2_simple_effects")
    rates, logistic, proportions = study2_behavior(data)
    save(rates, "study2_behavior_rates")
    save(logistic, "study2_logistic_regression")
    save(proportions, "study2_proportion_tests")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    study1 = prepare_composites(pd.read_excel(STUDY1_FILE, sheet_name="data"), S1_FACTORS, 1)
    study2 = prepare_composites(pd.read_excel(STUDY2_FILE, sheet_name="data"), S2_FACTORS, 2)
    validate_data(study1, 1, S1_ORDER)
    validate_data(study2, 2, S2_ORDER)
    study1_analysis(study1)
    study2_analysis(study2)
    print(f"Analysis complete. Results saved to: {RESULTS_DIR}")
    print(f"Study 1 N = {len(study1)}; Study 2 N = {len(study2)}")


if __name__ == "__main__":
    main()
