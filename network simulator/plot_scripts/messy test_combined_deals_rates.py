import IPython
import sys
import os
import typing as tp

__file__ = "plot_scripts/test_combined_centralities_rates.py"
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

# import setup  # noqa: E402
import results  # noqa: E402
import plotter  # noqa: E402
import compare  # noqa: E402

ultratb = IPython.core.ultratb
sys.excepthook = ultratb.FormattedTB(
    mode="Plain", color_scheme="Linux", call_pdb=False
)


def prep_results_for_plot(
    date: str,
    time: str,
    split: bool,
    fixed_idcs: tp.Optional[tp.List[int]] = None,
    first_file_idx: int = 0,
    # ) -> tp.Optional[tp.Tuple[tp.Any, tp.Any]]:
) -> tp.Dict[str, tp.Any]:
    results_list_flat = results.Results.load_results(
        date, time, split, fixed_idcs, first_file_idx
    )
    compare_results_list = compare.get_compare_results_list(
        results_list_flat,
        [
            ("attacker_data", "profits_stats", "mean"),
            ("attacker_data", "lag_profits_stats", "mean"),
            ("users_data", "oracle_accepts_cheat_stats", "mean"),
            ("users_data", "oracle_accepts_deal_stats", "mean"),
            ("attacker_data", "expenditure_stats", "mean"),
            ("attacker_data", "rejects_deals_stats", "mean"),
            ("attacker_data", "rejects_cheats_stats", "mean"),
            ("attacker_data", "last_accept_idcs_stats", "mean"),
        ],
    )
    # sorted_compare_results_list = sorted(
    #     compare_results_list, key=lambda x: x[1][0]  # type:ignore
    # )
    atk_types: tp.List[str]
    atk_types, multi_val_lst_lst = zip(*compare_results_list)
    # print(multi_val_lst_lst)

    income_means: tp.List[float] = [x[0] for x in multi_val_lst_lst]
    income_errorbars = compare.get_errors(
        results_list_flat, "attacker_data", "profits_stats"
    )
    income_tuple = (
        income_means,
        income_errorbars,
        "Rate: ",  # + atk_types[0].replace("_", " ").split()[0],
    )

    lag_income_means: tp.List[float] = [x[1] for x in multi_val_lst_lst]
    lag_income_errorbars = compare.get_errors(
        results_list_flat, "attacker_data", "lag_profits_stats"
    )
    lag_income_tuple = (
        lag_income_means,
        lag_income_errorbars,
        "Lag Cheats ",  # + atk_types[0].replace("_", " ").split()[0],
    )

    oracle_income_means: tp.List[float] = [x[2] for x in multi_val_lst_lst]
    oracle_income_errorbars = compare.get_errors(
        results_list_flat, "users_data", "oracle_accepts_cheat_stats"
    )
    oracle_income_tuple = (
        oracle_income_means,
        oracle_income_errorbars,
        "Oracle Cheats ",  # + atk_types[0].replace("_", " ").split()[0],
    )
    oracle_deals_means: tp.List[float] = [x[3] for x in multi_val_lst_lst]
    oracle_deals_errorbars = compare.get_errors(
        results_list_flat, "users_data", "oracle_accepts_deal_stats"
    )
    oracle_deals_tuple = (
        oracle_deals_means,
        oracle_deals_errorbars,
        "Oracle Deals ",  # + atk_types[0].replace("_", " ").split()[0],
    )

    expenditure_means: tp.List[float] = [x[4] for x in multi_val_lst_lst]
    expenditure_errorbars = compare.get_errors(
        results_list_flat, "attacker_data", "expenditure_stats"
    )
    expenditure_tuple = (
        expenditure_means,
        expenditure_errorbars,
        "Rate: ",  # + atk_types[0].replace("_", " ").split()[0],
    )

    rejects_deals_means: tp.List[float] = [x[5] for x in multi_val_lst_lst]
    rejects_deals_errorbars = compare.get_errors(
        results_list_flat, "attacker_data", "rejects_deals_stats"
    )
    rejects_deals_tuple = (
        rejects_deals_means,
        rejects_deals_errorbars,
        "Rate: ",  # + atk_types[0].replace("_", " ").split()[0],
    )

    rejects_cheats_means: tp.List[float] = [x[6] for x in multi_val_lst_lst]
    rejects_cheats_errorbars = compare.get_errors(
        results_list_flat, "attacker_data", "rejects_cheats_stats"
    )
    rejects_cheats_tuple = (
        rejects_cheats_means,
        rejects_cheats_errorbars,
        "Rate: ",  # + atk_types[0].replace("_", " ").split()[0],
    )

    last_accept_idcs_means: tp.List[float] = [x[7] for x in multi_val_lst_lst]
    last_accept_idcs_errorbars = compare.get_errors(
        results_list_flat, "attacker_data", "last_accept_idcs_stats"
    )
    last_accept_idcs_tuple = (
        last_accept_idcs_means,
        last_accept_idcs_errorbars,
        "Rate: ",  # + atk_types[0].replace("_", " ").split()[0],
    )

    rates = [r.data["attacker_data"]["rate"][0] for r in results_list_flat]
    rates_vars = (rates, "Rate")
    periods = [(1 / x) for x in rates]
    periods_vars = (periods, "Period")
    deals_cap = [
        r.data["attacker_data"]["action_caps"][0]["deal"]
        for r in results_list_flat
    ]
    deals_cap_vars = (deals_cap, "Deals Cap")
    skips_cap = [
        r.data["attacker_data"]["action_caps"][0]["skip"]
        for r in results_list_flat
    ]
    skips_cap_vars = (skips_cap, "Waiting Cap")
    atk_types_vars = (atk_types, "Attacker")
    graph_name = results_list_flat[0].data["repgraph_data"]["name"][0]

    results_prep_dict = {
        "income_tuple": income_tuple,
        "lag_income_tuple": lag_income_tuple,
        "oracle_income_tuple": oracle_income_tuple,
        "oracle_deals_tuple": oracle_deals_tuple,
        "expenditure_tuple": expenditure_tuple,
        "rejects_deals_tuple": rejects_deals_tuple,
        "rejects_cheats_tuple": rejects_cheats_tuple,
        "last_accept_idcs_tuple": last_accept_idcs_tuple,
        "rates_vars": rates_vars,
        "periods_vars": periods_vars,
        "deals_cap_vars": deals_cap_vars,
        "skips_cap_vars": skips_cap_vars,
        "atk_types_vars": atk_types_vars,
        "graph_name": graph_name,
    }
    return results_prep_dict


def prep_plot_dict(
    results_prep_dict_list: tp.List[tp.Dict[str, tp.Any]],
    var_type: str,
    val_type: str,
    ylabel: str,
    plot_type: str,
    plot_title: str,
    filename_base: str,
    xscale: str,
    yscale: str,
    transpose: bool,
    labels_suffixes: tp.List[str],
) -> tp.Dict[str, tp.Any]:
    vars_lst = [
        r_p_dict[var_type + "_vars"] for r_p_dict in results_prep_dict_list
    ]
    vals_tuple_list = [
        r_p_dict[val_type + "_tuple"] for r_p_dict in results_prep_dict_list
    ]
    vals_lst_lst, vals_err_lst_lst, vals_label_lst = zip(*vals_tuple_list)
    # old_vals = vals_lst_lst
    # old_errs = vals_err_lst_lst
    # TODO
    # ! Consider converting to np.arrays
    if transpose:
        variables, xlabel = list(zip(*vars_lst))
        variables = [v[0] for v in variables]
        xlabel = xlabel[0]
        vals_lst_lst = list(zip(*vals_lst_lst))
        vals_err_lst_lst = list(zip(*vals_err_lst_lst))
        vals_err_lst_lst = [list(zip(*e)) for e in vals_err_lst_lst]
    else:
        variables, xlabel = vars_lst[0]
        for vars_tmp, xlabel_tmp in vars_lst:
            if not ((variables == vars_tmp) and (xlabel == xlabel_tmp)):
                raise ValueError("All vars should be the same")
        vals_err_lst_lst = [list(zip(*e)) for e in vals_err_lst_lst]
    vals_label_lst = [
        vals_label_lst[0] + " " + labels_suffixes[idx]
        for idx in range(len(vals_lst_lst))
    ]
    # TODO
    # ! Debug
    # print("variables")
    # print(variables)
    # print("vals_lst_lst")
    # print(vals_lst_lst)
    # print("vals_err_lst_lst")
    # print(vals_err_lst_lst)
    # print("vals_label_lst")
    # print(vals_label_lst)
    graph_name_lst = [
        r_p_dict["graph_name"] for r_p_dict in results_prep_dict_list
    ]
    graph_name = graph_name_lst.pop()
    for gn_tmp in graph_name_lst:
        if not (graph_name == gn_tmp):
            raise ValueError("All graph names should be the same")

    results_plot_dict = {
        "variables": variables,
        "vals_lst_lst": vals_lst_lst,
        # "old_vals": old_vals,
        "vals_err_lst_lst": vals_err_lst_lst,
        # "old_errs": old_errs,
        "vals_label_lst": vals_label_lst,
        "xlabel": xlabel,
        "ylabel": ylabel,
        "plot_title": plot_title,
        "filename_base": filename_base,
        "plot_type": plot_type,
        "graph_name": graph_name,
        "xscale": xscale,
        "yscale": yscale,
    }
    return results_plot_dict


# var_type = "rates"
# var_type = "periods"
# var_type = "deals_cap"
var_type = "skips_cap"
# var_type = "atk_types"
val_type = "income"
# val_type = "lag_income"
# val_type = "oracle_income"
# val_type = "oracle_deals"
# val_type = "expenditure"
# val_type = "rejects_deals"
# val_type = "rejects_cheats"
# val_type = "last_accept_idcs"
if val_type in ["expenditure", "oracle_deals"]:
    ylabel = "Deals"
elif val_type in ["rejects_deals"]:
    ylabel = "Rejected Deals"
elif val_type in ["income", "lag_income", "oracle_income"]:
    ylabel = "Cheats"
elif val_type in ["rejects_cheats"]:
    ylabel = "Rejected Cheats"
elif val_type in ["last_accept_idcs"]:
    ylabel = "Last Successful Attacker Move"
else:
    raise ValueError("No valid val_type specified")
plot_title = ""
filename_base = var_type + " " + val_type
# plot_type = "std"
plot_type = "bar"
# xscale = "linear"
xscale = "log"
# yscale = "linear"
yscale = "log"
results_idcs = None
# results_idcs = [4, 7]
# results_idcs = list(range(2, 5))
# results_idcs = list(range(5, 8))
# results_idcs = [1, 3, 5, 7]
# results_idcs = [0, 2, 5, 7]
# results_idcs = [2, 5, 6, 7]
# results_idcs = [4, 5, 6, 7]
if var_type in ["deals_cap", "skips_cap"]:
    transpose = True
else:
    transpose = False
legend_kwargs = {
    # "loc": "best",
    "loc": "upper left",
    # "loc": "upper right",
    # "ncol": 1,
    # "ncol": 2,
    "ncol": 3,
    "fontsize": "x-small",
}
ybound: tp.Optional[tp.Dict[str, float]]
ybound = None
# ybound = {"lower": 0, "upper": 1 * 10 ** 6}
ybound = {"lower": 0, "upper": 4 * 10 ** 4}
# axhlines = None
axhlines = [(2000, "r", "Cheat Only Max")]
axhlines = [(27600, "r", "Cheat with 64 Deals Max")]


load_basic = False
load_ord = False
load_skip_short = False
load_skip_short_ord = False
load_skip_long = False
load_skip_long_ord = False
load_optimal = True

labels_suffixes_basic = [
    "1e0",
    "1e1",
    "1e2",
    "1e3",
    "1e4",
    "1e5",
    "1e6",
    "1e7",
]
labels_suffixes_ord = [
    "1e0, ord",
    "1e1, ord",
    "1e2, ord",
    "1e3, ord",
    "1e4, ord",
    "1e5, ord",
    "1e6, ord",
    "1e7, ord",
]
labels_suffixes_skip_short = [
    "1e0, short",
    "1e1, short",
    "1e2, short",
    "1e3, short",
    "1e4, short",
    "1e5, short",
    "1e6, short",
    "1e7, short",
]
labels_suffixes_skip_short_ord = [
    "1e0, short, ord",
    "1e1, short, ord",
    "1e2, short, ord",
    "1e3, short, ord",
    "1e4, short, ord",
    "1e5, short, ord",
    "1e6, short, ord",
    "1e7, short, ord",
]
labels_suffixes_skip_long = [
    "1e0, long",
    "1e1, long",
    "1e2, long",
    "1e3, long",
    "1e4, long",
    "1e5, long",
    "1e6, long",
    "1e7, long",
]
labels_suffixes_skip_long_ord = [
    "1e0, long, ord",
    "1e1, long, ord",
    "1e2, long, ord",
    "1e3, long, ord",
    "1e4, long, ord",
    "1e5, long, ord",
    "1e6, long, ord",
    "1e7, long, ord",
]

if results_idcs is not None:
    labels_suffixes_basic = [
        s for idx, s in enumerate(labels_suffixes_basic) if idx in results_idcs
    ]
    labels_suffixes_ord = [
        s for idx, s in enumerate(labels_suffixes_ord) if idx in results_idcs
    ]
    labels_suffixes_skip_short = [
        s
        for idx, s in enumerate(labels_suffixes_skip_short)
        if idx in results_idcs
    ]
    labels_suffixes_skip_short_ord = [
        s
        for idx, s in enumerate(labels_suffixes_skip_short_ord)
        if idx in results_idcs
    ]
    labels_suffixes_skip_long = [
        s
        for idx, s in enumerate(labels_suffixes_skip_long)
        if idx in results_idcs
    ]
    labels_suffixes_skip_long_ord = [
        s
        for idx, s in enumerate(labels_suffixes_skip_long_ord)
        if idx in results_idcs
    ]


if load_basic:
    prepped_results_0 = prep_results_for_plot(
        "2021-10-01", "22-23-06", True, results_idcs
    )
    prepped_results_4 = prep_results_for_plot(
        "2021-10-01", "01-12-52", True, results_idcs
    )
    # prepped_results_10 = prep_results_for_plot(
    #     "2021-09-28", "21-09-52", True, results_idcs
    # )
    prepped_results_16 = prep_results_for_plot(
        "2021-10-01", "00-20-29", True, results_idcs
    )
    prepped_results_64 = prep_results_for_plot(
        "2021-09-30", "22-40-15", True, results_idcs
    )
    # prepped_results_100 = prep_results_for_plot(
    #     "2021-09-28", "22-42-09", True, results_idcs
    # )
    # "2021-09-30", "15-42-43", True, results_idcs  # 300 runs
    prepped_results_256 = prep_results_for_plot(
        "2021-10-05", "05-29-56", True, results_idcs  # 500 runs
    )

    # prepped_results_256_attempts_stats = prep_results_for_plot(
    #     "22021-10-04", "15-22-25", True, results_idcs
    # )
    # prepped_results_256_attempts_stats_ord_maybe = prep_results_for_plot(
    #     "2021-10-03", "06-57-19", True, results_idcs
    # )
    # cap_evidence_for_rate_1e6 = prep_results_for_plot(
    #     "2021-09-30", "15-42-43", True, results_idcs
    # )

# temp = results.Results.load_results(
#     "2021-09-30", "15-42-43", True, results_idcs
# )

if load_ord:
    prepped_results_4_ord = prep_results_for_plot(
        "2021-10-01", "01-13-04", True, results_idcs
    )
    # prepped_results_10_ord = prep_results_for_plot(
    #     "2021-09-29", "00-35-09", True, results_idcs
    # )
    prepped_results_16_ord = prep_results_for_plot(
        "2021-10-01", "00-20-54", True, results_idcs
    )
    prepped_results_64_ord = prep_results_for_plot(
        "2021-09-30", "22-38-05", True, results_idcs
    )
    # prepped_results_100_ord = prep_results_for_plot(
    #     "2021-09-29", "00-49-24", True, results_idcs
    # )
    prepped_results_256_ord = prep_results_for_plot(
        "2021-10-01", "03-44-10", True, results_idcs
    )
    # cap_evidence_256_ord = prep_results_for_plot(
    #     "2021-09-30", "20-00-23", True, results_idcs
    # )

if load_skip_short:
    prepped_results_skip_1e3_4 = prep_results_for_plot(
        "2021-10-01", "17-47-32", True, results_idcs, 1
    )
    prepped_results_skip_1e3_64 = prep_results_for_plot(
        "2021-10-01", "20-17-05", True, results_idcs, 2
    )

if load_skip_short_ord:
    prepped_results_skip_1e3_4_ord = prep_results_for_plot(
        "2021-10-01", "17-47-43", True, results_idcs, 1
    )
    prepped_results_skip_1e3_64_ord = prep_results_for_plot(
        "2021-10-01", "20-17-12", True, results_idcs, 2
    )

if load_skip_long:
    prepped_results_skip_1e4_4 = prep_results_for_plot(
        "2021-10-01", "18-20-57", True, results_idcs, 1
    )
    prepped_results_skip_1e4_64 = prep_results_for_plot(
        "2021-10-01", "19-01-09", True, results_idcs, 2
    )

if load_skip_long_ord:
    prepped_results_skip_1e4_4_ord = prep_results_for_plot(
        "2021-10-01", "18-25-21", True, results_idcs, 2
    )
    prepped_results_skip_1e4_64_ord = prep_results_for_plot(
        "2021-10-01", "18-52-29", True, results_idcs, 2
    )

if load_optimal:
    # deal_to_thresh_evidence = prep_results_for_plot(
    #     "2021-10-02", "21-26-25", True, results_idcs
    # )
    optimal_skips_1e5 = prep_results_for_plot(
        "2021-10-05", "22-58-13", True, results_idcs
    )
    optimal_skips_1e6 = prep_results_for_plot(
        "2021-10-05", "23-22-55", True, results_idcs
    )
    optimal_skips_1e7 = prep_results_for_plot(
        "2021-10-06", "04-01-35", True, results_idcs
    )

if load_basic:
    prepped_results_list_basic = [
        # prepped_results_0,
        prepped_results_4,
        prepped_results_16,
        prepped_results_64,
        prepped_results_256,
    ]
if load_ord:
    prepped_results_list_ord = [
        prepped_results_4_ord,
        # prepped_results_16_ord,
        prepped_results_64_ord,
        # prepped_results_256_ord,
    ]
if load_skip_short:
    prepped_results_list_skip_short = [
        prepped_results_skip_1e3_4,
        prepped_results_skip_1e3_64,
    ]
if load_skip_short_ord:
    prepped_results_list_skip_short_ord = [
        prepped_results_skip_1e3_4_ord,
        prepped_results_skip_1e3_64_ord,
    ]
if load_skip_long:
    prepped_results_list_skip_long = [
        prepped_results_skip_1e4_4,
        prepped_results_skip_1e4_64,
    ]
if load_skip_long_ord:
    prepped_results_list_skip_long_ord = [
        prepped_results_skip_1e4_4_ord,
        prepped_results_skip_1e4_64_ord,
    ]
if load_optimal:
    prepped_results_list_optimal = [
        optimal_skips_1e5,
        optimal_skips_1e6,
        optimal_skips_1e7,
    ]

if load_basic:
    plot_dict_basic = prep_plot_dict(
        prepped_results_list_basic,
        var_type,
        val_type,
        ylabel,
        plot_type,
        plot_title,
        filename_base,
        xscale,
        yscale,
        transpose,
        labels_suffixes_basic,
    )
if load_ord:
    plot_dict_ord = prep_plot_dict(
        prepped_results_list_ord,
        var_type,
        val_type,
        ylabel,
        plot_type,
        plot_title,
        filename_base,
        xscale,
        yscale,
        transpose,
        labels_suffixes_ord,
    )
if load_skip_short:
    plot_dict_skip_short = prep_plot_dict(
        prepped_results_list_skip_short,
        var_type,
        val_type,
        ylabel,
        plot_type,
        plot_title,
        filename_base,
        xscale,
        yscale,
        transpose,
        labels_suffixes_skip_short,
    )
if load_skip_short_ord:
    plot_dict_skip_short_ord = prep_plot_dict(
        prepped_results_list_skip_short_ord,
        var_type,
        val_type,
        ylabel,
        plot_type,
        plot_title,
        filename_base,
        xscale,
        yscale,
        transpose,
        labels_suffixes_skip_short_ord,
    )
if load_skip_long:
    plot_dict_skip_long = prep_plot_dict(
        prepped_results_list_skip_long,
        var_type,
        val_type,
        ylabel,
        plot_type,
        plot_title,
        filename_base,
        xscale,
        yscale,
        transpose,
        labels_suffixes_skip_long,
    )
if load_skip_long_ord:
    plot_dict_skip_long_ord = prep_plot_dict(
        prepped_results_list_skip_long_ord,
        var_type,
        val_type,
        ylabel,
        plot_type,
        plot_title,
        filename_base,
        xscale,
        yscale,
        transpose,
        labels_suffixes_skip_long_ord,
    )
if load_optimal:
    plot_dict_optimal = prep_plot_dict(
        prepped_results_list_optimal,
        var_type,
        val_type,
        ylabel,
        plot_type,
        plot_title,
        filename_base,
        xscale,
        yscale,
        transpose,
        labels_suffixes_basic,
    )

if load_basic:
    plot_dict_template = plot_dict_basic
elif load_ord:
    plot_dict_template = plot_dict_ord
elif load_skip_short:
    plot_dict_template = plot_dict_skip_short
elif load_skip_long:
    plot_dict_template = plot_dict_skip_long
elif load_skip_short_ord:
    plot_dict_template = plot_dict_skip_short_ord
elif load_skip_long_ord:
    plot_dict_template = plot_dict_skip_long_ord
elif load_optimal:
    plot_dict_template = plot_dict_optimal
else:
    raise ValueError("No template plot_dict")

keys_to_mod = ("vals_lst_lst", "vals_err_lst_lst", "vals_label_lst")
plot_dict = {
    k: v for k, v in plot_dict_template.items() if k not in keys_to_mod
}

for key in keys_to_mod:
    zipped_dict_vals = []
    if load_basic:
        zipped_dict_vals.append(plot_dict_basic[key])
    if load_ord:
        zipped_dict_vals.append(plot_dict_ord[key])
    if load_skip_short:
        zipped_dict_vals.append(plot_dict_skip_short[key])
    if load_skip_short_ord:
        zipped_dict_vals.append(plot_dict_skip_short_ord[key])
    if load_skip_long:
        zipped_dict_vals.append(plot_dict_skip_long[key])
    if load_skip_long_ord:
        zipped_dict_vals.append(plot_dict_skip_long_ord[key])
    if load_optimal:
        zipped_dict_vals.append(plot_dict_optimal[key])
    plot_dict[key] = [
        val for zip_tuple in zip(*zipped_dict_vals) for val in zip_tuple
    ]

# TODO
# ! Debug
# variables = plot_dict["variables"]
# vals = plot_dict["vals_lst_lst"]
# errs = plot_dict["vals_err_lst_lst"]
# labels = plot_dict["vals_label_lst"]
# variables_ord = plot_dict_ord["variables"]
# vals_ord = plot_dict_ord["vals_lst_lst"]
# errs_ord = plot_dict_ord["vals_err_lst_lst"]
# labels_ord = plot_dict_ord["vals_label_lst"]
# old_vals = plot_dict["old_vals"]
# old_errs = plot_dict["old_errs"]

plotter_object = plotter.Plotter()
plotter_object.plot_results(
    **plot_dict, legend_kwargs=legend_kwargs, ybound=ybound, axhlines=axhlines
)
