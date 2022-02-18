import IPython
import sys
import os

__file__ = "plot_scripts/test_combined_centralities_rates.py"
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import typing as tp  # noqa: E402
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
    # ) -> tp.Optional[tp.Tuple[tp.Any, tp.Any]]:
) -> tp.Dict[str, tp.Any]:
    results_list_flat = results.Results.load_results(
        date, time, split, fixed_idcs
    )
    compare_results_list = compare.get_compare_results_list(
        results_list_flat,
        [
            ("attacker_data", "profits_stats", "mean"),
            ("attacker_data", "lag_profits_stats", "mean"),
            ("users_data", "oracle_accepts_cheat_stats", "mean"),
        ],
    )

    sorted_compare_results_list = sorted(
        compare_results_list, key=lambda x: x[1][0]  # type:ignore
    )
    atk_types: tp.List[str]
    atk_types, multi_val_lst_lst = zip(*sorted_compare_results_list)

    profits_means: tp.List[float] = [x[0] for x in multi_val_lst_lst]
    profits_errorbars = compare.get_errors(
        results_list_flat, "attacker_data", "profits_stats"
    )
    profits_tuple = (
        profits_means,
        profits_errorbars,
        "" + atk_types[0].split()[0],
    )

    lag_profits_means: tp.List[float] = [x[1] for x in multi_val_lst_lst]
    lag_profits_errorbars = compare.get_errors(
        results_list_flat, "attacker_data", "lag_profits_stats"
    )
    lag_profits_tuple = (
        lag_profits_means,
        lag_profits_errorbars,
        "Mean Lag Cheats " + atk_types[0].split()[0],
    )

    oracle_profits_means: tp.List[float] = [x[2] for x in multi_val_lst_lst]
    oracle_profits_errorbars = compare.get_errors(
        results_list_flat, "users_data", "oracle_accepts_cheat_stats"
    )
    oracle_profits_tuple = (
        oracle_profits_means,
        oracle_profits_errorbars,
        "Mean Oracle Cheats " + atk_types[0].split()[0],
    )

    rates = [r.data["attacker_data"]["rate"][0] for r in results_list_flat]
    rates_vars = (rates, "Rate")
    periods = [(1 / x) for x in rates]
    periods_vars = (periods, "Period")
    atk_types_vars = (atk_types, "Attacker")
    graph_name = results_list_flat[0].data["repgraph_data"]["name"][0]

    results_prep_dict = {
        "profits_tuple": profits_tuple,
        "lag_profits_tuple": lag_profits_tuple,
        "oracle_profits_tuple": oracle_profits_tuple,
        "rates_vars": rates_vars,
        "periods_vars": periods_vars,
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
) -> tp.Dict[str, tp.Any]:
    vars_lst = [
        r_p_dict[var_type + "_vars"] for r_p_dict in results_prep_dict_list
    ]
    variables, xlabel = vars_lst.pop()
    for vars_tmp, xlabel_tmp in vars_lst:
        if not ((variables == vars_tmp) and (xlabel == xlabel_tmp)):
            raise ValueError("All vars should be the same")

    vals_tuple_list = [
        r_p_dict[val_type + "_tuple"] for r_p_dict in results_prep_dict_list
    ]
    vals_lst_lst, vals_err_lst_lst, vals_label_lst = zip(*vals_tuple_list)
    vals_err_lst_lst = [list(zip(*e)) for e in vals_err_lst_lst]

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
        "vals_err_lst_lst": vals_err_lst_lst,
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


var_type = "rates"
# var_type = "periods"
# var_type = "atk_types"
val_type = "profits"
# val_type = "lag_profits"
# val_type = "oracle_profits"
ylabel = "Cheats"
plot_title = ""
filename_base = "centralities"
# plot_type = "std"
plot_type = "bar"
# xscale = "linear"
xscale = "log"
# yscale = "linear"
yscale = "log"
results_idcs = None
# results_idcs = list(range(0, 4))
results_idcs = list(range(4, 8))

prepped_results_flat = prep_results_for_plot(
    "2021-10-04", "03-29-49", True, results_idcs
)
# "2021-09-20", "23-54-44"
prepped_results_dc = prep_results_for_plot(
    "2021-10-04", "03-44-45", True, results_idcs
)
# "2021-09-23", "11-16-43"
prepped_results_spbc = prep_results_for_plot(
    "2021-10-04", "04-04-35", True, results_idcs
)
# "2021-09-23", "17-50-38"
prepped_results_cc = prep_results_for_plot(
    "2021-10-04", "04-23-01", True, results_idcs
)
prepped_results_ec = prep_results_for_plot(
    "2021-10-04", "04-50-19", True, results_idcs
)

prepped_results_list = [
    prepped_results_flat,
    prepped_results_dc,
    prepped_results_spbc,
    prepped_results_cc,
    prepped_results_ec,
]
plot_dict = prep_plot_dict(
    prepped_results_list,
    var_type,
    val_type,
    ylabel,
    plot_type,
    plot_title,
    filename_base,
    xscale,
    yscale,
)

plotter_object = plotter.Plotter()
plotter_object.plot_results(**plot_dict)
# TODO
# ! Redo with additional rate=1
# ! Just rename the existing `results k` to `results k+1`
# ! Then, do the rate=1 run and it'll automatically name itself
# fix_ax_out = test_combined_results(results_list_flat_flat, po, False)
# test_combined_results(results_list_flat_dc, po, True, fix_ax_out)
