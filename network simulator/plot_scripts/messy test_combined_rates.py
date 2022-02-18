import IPython
import sys
import os

__file__ = "plot_scripts/test_combined_rates.py"

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from pprint import pprint
import typing as tp  # noqa: E402
import results  # noqa: E402
import plotter  # noqa: E402
import compare  # noqa: E402

ultratb = IPython.core.ultratb
sys.excepthook = ultratb.FormattedTB(
    mode="Plain", color_scheme="Linux", call_pdb=False
)


def test_combined_results(
    results_list_flat: tp.List[results.Results],
    po: plotter.Plotter,
    finish_plot: bool,
    fig_ax_in: tp.Optional[tp.Tuple[tp.Any, tp.Any]] = None,
    clairvoyant: bool = False,
) -> tp.Optional[tp.Tuple[tp.Any, tp.Any]]:
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
    profits_means: tp.List[float]
    accepts_means: tp.List[float]
    lag_profits_means: tp.List[float]
    oracle_accepts_cheat_means: tp.List[float]
    atk_types, multi_values_list = zip(*sorted_compare_results_list)
    profits_means = [x[0] for x in multi_values_list]
    lag_profits_means = [x[1] for x in multi_values_list]
    oracle_accepts_cheat_means = [x[2] for x in multi_values_list]

    profits_errorbars = list(
        zip(
            *compare.get_errors(
                results_list_flat, "attacker_data", "profits_stats"
            )
        )
    )

    lag_profits_errorbars = list(
        zip(
            *compare.get_errors(
                results_list_flat, "attacker_data", "lag_profits_stats"
            )
        )
    )

    oracle_accepts_cheat_errorbars = list(
        zip(
            *compare.get_errors(
                results_list_flat, "users_data", "oracle_accepts_cheat_stats"
            )
        )
    )
    print()
    print("*****")
    print(profits_means)
    print("*****")
    print()
    values_labels_zipped_list = [
        (
            profits_means,
            "Mean Cheats",
            profits_errorbars,
        ),
        # (
        #     lag_profits_means,
        #     "Mean Lag Cheats",
        #     lag_profits_errorbars,
        # ),
        # (
        #     oracle_accepts_cheat_means,
        #     "Mean Oracle Cheats",
        #     oracle_accepts_cheat_errorbars,
        # ),
    ]

    values_list, values_label_list, values_errors_list = zip(
        *values_labels_zipped_list
    )
    pprint(values_label_list)
    if clairvoyant:
        values_label_list = ("Clairvoyant Mean Cheats",)
    rates = [r.data["attacker_data"]["rate"][0] for r in results_list_flat]
    variables, xlabel = (rates, "Rate")
    # periods = [(1 / x) for x in rates]
    # variables, xlabel = (periods, "Period")
    ylabel = "Cheats"
    title = "Test Plot"
    plot_name = "exp rate only cheats " + values_label_list[0]
    plot_type = "std"
    # xscale = "linear"
    xscale = "log"
    yscale = "linear"
    # yscale = "log"

    axhlines_arg = [
        # (5, "b", "Lag-free Cheats"),
        (2000, "r", "Max Cheats with Deals"),
    ]
    axvlines_arg = [
        # (3900, "g", "System Rate"),
    ]
    if clairvoyant:
        axhlines_arg = []
        axvlines_arg = []
    plot_return = po.plot_results(
        variables,
        values_list,
        values_errors_list,
        values_label_list,
        xlabel,
        ylabel,
        plot_type,
        "",
        plot_name,
        xscale,
        yscale,
        finish_plot=finish_plot,
        fig_ax_in=fig_ax_in,
        axhlines=axhlines_arg,
        axvlines=axvlines_arg,
    )
    return plot_return


# 300 runs
# results_list_flat_A = results.Results.load_results(
#     "2021-09-19", "16-08-08", True
# )
# 1000 runs
results_list_flat_rates_std = results.Results.load_results(
    "2021-10-01", "22-23-06", True
)

# results_list_flat_B = results.Results.load_results(
#     "2021-09-16", "03-49-32", True
# ) + results.Results.load_results("2021-09-16", "01-39-08", True)
# results_list_flat_B = results.Results.load_results(
#     "2021-09-19", "20-39-54", True
# )
# results_list_flat_B = results.Results.load_results(
#     "2021-09-19", "21-03-45", True
# )
# results_list_flat_C = results.Results.load_results(
#     "2021-09-22", "13-11-50", True
# )
# results_list_flat_D = results.Results.load_results(
#     "2021-09-27", "23-06-36", True
# )

# "2021-10-04", "00-24-52"  #
# "2021-10-06", "12-08-14"   # 1000 runs
results_list_flat_clair_std = results.Results.load_results(
    "2021-10-06", "12-08-14", True
)

# results_list_flat_A = results.Results.load_results(
#     "2021-10-03", "22-52-37", True
# )
# results_list_flat_B = results.Results.load_results(
#     "2021-10-04", "00-05-01", True
# )

po = plotter.Plotter()

test_combined_results(results_list_flat_rates_std, po, True)

# fig_ax_out = test_combined_results(results_list_flat_rates_std, po, False)
# tmp = test_combined_results(
#     results_list_flat_clair_std, po, True, fig_ax_out, clairvoyant=True
# )
