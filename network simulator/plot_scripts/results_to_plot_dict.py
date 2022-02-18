import IPython
import sys
import os
import typing as tp
import customtypes as ctp
import tools

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

    # Load the desired list of results.
    results_list_flat: tp.Sequence[
        results.Results
    ] = results.Results.load_results(
        date, time, split, fixed_idcs, first_file_idx
    )

    # List of pairs.
    # Each pair corresponds to a result from `results_list_flat`.
    # The first element is the name of the attacker from that result.
    # The second element is the list of data entries for that result as
    # specified by the list of keyword-triples.
    compare_results_list: tp.List[
        tp.Tuple[str, tp.List[tp.Any]]
    ] = compare.get_compare_results_list(
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

    # TODO
    # ? Get rid of this
    # sorted_compare_results_list = sorted(
    #     compare_results_list, key=lambda x: x[1][0]  # type:ignore
    # )

    # Unzip the list of pairs into a pair of lists.
    # The first pair is the list of attacker names per result.
    # The second pair is the list of data entries per result.
    atk_types: tp.List[str]
    atk_types, data_entries_per_result = zip(*compare_results_list)

    # The first data entry is the income/profit mean.
    income_means: tp.List[float] = [x[0] for x in data_entries_per_result]
    income_errorbars: tp.List[tp.Tuple[float, float]] = compare.get_errors(
        results_list_flat, "attacker_data", "profits_stats"
    )
    # Construct a triple containing:
    # 1. The list of values per attacker/result.
    # 2. The list of errors per attacker/result for that data entry (the mean,
    #    in this case).
    income_ytriple: ctp.ytriple = (
        income_means,
        income_errorbars,
        "Rate: ",  # + atk_types[0].replace("_", " ").split()[0],
    )

    lag_income_means: tp.List[float] = [x[1] for x in data_entries_per_result]
    lag_income_errorbars: tp.List[tp.Tuple[float, float]] = compare.get_errors(
        results_list_flat, "attacker_data", "lag_profits_stats"
    )
    lag_income_ytriple: ctp.ytriple = (
        lag_income_means,
        lag_income_errorbars,
        "Lag Cheats ",  # + atk_types[0].replace("_", " ").split()[0],
    )

    oracle_income_means: tp.List[float] = [
        x[2] for x in data_entries_per_result
    ]
    oracle_income_errorbars: tp.List[
        tp.Tuple[float, float]
    ] = compare.get_errors(
        results_list_flat, "users_data", "oracle_accepts_cheat_stats"
    )
    oracle_income_ytriple: ctp.ytriple = (
        oracle_income_means,
        oracle_income_errorbars,
        "Oracle Cheats ",  # + atk_types[0].replace("_", " ").split()[0],
    )
    oracle_deals_means: tp.List[float] = [x[3] for x in data_entries_per_result]
    oracle_deals_errorbars: tp.List[
        tp.Tuple[float, float]
    ] = compare.get_errors(
        results_list_flat, "users_data", "oracle_accepts_deal_stats"
    )
    oracle_deals_ytriple: ctp.ytriple = (
        oracle_deals_means,
        oracle_deals_errorbars,
        "Oracle Deals ",  # + atk_types[0].replace("_", " ").split()[0],
    )

    expenditure_means: tp.List[float] = [x[4] for x in data_entries_per_result]
    expenditure_errorbars: tp.List[tp.Tuple[float, float]] = compare.get_errors(
        results_list_flat, "attacker_data", "expenditure_stats"
    )
    expenditure_ytriple: ctp.ytriple = (
        expenditure_means,
        expenditure_errorbars,
        "Rate: ",  # + atk_types[0].replace("_", " ").split()[0],
    )

    rejects_deals_means: tp.List[float] = [
        x[5] for x in data_entries_per_result
    ]
    rejects_deals_errorbars: tp.List[
        tp.Tuple[float, float]
    ] = compare.get_errors(
        results_list_flat, "attacker_data", "rejects_deals_stats"
    )
    rejects_deals_ytriple: ctp.ytriple = (
        rejects_deals_means,
        rejects_deals_errorbars,
        "Rate: ",  # + atk_types[0].replace("_", " ").split()[0],
    )

    rejects_cheats_means: tp.List[float] = [
        x[6] for x in data_entries_per_result
    ]
    rejects_cheats_errorbars: tp.List[
        tp.Tuple[float, float]
    ] = compare.get_errors(
        results_list_flat, "attacker_data", "rejects_cheats_stats"
    )
    rejects_cheats_ytriple: ctp.ytriple = (
        rejects_cheats_means,
        rejects_cheats_errorbars,
        "Rate: ",  # + atk_types[0].replace("_", " ").split()[0],
    )

    last_accept_idcs_means: tp.List[float] = [
        x[7] for x in data_entries_per_result
    ]
    last_accept_idcs_errorbars: tp.List[
        tp.Tuple[float, float]
    ] = compare.get_errors(
        results_list_flat, "attacker_data", "last_accept_idcs_stats"
    )
    last_accept_idcs_ytriple: ctp.ytriple = (
        last_accept_idcs_means,
        last_accept_idcs_errorbars,
        "Rate: ",  # + atk_types[0].replace("_", " ").split()[0],
    )

    # List of attacker rate per result.
    rates: tp.List[float] = [
        r.data["attacker_data"]["rate"][0] for r in results_list_flat
    ]
    # Each `*_w_varname` is a pair that represents a simulation variable to be
    # plotted along the x-axis.
    # The first element of the pair is a list of values for that variable per
    # result/attacker.
    # The second element is the name of that variable.
    rates_w_varname: ctp.xvals_w_varname[float] = (rates, "Rate")
    periods: tp.List[float] = [(1 / x) for x in rates]
    periods_w_varname: ctp.xvals_w_varname[float] = (periods, "Period")
    deals_cap: tp.List[int] = [
        r.data["attacker_data"]["action_caps"][0]["deal"]
        for r in results_list_flat
    ]
    deals_cap_w_varname: ctp.xvals_w_varname[int] = (deals_cap, "Deals Cap")
    skips_cap: tp.List[int] = [
        r.data["attacker_data"]["action_caps"][0]["skip"]
        for r in results_list_flat
    ]
    skips_cap_w_varname: ctp.xvals_w_varname[int] = (skips_cap, "Waiting Cap")
    atk_types_w_varname: ctp.xvals_w_varname[str] = (atk_types, "Attacker")
    graph_name: str = results_list_flat[0].data["repgraph_data"]["name"][0]

    # This dict contains the collected xvals and ytriples.
    # It also contains the graph type/name used for these results.
    results_prep_dict: ctp.prep_dict = {
        "income_ytriple": income_ytriple,
        "lag_income_ytriple": lag_income_ytriple,
        "oracle_income_ytriple": oracle_income_ytriple,
        "oracle_deals_ytriple": oracle_deals_ytriple,
        "expenditure_ytriple": expenditure_ytriple,
        "rejects_deals_ytriple": rejects_deals_ytriple,
        "rejects_cheats_ytriple": rejects_cheats_ytriple,
        "last_accept_idcs_ytriple": last_accept_idcs_ytriple,
        "rates_w_varname": rates_w_varname,
        "periods_w_varname": periods_w_varname,
        "deals_cap_w_varname": deals_cap_w_varname,
        "skips_cap_w_varname": skips_cap_w_varname,
        "atk_types_w_varname": atk_types_w_varname,
        "graph_name": graph_name,
    }
    return results_prep_dict


def prep_plot_dict(
    # List of dicts containing xvals and ytuples.
    results_prep_dict_list: tp.List[ctp.prep_dict],
    # The xvals to be plotted.
    xvals_type: str,
    # The yval to be plotted.
    yval_type: str,
    # The name of the yval to be plotted.
    ylabel: str,
    # The plot can be "std" (standard) plot or a "bar" plot.
    plot_type: str,
    # Title of the plot.
    plot_title: str,
    # Base string for the plot file name.
    filename_base: str,
    # The scale can either be "linear" or "log".
    xscale: str,
    yscale: str,
    # TODO
    # ? What does this do
    transpose: bool,
    # A suffix to be added to both x and y labels.
    labels_suffixes: tp.Optional[tp.List[str]] = None,
) -> tp.Dict[str, tp.Any]:
    # Extract a list of the specified xvals from each dict.
    xvals_w_varname_lst: tp.List[ctp.xvals_w_varname_any] = [
        tp.cast(ctp.xvals_w_varname_any, r_p_dict[xvals_type + "_w_varname"])
        for r_p_dict in results_prep_dict_list
    ]

    # Extract a list of the specified ytriple from each dict.
    ytriples: tp.List[ctp.ytriple] = [
        tp.cast(ctp.ytriple, r_p_dict[yval_type + "_ytriple"])
        for r_p_dict in results_prep_dict_list
    ]

    # Each element of this list is a sequence of yvals.
    # Each yval corresponds to an xval in `xvals`.
    yvals_lst: ctp.yvals

    # Each element of this list is a sequence of pairs.
    # Each pair is the +/- error of the corresponding value in `yvals_lst`.
    yvals_errs_pairs_lst: ctp.yerrs_pairs

    # Each element in this list is the name of the corresponding sequence of
    # yvalues in `yvals_lst`.
    yvals_labels: tp.List[str]

    # Unzip the list of ytriples to extract the above three lists.
    yvals_lst, yvals_errs_pairs_lst, yvals_labels = tp.cast(
        tp.Tuple[
            tp.List[tp.List[float]],
            tp.List[tp.List[tp.Tuple[float, float]]],
            tp.List[str],
        ],
        tuple(zip(*ytriples)),
    )
    yvals_lst = list(yvals_lst)
    yvals_errs_pairs_lst = list(yvals_errs_pairs_lst)
    yvals_labels = list(yvals_labels)

    xvals: ctp.xvals
    xlabel: str
    yvals_errs_split_lst: ctp.yerrs_split

    # TODO
    # ? Why is this here
    # old_yvals = yvals_lst
    # old_errs = yvals_errs_pairs_lst
    # TODO
    # ! Consider converting to np.arrays
    # TODO
    # ? What does this do
    if transpose:
        # raise ValueError("Transpose not implemented.")
        xvals, xlabels = list(zip(*xvals_w_varname_lst))
        xvals = [v[0] for v in xvals]
        xlabel = xlabels[0]
        yvals_lst = list(zip(*yvals_lst))
        yvals_errs_pairs_lst = list(zip(*yvals_errs_pairs_lst))
    else:
        # Get the first sequence of `xvals` from the list.
        # Get the corresponding label of the first sequence.
        xvals, xlabel = xvals_w_varname_lst[0]

        # Check that all the sequences and labels are the same.
        # This ensures that every sequence in `yvals` resulted from the same set
        # of `xvals`.
        xlabel_cmpr: str
        for xvals_cmpr, xlabel_cmpr in xvals_w_varname_lst:
            if not ((xvals == xvals_cmpr) and (xlabel == xlabel_cmpr)):
                raise ValueError("All vars should be the same")

        # Rearrange the sequences of error pairs into pairs of error sequences to
        # match the expected input of `pyplot.errorbar`.
        # ! The `yerr_pairs_to_split` function is used instead of the below list
        # ! comprehension for the sake of typing.
        # ! [list(zip(*e)) for e in yvals_errs_pairs_lst]
    yvals_errs_split_lst = tools.yerr_pairs_to_split(yvals_errs_pairs_lst)
    if labels_suffixes is not None:
        yvals_labels = [
            yvals_labels[0] + " " + labels_suffixes[idx]
            for idx in range(len(yvals_lst))
        ]
    else:
        yvals_labels = [yvals_labels[0] for idx in range(len(yvals_lst))]
    graph_name_lst = [
        r_p_dict["graph_name"] for r_p_dict in results_prep_dict_list
    ]
    graph_name = graph_name_lst.pop()
    for gn_tmp in graph_name_lst:
        if not (graph_name == gn_tmp):
            raise ValueError("All graph names should be the same")

    # A dict ready for plotting containing:
    # 1. The list of xvals.
    # 2. The lists of yvals.
    # 3. The lists of yvals errs.
    # 4. The list of yvals labels.
    # 5. The variable names for the x and y axes.
    # 6. Etc
    results_plot_dict = {
        "xvals": xvals,
        "yvals_lst": yvals_lst,
        # "old_yvals": old_yvals,
        "yvals_errs_split_lst": yvals_errs_split_lst,
        # "old_errs": old_errs,
        "yvals_labels": yvals_labels,
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
