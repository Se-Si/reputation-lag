# import customtypes as ctp
import typing as tp
import results
import plotter


def get_compare_results_list(
    results_list_flat: tp.Sequence[results.Results],
    kw_triples: tp.List[tp.Tuple[str, str, str]],
) -> tp.List[tp.Tuple[str, tp.List[tp.Any]]]:
    return_list = []
    for r in results_list_flat:
        attacker_name: str
        # If the attacker's name has been set in the results then get it.
        try:
            attacker_name = r.data["attacker_data"]["name"][0]
        # If not, construct the attacker's name from the relevant details.
        except KeyError:
            order_flag_bool = (
                r.data["attacker_data"]["attacker_type"][0][1] == "ordered"
            )
            deal_flag_bool = (
                r.data["attacker_data"]["action_probabilities"][0]["deal"] > 0
            )
            skip_flag_bool = (
                r.data["attacker_data"]["attacker_type"][0][1] == "skip"
            ) or (
                r.data["attacker_data"]["action_probabilities"][0]["skip"] > 0
            )
            order_skip_flag_bool = (
                r.data["attacker_data"]["attacker_type"][0][1] == "ordered_skip"
            )

            ord_flag = ""
            deal_flag = ""
            skip_flag = ""
            if order_flag_bool:
                ord_flag = "_ord"
            if deal_flag_bool:
                deal_flag = "_deal"
            if skip_flag_bool:
                skip_flag = "_skp"
            if order_skip_flag_bool:
                ord_flag = "_ord"
                skip_flag = "_skp"
            attacker_name = (
                r.data["attacker_data"]["attacker_type_at_creation"][0][0]
                + ord_flag
                + deal_flag
                + skip_flag
                + " "
                + str(r.data["attacker_data"]["rate"][0])
            )
        # Form a list of the desired data entries for the current results object.
        return_sub_list = [
            r.data[kw1][kw2][kw3] for kw1, kw2, kw3 in kw_triples
        ]
        # Append the above list to the list of lists of desired data.
        return_list.append((attacker_name, return_sub_list))
    return return_list


def attacker_histogram(
    results_list_flat: tp.List[results.Results], plotter_object: plotter.Plotter
) -> None:
    for results_obj in results_list_flat:
        profits = results_obj.data["attacker_data"]["profits"]
        attacker_name = (
            str(
                results_obj.data["attacker_data"]["attacker_type_at_creation"][
                    0
                ][0]
            )
            + " "
            + str(results_obj.data["attacker_data"]["rate"][0])
            + " "
            + (
                "d"
                if results_obj.data["attacker_data"]["action_probabilities"][0][
                    "deal"
                ]
                == 1
                else "nd"
            )
        )
        plotter_object.plot_histogram(
            attacker_name,
            profits,
            60,
        )


def get_errors(
    results_list_flat: tp.Sequence[results.Results], kw1: str, kw2: str
) -> tp.List[tp.Tuple[float, float]]:
    # errmaxs: tp.List[float] = []
    # errmins: tp.List[float] = []
    errs_lst: tp.List[tp.Tuple[float, float]] = []
    for x in results_list_flat:
        mu = x.data[kw1][kw2]["mean"]  # type:ignore
        # TODO
        # ! Fix all this
        # sigma = x.data[kw1]["profits_stats"]["variance"]  # type:ignore
        # N = len(x.data[kw1]["profits"])
        # a, b = scipy.stats.norm.interval(
        #     0.95,
        #     loc=mu,
        #     scale=(sigma / np.sqrt(N)),
        # )
        # err = (b - a) / 2
        min_val = x.data[kw1][kw2]["tenth"]  # type:ignore
        max_val = x.data[kw1][kw2]["ninetieth"]  # type:ignore
        # TODO
        # ! Debug
        # errmins.append(str(mu - min_val) + "min")
        # errmaxs.append(str(max_val - mu) + "max")
        # errmin = str(mu - min_val) + "min"
        # errmax = str(max_val - mu) + "max"
        errmin = mu - min_val
        errmax = max_val - mu
        errs_lst.append((errmin, errmax))
    return errs_lst


def test_combined_results(
    results_list_flat: tp.List[results.Results],
    po: plotter.Plotter,
    finish_plot: bool,
    fig_ax_in: tp.Optional[tp.Tuple[tp.Any, tp.Any]] = None,
) -> tp.Optional[tp.Tuple[tp.Any, tp.Any]]:
    compare_results_list = get_compare_results_list(
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
    atk_profits_means: tp.List[float]
    atk_accepts_means: tp.List[float]
    atk_lag_profits_means: tp.List[float]
    oracle_accepts_cheat_means: tp.List[float]
    atk_types, multi_values_list = zip(*sorted_compare_results_list)
    atk_profits_means = [x[0] for x in multi_values_list]
    atk_lag_profits_means = [x[1] for x in multi_values_list]
    oracle_accepts_cheat_means = [x[2] for x in multi_values_list]

    profits_errorbars = get_errors(
        results_list_flat, "attacker_data", "profits_stats"
    )

    lag_profits_errorbars = get_errors(
        results_list_flat, "attacker_data", "lag_profits_stats"
    )

    oracle_accepts_cheat_errorbars = get_errors(
        results_list_flat, "users_data", "oracle_accepts_cheat_stats"
    )

    values_labels_zipped_list = [
        (
            atk_profits_means,
            "Mean Cheats",
            profits_errorbars,
        ),
        (
            atk_lag_profits_means,
            "Mean Lag Cheats",
            lag_profits_errorbars,
        ),
        (
            oracle_accepts_cheat_means,
            "Mean Oracle Cheats",
            oracle_accepts_cheat_errorbars,
        ),
    ]

    values_list, values_label_list, values_errors_list = zip(
        *values_labels_zipped_list
    )
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

    plot_return = po.plot_results(
        variables,
        values_list,
        values_errors_list,
        values_label_list,
        xlabel,
        ylabel,
        plot_type,
        title,
        plot_name,
        xscale,
        yscale,
        finish_plot=finish_plot,
        fig_ax_in=fig_ax_in,
    )
    return plot_return
