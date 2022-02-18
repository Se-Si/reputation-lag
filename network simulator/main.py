import basicsim
import setup
import IPython
import sys

ultratb = IPython.core.ultratb
sys.excepthook = ultratb.FormattedTB(
    mode="Plain", color_scheme="Linux", call_pdb=False
)
print_iter = basicsim.tools.print_iter
attacker_setup = setup.AttackerSetup()
test = setup.TestSetup(attacker_setup)
initial_test = "live_test_multi_atk"
parameters = {
    "test": test,
    "initial_test": initial_test,
}
test.plotter_object.pickle_save(parameters, "parameters")

# Run the test
results_dict = test.tester_object.live_test_graphs(
    test.repgraph_args_list,
    initial_test,
    test.atk_test_kwargs,
    test.print_info,
)

# results_dict = results_dict_graphs

# Alternative Test Result Formatting
# results_list_k_v = basicsim.tools.nested_dict_to_nested_list(
#     results_dict, True
# )
# results_list_flat = basicsim.tools.get_results_objects(results_list_k_v)
#
# for idx, r in enumerate(results_list_flat):
#     test.plotter_object.pickle_save(r, "results_dict" + " " + str(idx))

# compare_results_list = compare.get_compare_results_list(
#     results_list_flat,
#     [
#         ("attacker_data", "profits_stats", "mean"),
#         ("attacker_data", "lag_cheats_stats", "mean"),
#         ("users_data", "oracle_accepts_cheat_stats", "mean"),
#     ],
# )
#
# sorted_compare_results_list = sorted(
#     compare_results_list, key=lambda x: x[1][0]  # type:ignore
# )
#
# atk_types: tp.List[str]
# atk_profits_means: tp.List[float]
# atk_accepts_means: tp.List[float]
# atk_lag_cheats_means: tp.List[float]
# oracle_accepts_cheat_means: tp.List[float]
# atk_types, multi_values_list = zip(*sorted_compare_results_list)
# atk_profits_means = [x[0] for x in multi_values_list]
# atk_lag_cheats_means = [x[1] for x in multi_values_list]
# oracle_accepts_cheat_means = [x[2] for x in multi_values_list]
#
# profits_errorbars = compare.get_errors(
#     results_list_flat, "attacker_data", "profits_stats"
# )
#
# lag_cheats_errorbars = compare.get_errors(
#     results_list_flat, "attacker_data", "lag_cheats_stats"
# )
#
# oracle_accepts_cheat_errorbars = compare.get_errors(
#     results_list_flat, "users_data", "oracle_accepts_cheat_stats"
# )
#
# values_labels_zipped_list = [
#     (
#         atk_profits_means,
#         "Mean Profits",
#         profits_errorbars,
#     ),
#     # (
#     #     atk_lag_cheats_means,
#     #     "Mean Lag Profits",
#     #     lag_cheats_errorbars,
#     # ),
#     (
#         oracle_accepts_cheat_means,
#         "Mean Oracle Cheats",
#         oracle_accepts_cheat_errorbars,
#     ),
# ]
#
# values_list, values_label_list, values_errors_list = zip(
#     *values_labels_zipped_list
# )
#
# test.plotter_object.plot_bar(
#     atk_types,
#     values_list,
#     values_label_list,
#     "Cheats",
#     "Cheats per Attacker",
#     "profits_and_oracle_accepts_cheat",
#     error_values_list=values_errors_list,
# )

attacker_metrics = False
if attacker_metrics:
    pass
    # TODO
    # for x in results_list_flat:
    #     test.plotter_object.plot_bar(
    #         range(len(x.data["attacker_data"]["profits"])),
    #         [x.data["attacker_data"]["profits"]],
    #         x.data["attacker_data"]["attacker_type"][0][0],
    #     )
    # for idx, s in enumerate(x.sims):
    #     plot_name_temp = (
    #         x.data["attacker_data"]["attacker_type"][0][0] + " " + str(idx)
    #     )
    #     s.attacker.plot_seq(plot_name_temp, plotter_object)
    #     s.attacker.plot_weights(plot_name_temp, plotter_object)
    # compare.attacker_histogram(results_list_flat, test.plotter_object)
