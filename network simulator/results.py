from __future__ import annotations
import copy as cp
import networkx as nx
import pickle
import typing as tp
import typing_extensions as tp_ext
import collections as clct
import customtypes as ctp
import simulator
import tools

# import plotter

Simulator = simulator.Simulator


class Results:
    # These correspond to the possible tests
    # They serve as a way to retreive results by test
    class ResultsTags(tp_ext.TypedDict):
        graph: float
        attacker: float

    def __init__(
        self,
        sims: tp.List[Simulator],
        keep_sims: bool = True,
        keep_repgraphs: bool = True,
        repgraph_properties: bool = True,
        lightweight: bool = False,
        # lightweight: bool = False,
    ):
        self.is_instance_Results = None
        self.lightweight = lightweight
        # TODO this sim and repgraph stuff is a mess, sort it out
        # Save sims to Results object
        self.sims = sims
        # save reference repgraph to Results
        self.repgraphs = [sim.repgraph for sim in sims]
        # self.repgraph_name = self.repgraph.name
        sim: Simulator
        # TODO
        # Delete the sims' reference to the repgraphs
        # Now only self.repgraphs refers to the repgraps
        for sim in sims:
            del sim.repgraph
        # Data object
        # # Data Dict
        self.data: ctp.ResultsDataDict
        # # Sim data
        if sims == []:
            self.data = {
                "sim_data": {},
                "repgraph_data": {},
                "users_data": {},
                "attacker_data": {},
            }
        elif lightweight:
            sim_data_stop_condition = [sim.stop_condition for sim in sims]
            sim_data_runtime_list = [sim.runtime for sim in sims]
            sim_data_runtime_stats = tools.basic_stats(sim_data_runtime_list)
            sim_data_attack_ordered = [sim.attack_ordered for sim in sims[0:1]]
            sim_data_clairvoyance = [sim.clairvoyance for sim in sims[0:1]]
            users_data_oracle_accepts_num_deals = [
                sim.oracle_accepts_num_deals for sim in sims
            ]
            users_data_oracle_accepts_num_cheats = [
                sim.oracle_accepts_num_cheats for sim in sims
            ]
            users_data_oracle_accepts_deal_stats = tools.basic_stats(
                [i for i in users_data_oracle_accepts_num_deals]
            )
            users_data_oracle_accepts_cheat_stats = tools.basic_stats(
                [i for i in users_data_oracle_accepts_num_cheats]
            )
            users_data_oracle_rejects_num_deals = [
                sim.oracle_rejects_num_deals for sim in sims
            ]
            users_data_oracle_rejects_num_cheats = [
                sim.oracle_rejects_num_cheats for sim in sims
            ]
            users_data_oracle_rejects_deal_stats = tools.basic_stats(
                [i for i in users_data_oracle_rejects_num_deals]
            )
            users_data_oracle_rejects_cheat_stats = tools.basic_stats(
                [i for i in users_data_oracle_rejects_num_cheats]
            )

            # # Attacker data
            attacker_data_name = [sim.attacker.name for sim in sims[0:1]]
            attacker_data_attacker_type_at_creation = [
                sim.attacker.attacker_type_at_creation for sim in sims[0:1]
            ]
            attacker_data_attacker_type = [
                sim.attacker.attacker_type for sim in sims[0:1]
            ]
            attacker_data_seq_length = [sim.total_attack_idx for sim in sims]
            attacker_data_seq_length_stats = tools.basic_stats(
                attacker_data_seq_length
            )
            attacker_data_last_accept_idcs = [
                sim.last_accept_idx for sim in sims
            ]
            attacker_data_last_accept_idcs_stats = tools.basic_stats(
                attacker_data_last_accept_idcs
            )
            attacker_data_last_accept_action = [
                sim.last_accept_action for sim in sims
            ]
            action_count_tmp: tp.Dict[ctp.attacker_action, int] = dict(
                clct.Counter(attacker_data_last_accept_action)
            )
            action_count_tmp["skip"] = 0
            attacker_data_last_accept_action_count: ctp.IntPerAction = tp.cast(
                ctp.IntPerAction, action_count_tmp
            )
            attacker_data_last_accept_time = [
                sim.last_accept_time for sim in sims
            ]
            attacker_data_last_accept_time_stats = tools.basic_stats(
                attacker_data_last_accept_time
            )
            attacker_data_rate = [sim.attacker.rate for sim in sims[0:1]]
            attacker_data_back_off = [
                sim.attacker.back_off for sim in sims[0:1]
            ]
            attacker_data_flip_victim_weight = [
                sim.attacker.flip_victim_weight for sim in sims[0:1]
            ]
            attacker_data_action_probabilities = [
                sim.attacker.action_probabilities for sim in sims[0:1]
            ]
            attacker_data_action_caps = [sim.action_caps for sim in sims[0:1]]
            attacker_data_profits = [
                sim.oracle_messages_num_cheats for sim in sims
            ]
            attacker_data_lag_profits = [
                profits - oracle_accepts_cheats
                for profits, oracle_accepts_cheats in zip(
                    attacker_data_profits,
                    users_data_oracle_accepts_num_cheats,
                )
            ]
            attacker_data_expenditure = [
                sim.oracle_messages_num_deals for sim in sims
            ]
            attacker_data_lag_expenditure = [
                expenditure - oracle_accepts_deals
                for expenditure, oracle_accepts_deals in zip(
                    attacker_data_expenditure,
                    users_data_oracle_accepts_num_deals,
                )
            ]
            attacker_data_rejects_deals = [
                sim.rejects_num_deals for sim in sims
            ]
            attacker_data_rejects_cheats = [
                sim.rejects_num_cheats for sim in sims
            ]
            attacker_data_attempts_deals = [
                accepts + rejects
                for accepts, rejects in zip(
                    attacker_data_expenditure, attacker_data_rejects_deals
                )
            ]
            attacker_data_attempts_cheats = [
                accepts + rejects
                for accepts, rejects in zip(
                    attacker_data_profits, attacker_data_rejects_cheats
                )
            ]
            attacker_data_profits_stats = tools.basic_stats(
                attacker_data_profits
            )
            attacker_data_lag_profits_stats = tools.basic_stats(
                attacker_data_lag_profits
            )
            attacker_data_expenditure_stats = tools.basic_stats(
                attacker_data_expenditure
            )
            attacker_data_lag_expenditure_stats = tools.basic_stats(
                attacker_data_lag_expenditure
            )
            attacker_data_rejects_deals_stats = tools.basic_stats(
                attacker_data_rejects_deals
            )
            attacker_data_rejects_cheats_stats = tools.basic_stats(
                attacker_data_rejects_cheats
            )
            attacker_data_attempts_deals_stats = tools.basic_stats(
                attacker_data_attempts_deals
            )
            attacker_data_attempts_cheats_stats = tools.basic_stats(
                attacker_data_attempts_cheats
            )
            print(
                tools.basic_stats(
                    [sim.runtime - sim.last_accept_time for sim in sims]
                )
            )
            self.data = {
                "sim_data": {
                    "stop_condition": sim_data_stop_condition,
                    "runtime_stats": sim_data_runtime_stats,
                    "spread_time_stats": tools.basic_stats(
                        [sim.runtime - sim.last_accept_time for sim in sims]
                    ),
                    "attack_ordered": sim_data_attack_ordered,
                    "clairvoyance": sim_data_clairvoyance,
                    "messages": [sim.oracle_user.messages for sim in sims],
                },
                # TODO
                # ! This is a temporary test for now
                "repgraph_data": {
                    "name": [g.name for g in self.repgraphs[0:1]]
                    # "dc": [sim.dc for sim in sims[0:1]],
                },
                "users_data": {
                    "oracle_accepts_deals": (
                        users_data_oracle_accepts_num_deals
                    ),
                    "oracle_accepts_deal_stats": (
                        users_data_oracle_accepts_deal_stats
                    ),
                    "oracle_accepts_cheats": (
                        users_data_oracle_accepts_num_cheats
                    ),
                    "oracle_accepts_cheat_stats": (
                        users_data_oracle_accepts_cheat_stats
                    ),
                    "oracle_rejects_deals": (
                        users_data_oracle_rejects_num_deals
                    ),
                    "oracle_rejects_deal_stats": (
                        users_data_oracle_rejects_deal_stats
                    ),
                    "oracle_rejects_cheats": (
                        users_data_oracle_rejects_num_cheats
                    ),
                    "oracle_rejects_cheat_stats": (
                        users_data_oracle_rejects_cheat_stats
                    ),
                },
                "attacker_data": {
                    "name": attacker_data_name[0:1],
                    "attacker_type_at_creation": (
                        attacker_data_attacker_type_at_creation[0:1]
                    ),
                    "attacker_type": attacker_data_attacker_type[0:1],
                    "seq_length_stats": attacker_data_seq_length_stats,
                    "last_accept_idcs_stats": (
                        attacker_data_last_accept_idcs_stats
                    ),
                    "last_accept_action_count": (
                        attacker_data_last_accept_action_count
                    ),
                    "last_accept_time_stats": (
                        attacker_data_last_accept_time_stats
                    ),
                    "rate": attacker_data_rate[0:1],
                    "back_off": attacker_data_back_off[0:1],
                    "flip_victim_weight": attacker_data_flip_victim_weight[0:1],
                    # "attack_seqs": attacker_data_attack_seqs,
                    "action_probabilities": attacker_data_action_probabilities[
                        0:1
                    ],
                    "action_caps": attacker_data_action_caps[0:1],
                    "profits": attacker_data_profits,
                    "profits_stats": attacker_data_profits_stats,
                    "lag_profits_stats": attacker_data_lag_profits_stats,
                    "expenditure": attacker_data_expenditure,
                    "expenditure_stats": attacker_data_expenditure_stats,
                    "lag_expenditure_stats": (
                        attacker_data_lag_expenditure_stats
                    ),
                    "rejects_deals_stats": attacker_data_rejects_deals_stats,
                    "rejects_deals": attacker_data_rejects_deals,
                    "rejects_cheats_stats": attacker_data_rejects_cheats_stats,
                    "rejects_cheats": attacker_data_rejects_cheats,
                    "attempts_deals_stats": attacker_data_attempts_deals_stats,
                    "attempts_cheats_stats": (
                        attacker_data_attempts_cheats_stats
                    ),
                },
            }
        else:
            sim_data_users_trust_histories = [
                sim.users_trust_history for sim in sims
            ]
            sim_data_terminating_indices: tp.List[int] = [
                sim.terminating_event_index for sim in sims
            ]
            sim_data_terminating_indices_stats = tools.basic_stats(
                sim_data_terminating_indices
            )
            # # Repgraphs data
            repgraph_data_name = [g.name for g in self.repgraphs]
            repgraph_data_average_clustering_per_graph = [
                nx.average_clustering(g.to_undirected())
                if repgraph_properties
                else None
                for g in self.repgraphs
            ]
            repgraph_data_average_clustering_stats = tools.basic_stats(
                repgraph_data_average_clustering_per_graph
            )
            # # TODO These are currently commented out
            # # as they take too long.
            # self.repgraph_sigmas = [
            #     nx.sigma(g.to_undirected())
            #     for g in self.repgraphs[0:1]
            # ]
            # self.repgraph_omegas = [
            #     nx.omega(g.to_undirected())
            #     for g in self.repgraphs[0:1]
            # ]
            # # User data
            users_data_user_thresholds = tp.cast(
                tp.List[int], [sim.user_threshold for sim in sims]
            )
            users_data_user_thresholds_stats = tools.basic_stats(
                users_data_user_thresholds
            )
            users_data_num_weighted_cheats = [
                [user.num_weighted_cheats(own=False) for user in sim.users]
                for sim in sims
            ]
            users_data_num_own_weighted_cheats = [
                [user.num_weighted_cheats(own=True) for user in sim.users]
                for sim in sims
            ]
            users_data_oracle_accepts = [sim.oracle_accepts for sim in sims]
            users_data_oracle_rejects = [sim.oracle_rejects for sim in sims]
            users_data_oracle_accepts_deal_stats = tools.basic_stats(
                [len(i["deal"]) for i in users_data_oracle_accepts]
            )
            users_data_oracle_accepts_cheat_stats = tools.basic_stats(
                [len(i["cheat"]) for i in users_data_oracle_accepts]
            )
            users_data_oracle_rejects_deal_stats = tools.basic_stats(
                [len(i["deal"]) for i in users_data_oracle_rejects]
            )
            users_data_oracle_rejects_cheat_stats = tools.basic_stats(
                [len(i["cheat"]) for i in users_data_oracle_rejects]
            )
            # TODO
            # ! Rename this
            users_data_thing = [
                [user.cheats() for user in sim.users] for sim in sims
            ]
            # # Attacker data
            attacker_data_attacker_type_at_creation = [
                sim.attacker.attacker_type_at_creation for sim in sims
            ]
            attacker_data_attacker_type = [
                sim.attacker.attacker_type for sim in sims
            ]
            attacker_data_rate = [sim.attacker.rate for sim in sims]
            attacker_data_back_off = [sim.attacker.back_off for sim in sims]
            attacker_data_flip_victim_weight = [
                sim.attacker.flip_victim_weight for sim in sims
            ]
            attacker_data_attack_seqs = [sim.attacker.seq for sim in sims]
            attacker_data_action_probabilities = [
                sim.attacker.action_probabilities for sim in sims
            ]
            # TODO
            # ! Assumes cheats only
            attacker_data_profits = [
                len(sim.attacker.transactions["cheat"]) for sim in sims
            ]
            attacker_data_lag_profits = [
                len(sim.attacker.transactions["cheat"])
                - len(sim.oracle_accepts["cheat"])
                for sim in sims
            ]
            attacker_data_expenditure = [
                len(sim.attacker.transactions["deal"]) for sim in sims
            ]
            attacker_data_lag_expenditure = [
                len(sim.attacker.transactions["deal"])
                - len(sim.oracle_accepts["deal"])
                for sim in sims
            ]
            # TODO
            # ! Assumes cheats only
            attacker_data_rejects = [
                len(sim.attacker.rejects["cheat"]) for sim in sims
            ]
            # TODO
            # ! Assumes cheats only
            attacker_data_attempts = [
                len(sim.attacker.rejects["cheat"])
                + len(sim.attacker.transactions["cheat"])
                for sim in sims
            ]
            attacker_data_ratio_rejects_attempts = [
                len(sim.attacker.rejects["cheat"])
                / (
                    len(sim.attacker.rejects["cheat"])
                    + len(sim.attacker.transactions["cheat"])
                )
                if len(sim.attacker.rejects["cheat"]) > 0
                # TODO
                # ? Is this meaningful
                else 0
                for sim in sims
            ]
            attacker_data_accepts = [
                user_trust_history.count("accept")
                for user_trust_history in [
                    sim.users_trust_history for sim in sims
                ]
            ]
            attacker_data_lag_accepts = [
                user_trust_history.count("accept")
                - (len(oracle_accepts["deal"]) + len(oracle_accepts["cheat"]))
                for user_trust_history, oracle_accepts in zip(
                    [sim.users_trust_history for sim in sims],
                    [sim.oracle_accepts for sim in sims],
                )
            ]
            attacker_data_lag_rejects = [
                user_trust_history.count("reject")
                - (len(oracle_rejects["deal"]) + len(oracle_rejects["cheat"]))
                for user_trust_history, oracle_rejects in zip(
                    [sim.users_trust_history for sim in sims],
                    [sim.oracle_rejects for sim in sims],
                )
            ]
            attacker_data_transactions_per_user = [
                sim.attacker.transactions_per_user for sim in sims
            ]
            attacker_data_rejects_per_user = [
                sim.attacker.rejects_per_user for sim in sims
            ]
            attacker_data_profits_stats = tools.basic_stats(
                attacker_data_accepts
            )
            attacker_data_lag_profits_stats = tools.basic_stats(
                attacker_data_rejects
            )
            attacker_data_expenditure_stats = tools.basic_stats(
                attacker_data_profits
            )
            attacker_data_lag_expenditure_stats = tools.basic_stats(
                attacker_data_lag_profits
            )
            attacker_data_rejects_stats = tools.basic_stats(
                attacker_data_expenditure
            )
            attacker_data_attempts_stats = tools.basic_stats(
                attacker_data_lag_expenditure
            )
            attacker_data_ratio_rejects_attempts_stats = tools.basic_stats(
                attacker_data_attempts
            )
            attacker_data_accepts_stats = tools.basic_stats(
                attacker_data_ratio_rejects_attempts
            )
            attacker_data_lag_accepts_stats = tools.basic_stats(
                attacker_data_lag_accepts
            )
            attacker_data_lag_rejects_stats = tools.basic_stats(
                attacker_data_lag_rejects
            )
            self.data = {
                "sim_data": {
                    "users_trust_histories": sim_data_users_trust_histories,
                    "terminating_indices": sim_data_terminating_indices,
                    "terminating_indices_stats": (
                        sim_data_terminating_indices_stats
                    ),
                },
                "repgraph_data": {
                    "name": repgraph_data_name,
                    "average_clustering_per_graph": (
                        repgraph_data_average_clustering_per_graph
                    ),
                    "average_clustering_stats": (
                        repgraph_data_average_clustering_stats
                    ),
                    # # TODO These are currently commented out
                    # # as they take too long.
                    # self.repgraph_sigmas = [
                    #     nx.sigma(g.to_undirected())
                    #     for g in self.repgraphs[0:1]
                    # ]
                    # self.repgraph_omegas = [
                    #     nx.omega(g.to_undirected())
                    #     for g in self.repgraphs[0:1]
                    # ]
                },
                "users_data": {
                    "user_thresholds": users_data_user_thresholds,
                    "user_thresholds_stats": users_data_user_thresholds_stats,
                    "num_weighted_cheats": users_data_num_weighted_cheats,
                    "num_own_weighted_cheats": (
                        users_data_num_own_weighted_cheats
                    ),
                    "oracle_accepts": users_data_oracle_accepts,
                    "oracle_accepts_deal_stats": (
                        users_data_oracle_accepts_deal_stats
                    ),
                    "oracle_accepts_cheat_stats": (
                        users_data_oracle_accepts_cheat_stats
                    ),
                    "oracle_rejects": users_data_oracle_rejects,
                    "oracle_rejects_deal_stats": (
                        users_data_oracle_rejects_deal_stats
                    ),
                    "oracle_rejects_cheat_stats": (
                        users_data_oracle_rejects_cheat_stats
                    ),
                    # TODO
                    # ! Rename this
                    "thing": users_data_thing,
                },
                "attacker_data": {
                    "attacker_type_at_creation": (
                        attacker_data_attacker_type_at_creation
                    ),
                    "attacker_type": attacker_data_attacker_type,
                    "rate": attacker_data_rate,
                    "back_off": attacker_data_back_off,
                    "flip_victim_weight": attacker_data_flip_victim_weight,
                    "attack_seqs": attacker_data_attack_seqs,
                    "action_probabilities": attacker_data_action_probabilities,
                    # TODO
                    # ! Assumes cheats only
                    "profits": attacker_data_profits,
                    "lag_profits": attacker_data_lag_profits,
                    "expenditure": attacker_data_expenditure,
                    "lag_expenditure": attacker_data_lag_expenditure,
                    # TODO
                    # ! Assumes cheats only
                    "rejects": attacker_data_lag_rejects,
                    # TODO
                    # ! Assumes cheats only
                    "attempts": attacker_data_attempts,
                    "ratio_rejects_attempts": (
                        attacker_data_ratio_rejects_attempts
                    ),
                    "accepts": attacker_data_accepts,
                    "lag_accepts": attacker_data_lag_accepts,
                    "lag_rejects": attacker_data_lag_rejects,
                    "transactions_per_user": (
                        attacker_data_transactions_per_user
                    ),
                    "rejects_per_user": attacker_data_rejects_per_user,
                    "profits_stats": attacker_data_profits_stats,
                    "lag_profits_stats": attacker_data_lag_profits_stats,
                    "expenditure_stats": attacker_data_expenditure_stats,
                    "lag_expenditure_stats": (
                        attacker_data_lag_expenditure_stats
                    ),
                    "rejects_stats": attacker_data_rejects_stats,
                    "attempts_stats": attacker_data_attempts_stats,
                    "ratio_rejects_attempts_stats": (
                        attacker_data_ratio_rejects_attempts_stats
                    ),
                    "accepts_stats": attacker_data_accepts_stats,
                    "lag_accepts_stats": attacker_data_lag_accepts_stats,
                    "lag_rejects_stats": attacker_data_lag_rejects_stats,
                },
            }
        # Set tags
        self.tags: tp.List[str] = []
        # If the sims are to be deleted
        if not keep_sims:
            # Delete the sims
            del self.sims
        # If the reference repgraph is to be deleted
        if not keep_repgraphs:
            # Delete the repgraphs
            del self.repgraphs[1:]

    def get_data(self, data_type, data_value):  # type: ignore
        return self.data[data_type][data_value]  # type: ignore

    def sort(self, var_name):  # type: ignore
        var = getattr(self, var_name)
        setattr(self, var_name + "_sorted", sorted(var))

    @staticmethod
    def print_stats(results, attacker_types):  # type: ignore
        print()
        for idx, x in enumerate(results):
            print("%%%%%%%% {} %%%%%%%%".format(attacker_types[idx]))
            print()
            print("Profits:")
            print(x.profits_stats)
            print()
            print("rejects")
            print(x.rejects_stats)
            print()
            print("Terminating Indices")
            print(x.terminating_indices_stats)
            print()
            print()

    @staticmethod
    def filter_results(results, filter_keys):  # type: ignore
        results_out = cp.deepcopy(results)
        if isinstance(filter_keys, str):
            filter_keys = (filter_keys,)
        filter_keys = set(
            k for k in results.keys() for k_f in filter_keys if k_f in k
        )
        for k in filter_keys:
            results_out.pop(k)
        return results_out

    @staticmethod
    def print_results() -> None:
        pass

    @staticmethod
    def load_results(
        date: str,
        time: str,
        split: bool,
        fixed_idcs: tp.Optional[tp.List[int]] = None,
        first_file_idx: int = 0,
    ) -> tp.List[Results]:
        results_list_flat = []
        idx = first_file_idx
        if split:
            while True:
                try:
                    with open(
                        "saved_plots/"
                        + date
                        + " "
                        + time
                        + " "
                        + "plots/results_dict "
                        + str(idx)
                        + ".pickle",
                        "rb",
                    ) as f:
                        if (fixed_idcs is None) or (idx in fixed_idcs):
                            results_list_flat.append(pickle.load(f))
                    # print(idx)
                    idx += 1
                except FileNotFoundError:
                    # print(idx)
                    print("File {} does not exist.".format(idx))
                    break
        else:
            with open(
                "plots/"
                + date
                + " "
                + time
                + " "
                + "plots/results_dict"
                + ".pickle",
                "rb",
            ) as f:
                results_dict = pickle.load(f)
                results_list_k_v = tools.nested_dict_to_nested_list(
                    results_dict, True
                )
                results_list_flat = tools.get_results_objects(results_list_k_v)
        return results_list_flat

    @staticmethod
    def load_parameters(
        date: str,
        time: str,
        param_file: str,
    ) -> tp.Dict[str, tp.Any]:
        with open(
            "plots/"
            + date
            + " "
            + time
            + " "
            + "plots/"
            + param_file
            + ".pickle",
            "rb",
        ) as f:
            return tp.cast(tp.Dict[str, tp.Any], pickle.load(f))

    @staticmethod
    def merge_results(
        results_list: tp.List[Results], lightweight: bool
    ) -> Results:
        return_results_object = Results([], False, False, False, True)

        users_data_oracle_accepts = []
        users_data_oracle_rejects = []
        attacker_data_attacker_type_at_creation = []
        attacker_data_attacker_type = []
        attacker_data_rate = []
        attacker_data_action_probabilities = []
        attacker_data_profits = []
        attacker_data_lag_profits = []
        attacker_data_expenditure = []
        attacker_data_lag_expenditure = []

        for r in results_list:
            # Users Data
            users_data_oracle_accepts += r.data["users_data"]["oracle_accepts"]
            users_data_oracle_rejects += r.data["users_data"]["oracle_rejects"]
            # Attacker Data
            # attacker_type_at_creation
            attacker_data_attacker_type_at_creation += r.data["attacker_data"][
                "attacker_type_at_creation"
            ]
            # attacker_type
            attacker_data_attacker_type += r.data["attacker_data"][
                "attacker_type"
            ]
            # rate
            attacker_data_rate += r.data["attacker_data"]["rate"]
            # action_probabilities
            attacker_data_action_probabilities += r.data["attacker_data"][
                "action_probabilities"
            ]
            # profits

            attacker_data_profits += r.data["attacker_data"]["profits"]
            # lag_profits
            attacker_data_lag_profits += r.data["attacker_data"]["lag_profits"]
            # expenditure
            attacker_data_expenditure += r.data["attacker_data"]["expenditure"]
            # lag_expenditure
            attacker_data_lag_expenditure += r.data["attacker_data"][
                "lag_expenditure"
            ]

            users_data_oracle_accepts_deal_stats = tools.basic_stats(
                [len(i["deal"]) for i in users_data_oracle_accepts]
            )
            users_data_oracle_accepts_cheat_stats = tools.basic_stats(
                [len(i["cheat"]) for i in users_data_oracle_accepts]
            )

            users_data_oracle_rejects_deal_stats = tools.basic_stats(
                [len(i["deal"]) for i in users_data_oracle_rejects]
            )

            users_data_oracle_rejects_cheat_stats = tools.basic_stats(
                [len(i["cheat"]) for i in users_data_oracle_rejects]
            )

            attacker_data_profits_stats = tools.basic_stats(
                attacker_data_profits
            )

            attacker_data_lag_profits_stats = tools.basic_stats(
                attacker_data_lag_profits
            )

            attacker_data_expenditure_stats = tools.basic_stats(
                attacker_data_expenditure
            )

            attacker_data_lag_expenditure_stats = tools.basic_stats(
                attacker_data_lag_expenditure
            )
        if lightweight:
            return_results_object.data["users_data"][
                "oracle_accepts_deal_stats"
            ] = users_data_oracle_accepts_deal_stats
            return_results_object.data["users_data"][
                "oracle_accepts_cheat_stats"
            ] = users_data_oracle_accepts_cheat_stats
            return_results_object.data["users_data"][
                "oracle_rejects_deal_stats"
            ] = users_data_oracle_rejects_deal_stats
            return_results_object.data["users_data"][
                "oracle_rejects_cheat_stats"
            ] = users_data_oracle_rejects_cheat_stats
            return_results_object.data["attacker_data"][
                "rate"
            ] = attacker_data_rate[0:1]
            return_results_object.data["attacker_data"][
                "attacker_type"
            ] = attacker_data_attacker_type[0:1]
            return_results_object.data["attacker_data"][
                "attacker_type_at_creation"
            ] = attacker_data_attacker_type_at_creation[0:1]
            return_results_object.data["attacker_data"][
                "action_probabilities"
            ] = attacker_data_action_probabilities[0:1]
            return_results_object.data["attacker_data"][
                "profits_stats"
            ] = attacker_data_profits_stats
            return_results_object.data["attacker_data"][
                "lag_profits_stats"
            ] = attacker_data_lag_profits_stats
            return_results_object.data["attacker_data"][
                "expenditure_stats"
            ] = attacker_data_expenditure_stats
            return_results_object.data["attacker_data"][
                "lag_expenditure_stats"
            ] = attacker_data_lag_expenditure_stats
        elif return_results_object.lightweight:
            for r in results_list:
                # Users Data
                return_results_object.data["users_data"][
                    "oracle_accepts"
                ] += r.data["users_data"]["oracle_accepts"]
                return_results_object.data["users_data"][
                    "oracle_rejects"
                ] += r.data["users_data"]["oracle_rejects"]
                # Attacker Data
                # attacker_type_at_creation
                return_results_object.data["attacker_data"][
                    "attacker_type_at_creation"
                ] += r.data["attacker_data"]["attacker_type_at_creation"]
                # attacker_type
                return_results_object.data["attacker_data"][
                    "attacker_type"
                ] += r.data["attacker_data"]["attacker_type"]
                # rate
                return_results_object.data["attacker_data"]["rate"] += r.data[
                    "attacker_data"
                ]["rate"]
                # action_probabilities
                return_results_object.data["attacker_data"][
                    "action_probabilities"
                ] += r.data["attacker_data"]["action_probabilities"]
                # profits
                return_results_object.data["attacker_data"][
                    "profits"
                ] += r.data["attacker_data"]["profits"]
                # lag_profits
                return_results_object.data["attacker_data"][
                    "lag_profits"
                ] += r.data["attacker_data"]["lag_profits"]
                # expenditure
                return_results_object.data["attacker_data"][
                    "expenditure"
                ] += r.data["attacker_data"]["expenditure"]
                # lag_expenditure
                return_results_object.data["attacker_data"][
                    "lag_expenditure"
                ] += r.data["attacker_data"]["lag_expenditure"]

                return_results_object.data["users_data"][
                    "oracle_accepts_deal_stats"
                ] = tools.basic_stats(
                    [
                        len(i["deal"])
                        for i in return_results_object.data["users_data"][
                            "oracle_accepts"
                        ]
                    ]
                )
                return_results_object.data["users_data"][
                    "oracle_accepts_cheat_stats"
                ] = tools.basic_stats(
                    [
                        len(i["cheat"])
                        for i in return_results_object.data["users_data"][
                            "oracle_accepts"
                        ]
                    ]
                )

                return_results_object.data["users_data"][
                    "oracle_rejects_deal_stats"
                ] = tools.basic_stats(
                    [
                        len(i["deal"])
                        for i in return_results_object.data["users_data"][
                            "oracle_rejects"
                        ]
                    ]
                )

                return_results_object.data["users_data"][
                    "oracle_rejects_cheat_stats"
                ] = tools.basic_stats(
                    [
                        len(i["cheat"])
                        for i in return_results_object.data["users_data"][
                            "oracle_rejects"
                        ]
                    ]
                )

                return_results_object.data["attacker_data"][
                    "profits_stats"
                ] = tools.basic_stats(
                    return_results_object.data["attacker_data"]["profits"]
                )

                return_results_object.data["attacker_data"][
                    "lag_profits_stats"
                ] = tools.basic_stats(
                    return_results_object.data["attacker_data"]["lag_profits"]
                )

                return_results_object.data["attacker_data"][
                    "expenditure_stats"
                ] = tools.basic_stats(
                    return_results_object.data["attacker_data"]["expenditure"]
                )

                return_results_object.data["attacker_data"][
                    "lag_expenditure_stats"
                ] = tools.basic_stats(
                    return_results_object.data["attacker_data"][
                        "lag_expenditure"
                    ]
                )
        return return_results_object
