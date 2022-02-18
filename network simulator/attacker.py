from __future__ import annotations


# TODO typing
import networkx as nx
import numpy as np
import inspect
import collections
import copy
import typing as tp
import customtypes as ctp

# import plotter
import tools


Counter = collections.Counter


class Attacker:
    actions: ctp.attacker_action_list = [
        "deal",
        "skip",
        "cheat",
    ]
    actions_set: ctp.attacker_action_set = {
        "deal",
        "skip",
        "cheat",
    }
    attacker_metrics: tp.Tuple[str, ...] = (
        "rnd",
        "flat",
        "linear",
        "dc",
        "spbc",
        "cc",
        "ec",
        "enbc",
        "eebc",
        "enebc",
        "kc",
        "kebc",
    )

    def __init__(
        self,
        repgraph: ctp.RepGraph,
        attacker_type: tp.Optional[
            tp.Tuple[
                str,
                tp.Optional[
                    tp.Union[
                        tp.Callable[[tp.Sequence[float]], tp.Sequence[float]],
                        str,
                    ]
                ],
            ]
        ] = None,
        action_probabilities: tp.Optional[ctp.FloatPerAction] = None,
        action_caps: tp.Optional[ctp.IntPerAction] = None,
        back_off: tp.Optional[bool] = None,
        rate: tp.Any = None,
        use_victim_weight_prob: tp.Optional[ctp.FloatPerAction] = None,
        flip_victim_weight: bool = False,
        seq: tp.Optional[tp.List[tp.Tuple[str, int]]] = None,
        attacker_type_at_creation: tp.Optional[
            tp.Tuple[
                str,
                tp.Optional[
                    tp.Union[
                        tp.Callable[[tp.Sequence[float]], tp.Sequence[float]],
                        str,
                    ]
                ],
            ]
        ] = None,
    ) -> None:
        # Initialize random number generator
        self.rnd_gen = np.random.default_rng()
        # ! Never let the attacker modify their self.repgraph attribute
        # ! It is literally the repgraph object of the simulator
        # repgraph
        self.repgraph: ctp.RepGraph = repgraph
        self.victim_trust_status = np.ones(len(repgraph.nodes), dtype="bool")
        self.users_trust_state = [True] * len(repgraph.nodes)
        self.attacker_type_at_creation: tp.Tuple[
            str,
            tp.Optional[
                tp.Union[
                    tp.Callable[[tp.Sequence[float]], tp.Sequence[float]], str
                ]
            ],
        ]
        self.attacker_type: tp.Tuple[
            str,
            tp.Optional[
                tp.Union[
                    tp.Callable[[tp.Sequence[float]], tp.Sequence[float]], str
                ]
            ],
        ]
        if attacker_type is not None:
            if attacker_type_at_creation is not None:
                self.attacker_type_at_creation = attacker_type_at_creation
            else:
                self.attacker_type_at_creation = attacker_type
            self.attacker_type = attacker_type
        else:
            if attacker_type_at_creation is not None:
                self.attacker_type_at_creation = attacker_type_at_creation
            else:
                self.attacker_type_at_creation = ("rnd", None)
            self.attacker_type = ("rnd", None)
        if action_probabilities is not None:
            action_probabilities_sum: float = sum(
                tp.cast(tp.List[float], list(action_probabilities.values()))
            )
            self.action_probabilities: ctp.FloatPerAction = tp.cast(
                ctp.FloatPerAction,
                {
                    action: action_probabilities[action]
                    / action_probabilities_sum
                    for action in self.actions
                },
            )
        else:
            self.action_probabilities = {
                "deal": 1 / 3,
                "skip": 1 / 3,
                "cheat": 1 / 3,
            }
        if action_caps is not None:
            self.action_caps = action_caps
        else:
            self.action_caps = {}
        if back_off is not None:
            self.back_off = back_off
        else:
            self.back_off = False
        if rate is not None:
            self.rate = rate
        else:
            # self.rate = self.rnd_gen.uniform(1, 2)
            self.rate = 1
        self.use_victim_weight_prob: ctp.FloatPerAction
        if use_victim_weight_prob is not None:
            self.use_victim_weight_prob = use_victim_weight_prob
        else:
            self.use_victim_weight_prob = {
                "deal": 1,
                "skip": 0,
                "cheat": 1,
            }
        self.flip_victim_weight = flip_victim_weight
        if seq is not None:
            self.seq = seq
        else:
            self.seq = []
        self.transactions: ctp.EventsPerAction = {"deal": [], "cheat": []}
        self.rejects: ctp.EventsPerAction = {"deal": [], "cheat": []}
        self.transactions_per_user: ctp.NodesPerAction = {
            "deal": None,
            "cheat": None,
        }
        self.rejects_per_user: ctp.NodesPerAction = {
            "deal": None,
            "cheat": None,
        }
        self.victim_weights: np.ndarray
        if self.attacker_type[0] == "from_seq":
            self.victim_weights = self.get_victim_weights(  # type:ignore
                self.attacker_type_at_creation,
                self.repgraph,
                # as_dict=False,
                only_values=True,
            )
        else:
            self.victim_weights = self.get_victim_weights(  # type:ignore
                self.attacker_type,
                self.repgraph,
                # as_dict=False,
                only_values=True,
            )
        self.victim_weights_complement = 1 / self.victim_weights
        self.victim_weights /= self.victim_weights.sum()
        self.victim_weights_complement /= self.victim_weights_complement.sum()
        self.victim_weights_mod = self.victim_weights.copy()
        self.victim_weights_complement_mod = (
            self.victim_weights_complement.copy()
        )
        self.name = self.create_name()

    def get_victim_weights(  # type:ignore
        self,
        attacker_type: tp.Tuple[
            str,
            tp.Optional[
                tp.Union[
                    tp.Callable[[tp.Sequence[float]], tp.Sequence[float]], str
                ]
            ],
        ],
        repgraph,
        # as_dict: bool = False,
        only_values: bool = True,
    ) -> tp.Union[tp.Dict[tp.Any, float], np.ndarray]:
        weights: tp.Union[tp.Dict[tp.Any, float], np.ndarray]
        if attacker_type[0] == "rnd":
            weights_temp = {
                k: self.rnd_gen.uniform() for k in range(len(repgraph.nodes))
            }
        else:
            metric = getattr(self, "get_victim_weights_" + attacker_type[0])
            weights_temp = tp.cast(
                tp.Dict[tp.Any, float],
                metric(repgraph),
            )
        if (attacker_type[1] is None) or isinstance(attacker_type[1], str):
            if only_values:
                weights = np.fromiter(  # type:ignore
                    weights_temp.values(), dtype="float"
                )
            else:
                weights = weights_temp
        else:
            # Calculate weights to force
            weights_force_function: tp.Callable[
                [tp.Sequence[float]], tp.Sequence[float]
            ] = attacker_type[1]
            weights_to_force: tp.List[float] = list(
                weights_force_function(list(weights_temp.values()))
            )
            # Ensure we have a list of `(idx, val)` pairs
            # # where `idx` is a node index
            # # where `val` is the original weight of that node
            weights_temp_indexed_values: tp.List[tp.Tuple[int, float]] = [
                (i, v) for i, v in enumerate(weights_temp.values())
            ]
            # Resort the indexed original weights by the weight values
            weights_temp_indexed_values = sorted(
                weights_temp_indexed_values,
                key=lambda x: x[1],
            )
            # Zip the resorted original weights and the forced weights together
            weights_zipped: tp.List[
                tp.Tuple[tp.Tuple[int, float], float]
            ] = sorted(
                zip(weights_temp_indexed_values, weights_to_force),
                key=lambda x: x[0][0],
            )
            weights_unzipped: tp.Tuple[
                tp.Tuple[tp.Tuple[int, float], ...], tp.Tuple[float, ...]
            ]
            weights_unzipped = tp.cast(
                tp.Tuple[
                    tp.Tuple[tp.Tuple[int, float], ...], tp.Tuple[float, ...]
                ],
                tuple(zip(*weights_zipped)),
            )
            weights = np.fromiter(  # type:ignore
                weights_unzipped[1], dtype=float
            )
            # Resort the zipped weights by the node indices
            # Then zip them together
            # Sort by the weights_temp
            # Output the sorted weights forced
        return weights

    @staticmethod
    def get_victim_weights_flat(repgraph):  # type:ignore
        return {k: 1 for k in range(len(repgraph.nodes))}

    @staticmethod
    def get_victim_weights_linear(  # type:ignore
        repgraph, m: float = 1, c: int = 0
    ):
        return {k: (k * m + c) for k in range(len(repgraph.nodes))}

    @staticmethod
    def get_victim_weights_dc(repgraph):  # type:ignore
        return repgraph.dc

    @staticmethod
    def get_victim_weights_spbc(  # type:ignore
        repgraph,
    ):
        return repgraph.spbc

    @staticmethod
    def get_victim_weights_current_flow_betweenness_centrality(  # type:ignore
        repgraph,
    ):
        return nx.current_flow_betweenness_centrality(
            repgraph.to_undirected(), weight="period"
        )

    @staticmethod
    def get_victim_weights_cc(repgraph):  # type:ignore
        return repgraph.cc

    @staticmethod
    def get_victim_weights_ec(repgraph):  # type:ignore
        return repgraph.ec

    @staticmethod
    # eigenvector_node_betweenness_centrality
    def get_victim_weights_enbc(  # type:ignore
        repgraph,
    ):
        return nx.eigenvector_centrality(
            repgraph,
            weight="period",
            max_iter=100000,
            nstart=nx.betweenness_centrality(
                repgraph, weight="period", endpoints=True
            ),
        )

    @staticmethod
    # eigenvector_edge_betweenness_centrality
    def get_victim_weights_eebc(repgraph):  # type:ignore
        return nx.eigenvector_centrality_numpy(
            repgraph,
            weight="edge_betweenness",
            max_iter=100000,
        )

    @staticmethod
    # Eigenvector Node+Edge Betweenness Centrality
    def get_victim_weights_enebc(repgraph):  # type:ignore
        return nx.eigenvector_centrality(
            repgraph,
            weight="edge_betweenness",
            max_iter=100000,
            nstart=nx.betweenness_centrality(
                repgraph, weight="period", endpoints=True
            ),
        )

    @staticmethod
    def get_victim_weights_kc(repgraph):  # type:ignore
        return nx.katz_centrality(
            repgraph,
            weight="period",
            max_iter=100000,
        )

    @staticmethod
    def get_victim_weights_kebc(  # type:ignore
        repgraph,
    ):
        return nx.katz_centrality(
            repgraph,
            weight="edge_betweenness",
            max_iter=100000,
        )

    # def get_victim_weights_closeness_centrality(self):
    #     return np.fromiter(
    #         nx.closeness_centrality(
    #           self.repgraph, distance='period'
    #         ).values(),
    #         dtype='float'
    #     )

    @staticmethod
    def force_victim_weights_reciprocal_down(
        weights: tp.Sequence[float], base: int
    ) -> tp.Sequence[float]:
        return [1 / (base ** (i + 1)) for i in range(len(weights))]

    @classmethod
    def force_victim_weights_reciprocal_up(
        cls, weights: tp.Sequence[float], base: int
    ) -> tp.Sequence[float]:
        return sorted(cls.force_victim_weights_reciprocal_down(weights, base))

    @staticmethod
    def force_victim_weights_power_down(
        weights: tp.Sequence[float], power: float
    ) -> tp.Sequence[float]:
        return [((i + 1) ** -power) for i in range(len(weights))]

    @classmethod
    def force_victim_weights_power_up(
        cls, weights: tp.Sequence[float], power: float
    ) -> tp.Sequence[float]:
        return sorted(cls.force_victim_weights_power_down(weights, power))

    @staticmethod
    def force_victim_weights_linear_up(
        weights: tp.Sequence[float], slope: float
    ) -> tp.Sequence[float]:
        if not (slope > 0):
            raise ValueError("slope must be greater than 0")
        return [((i + 1) * slope) for i in range(len(weights))]

    @staticmethod
    def force_victim_weights_linear_down(
        weights: tp.Sequence[float], slope: float
    ) -> tp.Sequence[float]:
        if not (slope > 0):
            raise ValueError("slope must be greater than 0")
        return sorted(
            [((i + 1) * slope) for i in range(len(weights))], reverse=True
        )

    @staticmethod
    def force_victim_weights_sigmoid_up(
        weights: tp.Sequence[float],
        spread: float,
        shift: tp.Optional[int] = None,
    ) -> tp.Sequence[float]:
        return tools.sigmoid_distribution(len(weights), spread, shift)

    @classmethod
    def force_victim_weights_sigmoid_down(
        cls,
        weights: tp.Sequence[float],
        spread: float,
        shift: tp.Optional[int] = None,
    ) -> tp.Sequence[float]:
        return sorted(
            cls.force_victim_weights_sigmoid_up(weights, spread, shift),
            reverse=True,
        )

    def seq_gen_live(self) -> tp.Tuple[str, int]:
        action_probability_weights: tp.Sequence[float] = [
            self.action_probabilities[action] for action in self.actions
        ]
        attack_action = self.rnd_gen.choice(
            self.actions, p=action_probability_weights
        )
        if self.back_off:
            self.set_back_off()  # type:ignore
        if attack_action == "deal":
            if (self.use_victim_weight_prob["deal"] == 1) or (
                self.rnd_gen.random() < self.use_victim_weight_prob["deal"]
            ):
                if not self.flip_victim_weight:
                    attack_victim = int(
                        self.rnd_gen.choice(
                            self.repgraph.nodes, p=self.victim_weights_mod
                        )
                    )
                else:
                    attack_victim = int(
                        self.rnd_gen.choice(
                            self.repgraph.nodes,
                            p=self.victim_weights_complement_mod,
                        )
                    )
            else:
                attack_victim = int(self.rnd_gen.choice(self.repgraph.nodes))
        elif attack_action == "cheat":
            if (self.use_victim_weight_prob["cheat"] == 1) or (
                self.rnd_gen.random() < self.use_victim_weight_prob["cheat"]
            ):
                if not self.flip_victim_weight:
                    attack_victim = int(
                        self.rnd_gen.choice(
                            self.repgraph.nodes,
                            p=self.victim_weights_complement_mod,
                        )
                    )
                else:
                    attack_victim = int(
                        self.rnd_gen.choice(
                            self.repgraph.nodes, p=self.victim_weights_mod
                        )
                    )
            else:
                attack_victim = int(self.rnd_gen.choice(self.repgraph.nodes))
        else:
            attack_victim = -2
        attack_seq_element: tp.Tuple[str, int]
        attack_seq_element = attack_action, attack_victim
        self.seq.append(attack_seq_element)
        return attack_seq_element

    # def seq_gen_chunk(self, granularity: int) -> tp.List[tp.Tuple[str, int]]:
    def seq_gen_chunk(
        self,
        granularity: int,
        force_actions: tp.Optional[ctp.attacker_action_set] = None,
        lightweight: bool = False,
        clairvoyance: bool = False,
        users_trust_state: tp.Optional[tp.List[bool]] = None,
    ) -> None:
        action_probability_weights: tp.Sequence[float]
        if force_actions is None:
            action_probability_weights = [
                self.action_probabilities[action] for action in self.actions
            ]
        else:
            action_probability_weights = [
                self.action_probabilities[action]
                if (action in force_actions)
                else 0
                for action in self.actions
            ]
            sum_act_prob = sum(action_probability_weights)
            action_probability_weights = [
                act_prob / sum_act_prob
                for act_prob in action_probability_weights
            ]
        attack_actions = self.rnd_gen.choice(
            self.actions, size=granularity, p=action_probability_weights
        )
        attack_actions_count = Counter(attack_actions)
        deal_count = attack_actions_count["deal"]
        skip_count = attack_actions_count["skip"]
        cheat_count = attack_actions_count["cheat"]
        if users_trust_state is not None:
            self.users_trust_state = users_trust_state.copy()
        if clairvoyance:
            victim_weights_mod_tmp = np.where(
                self.users_trust_state, self.victim_weights_mod, 0
            )
            if victim_weights_mod_tmp.sum() == 0:
                victim_weights_mod_tmp[:] = 1
            victim_weights_mod_tmp /= victim_weights_mod_tmp.sum()
            victim_weights_complement_mod_tmp = np.where(
                self.users_trust_state, self.victim_weights_complement_mod, 0
            )
            if victim_weights_complement_mod_tmp.sum() == 0:
                victim_weights_complement_mod_tmp[:] = 1
            victim_weights_complement_mod_tmp /= (
                victim_weights_complement_mod_tmp.sum()
            )
        else:
            victim_weights_mod_tmp = self.victim_weights_mod
            victim_weights_complement_mod_tmp = (
                self.victim_weights_complement_mod
            )
        if not self.flip_victim_weight:
            deal_victims = self.rnd_gen.choice(
                len(self.repgraph.nodes),
                size=deal_count,
                p=victim_weights_mod_tmp,
            )
        else:
            deal_victims = self.rnd_gen.choice(
                len(self.repgraph.nodes),
                size=deal_count,
                p=victim_weights_complement_mod_tmp,
            )
        if not self.flip_victim_weight:
            cheat_victims = list(
                self.rnd_gen.choice(
                    len(self.repgraph.nodes),
                    size=cheat_count,
                    p=victim_weights_complement_mod_tmp,
                )
            )
        else:
            cheat_victims = list(
                self.rnd_gen.choice(
                    len(self.repgraph.nodes),
                    size=cheat_count,
                    p=victim_weights_mod_tmp,
                )
            )
        attack_victims = np.empty(
            deal_count + skip_count + cheat_count, dtype=int
        )
        attack_victims[np.where(attack_actions == "deal")] = deal_victims
        attack_victims[np.where(attack_actions == "skip")] = -2
        attack_victims[np.where(attack_actions == "cheat")] = cheat_victims
        attack_seq_elements: tp.List[tp.Tuple[ctp.attacker_action, int]]
        attack_seq_elements = list(zip(attack_actions, attack_victims))
        if lightweight:
            self.seq = attack_seq_elements  # type:ignore
        else:
            self.seq.extend(attack_seq_elements)
        del deal_victims
        del cheat_victims
        del attack_actions
        del attack_victims
        del attack_seq_elements
        # return copy.copy(attack_seq_elements)

    # Take an attack action and return the probability distribution of
    def get_victim_for_chunk(self, attack_action: str) -> int:
        if attack_action == "deal":
            if (self.use_victim_weight_prob["deal"] == 1) or (
                self.rnd_gen.random() < self.use_victim_weight_prob["deal"]
            ):
                if not self.flip_victim_weight:
                    attack_victim = int(
                        self.rnd_gen.choice(
                            self.repgraph.nodes, p=self.victim_weights_mod
                        )
                    )
                else:
                    attack_victim = int(
                        self.rnd_gen.choice(
                            self.repgraph.nodes,
                            p=self.victim_weights_complement_mod,
                        )
                    )
            else:
                attack_victim = int(self.rnd_gen.choice(self.repgraph.nodes))
        elif attack_action == "cheat":
            if (self.use_victim_weight_prob["cheat"] == 1) or (
                self.rnd_gen.random() < self.use_victim_weight_prob["cheat"]
            ):
                if not self.flip_victim_weight:
                    attack_victim = int(
                        self.rnd_gen.choice(
                            self.repgraph.nodes,
                            p=self.victim_weights_complement_mod,
                        )
                    )
                else:
                    attack_victim = int(
                        self.rnd_gen.choice(
                            self.repgraph.nodes, p=self.victim_weights_mod
                        )
                    )
            else:
                attack_victim = int(self.rnd_gen.choice(self.repgraph.nodes))
        else:
            attack_victim = -2
        return attack_victim

    def set_back_off(self):  # type:ignore
        if self.victim_trust_status.any():
            self.victim_weights_mod[
                np.where(self.victim_trust_status is False)  # type:ignore
            ] = 0
            self.victim_weights_mod /= self.victim_weights_mod.sum()
            self.victim_weights_complement_mod[
                np.where(self.victim_trust_status is False)  # type:ignore
            ] = 0
            self.victim_weights_complement_mod /= (
                self.victim_weights_complement_mod.sum()
            )
        else:
            self.victim_weights_mod = self.victim_weights.copy()
            self.victim_weights_complement_mod = (
                self.victim_weights_complement.copy()
            )

    def num_rejects(self):  # type:ignore
        num = 0
        for choice in self.rejects.keys():
            num += len(self.rejects[choice])  # type:ignore
        return num

    def rejected_by(self):  # type:ignore
        return set(
            rejection[1][1]
            for rejection_list in self.rejects.values()
            for rejection in rejection_list  # type:ignore
        )

    def set_seq(self, seq):  # type:ignore
        self.seq = seq.copy()

    def update_transactions(self, time, event):  # type:ignore
        self.transactions[event[0]].append((time, event))  # type:ignore

    def update_rejects(self, time, event):  # type:ignore
        self.rejects[event[0]].append((time, event))  # type:ignore

    @classmethod
    def choices_key(cls, x):  # type:ignore
        return cls.actions.index(x[0])

    # TODO
    # Generalize this across different distributions (and therefore quantities)
    # of added `skip` actions as it currently assumes 1/3 of actions are `skip`
    def add_skips_to_seq(self, seq):  # type:ignore
        seq = copy.deepcopy(seq)
        seq_iter = iter(seq)
        result_len = int(np.floor(1.5 * len(seq)))
        skip_indices = self.rnd_gen.choice(
            result_len, size=int(np.floor(0.5 * len(seq))), replace=False
        )
        result_seq = []
        skip_idx = 0
        seq_idx = 0
        for idx in range(result_len):
            if idx in skip_indices:
                skip_idx += 1
            else:
                seq_idx += 1
            result_seq.append(
                ("skip", -2) if idx in skip_indices else next(seq_iter)
            )
        return result_seq

    @classmethod
    def order_seq(cls, seq):  # type:ignore
        return sorted(seq, key=cls.choices_key)

    @classmethod
    def construct_from_template(cls, attacker: Attacker) -> Attacker:
        return cls(
            attacker.repgraph,
            attacker.attacker_type,
            rate=attacker.rate,
            seq=attacker.seq,
        )

    @classmethod
    def construct_from_template_with_seq(  # type:ignore
        cls,
        attacker: Attacker,
    ):
        return Attacker(
            repgraph=attacker.repgraph,
            attacker_type=attacker.attacker_type,
            action_probabilities=attacker.action_probabilities,
            action_caps=attacker.action_caps,
            back_off=attacker.back_off,
            rate=attacker.rate,
            use_victim_weight_prob=attacker.use_victim_weight_prob,
            flip_victim_weight=attacker.flip_victim_weight,
            seq=attacker.seq,
            attacker_type_at_creation=attacker.attacker_type_at_creation,
        )

    @classmethod
    def kwargs_dict(cls):  # type:ignore
        init_kwargs = list(inspect.signature(cls.__init__).parameters)
        # This method is useful for generating attacker kwargs.
        # But be careful about seq, because they're mutable.
        return {k: None for k in init_kwargs if not k == "seq"}

    @classmethod
    def kwargs_list(cls, reject_list):  # type:ignore
        init_kwargs = list(inspect.signature(cls.__init__).parameters)
        # This method is useful for generating attacker kwargs.
        # But be careful about seq, because they're mutable.
        return [
            parameter
            for parameter in init_kwargs
            if parameter not in reject_list
        ]

    @staticmethod
    def seq_gen_from_seq():  # type:ignore
        pass

    @staticmethod
    def iter_to_victim_distribution(seq, num_victims):  # type:ignore
        if isinstance(seq[0], tuple):
            seq = [x[1] for x in seq]
        count_per_victim = Counter(seq)  # type:ignore
        victim_distribution = np.asarray(
            [
                count_per_victim[victim] if victim in count_per_victim else 0
                for victim in range(num_victims)
            ],
            dtype="float",
        )
        victim_distribution /= victim_distribution.sum()
        return victim_distribution

    def create_name(self) -> str:
        order_flag_bool = self.attacker_type[1] == "ordered"
        deal_flag_bool = (self.action_probabilities["deal"] > 0) and (
            self.action_caps["deal"] > 0
        )
        skip_flag_bool = (self.attacker_type[1] == "skip") or (
            (self.action_probabilities["skip"] > 0)
            and (self.action_caps["skip"] > 0)
        )
        order_skip_flag_bool = self.attacker_type[1] == "ordered_skip"

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
        attacker_name: str = (
            self.attacker_type_at_creation[0]
            + ord_flag
            + deal_flag
            + skip_flag
            + " "
            + str(self.rate)
        )
        return attacker_name

    def get_transactions_per_user(self) -> None:
        for action in self.actions:
            if action == "skip":
                pass
            else:
                x = [i[1][1] for i in self.transactions[action]]
                y = dict(Counter(x))
                self.transactions_per_user[action] = {
                    k: (y[k] if (k in y) else 0)
                    for k in range(len(self.repgraph.nodes))
                }

    def get_rejects_per_user(self) -> None:
        for action in self.actions:
            if action == "skip":
                pass
            else:
                x = [i[1][1] for i in self.rejects[action]]
                y = dict(Counter(x))
                self.rejects_per_user[action] = {
                    k: (y[k] if (k in y) else 0)
                    for k in range(len(self.repgraph.nodes))
                }

    # TODO
    # def plot_weights(
    #     self,
    #     plot_name: str,
    #     plotter_obj: tp.Optional[plotter.Plotter] = None,
    #     weights: tp.Optional[np.ndarray] = None,
    # ) -> None:
    #     weights_complement: tp.Optional[np.ndarray] = None
    #     weights_complement_mod: tp.Optional[np.ndarray] = None
    #     if plotter_obj is None:
    #         raise ValueError("Oops")
    #         # plotter_obj = plotter.Plotter()
    #     if weights is None:
    #         plot_name += " self_weights"
    #         weights = self.victim_weights
    #         weights_mod = self.victim_weights_mod
    #         weights_complement = self.victim_weights_complement
    #         weights_complement_mod = self.victim_weights_complement_mod
    #     else:
    #         plot_name += " input_weights"
    #     node_idcs = [i for i in range(len(weights))]
    #     plotter_obj.plot_results(
    #         node_idcs,
    #         [list(weights)],
    #         ["Weights"],
    #         "Node",
    #         "Prob. Density",
    #         "Attacker Victim Weights",
    #         plot_name,
    #         "bar",
    #     )
    #     if weights_mod is not None:
    #         plotter_obj.plot_results(
    #             node_idcs,
    #             [list(weights_mod)],
    #             ["Modified Weights"],
    #             "Node",
    #             "Prob. Density",
    #             "Modified Attacker Victim Weights",
    #             plot_name + "_mod",
    #             "bar",
    #         )
    #     if weights_complement is not None:
    #         plotter_obj.plot_results(
    #             node_idcs,
    #             [list(weights_complement)],
    #             ["Complement Weights"],
    #             "Node",
    #             "Prob. Density",
    #             "Complement Attacker Victim Weights",
    #             plot_name + "_complement",
    #             "bar",
    #         )
    #     if weights_complement_mod is not None:
    #         plotter_obj.plot_results(
    #             node_idcs,
    #             [list(weights_complement_mod)],
    #             ["Modified Complement Weights"],
    #             "Node",
    #             "Prob. Density",
    #             "Modified Complement Attacker Victim Weights",
    #             plot_name + "_complement_mod",
    #             "bar",
    #         )

    # TODO
    # def plot_seq(
    #     self,
    #     plot_name: str,
    #     plotter_obj: tp.Optional[plotter.Plotter] = None,
    #     seq: tp.Optional[tp.List[tp.Tuple[str, int]]] = None,
    # ) -> None:
    #     if plotter_obj is None:
    #         plotter_obj = plotter.Plotter()
    #     if seq is None:
    #         plot_name += " self_seq"
    #         seq = self.seq
    #     else:
    #         plot_name += " input_seq"
    #     node_idcs = [i for i in range(len(self.repgraph.nodes))]
    #     seq_victims = [attack[1] for attack in seq]
    #     seq_victims_count = [0] * len(node_idcs)
    #     for n, c in Counter(seq_victims).items():
    #         seq_victims_count[n] = c
    #     plotter_obj.plot_results(
    #         node_idcs,
    #         [list(seq_victims_count)],
    #         ["Num. of times chosen"],
    #         "Node",
    #         "Num. of times chosen",
    #         "Victim Choice Distribution",
    #         plot_name,
    #         "bar",
    #     )

    # TODO
    # def plot_transactions_per_user(
    #     self,
    #     plot_name: str,
    #     plotter_obj: tp.Optional[plotter.Plotter] = None,
    # ) -> None:
    #     if (self.transactions_per_user["deal"] is None) or (
    #         self.transactions_per_user["cheat"] is None
    #     ):
    #         self.get_transactions_per_user()
    #     for action in self.actions:
    #         if action == "skip":
    #             pass
    #         else:
    #             temp_dict = tp.cast(
    #                 tp.Dict[int, int], self.transactions_per_user[action]
    #             )
    #             if plotter_obj is None:
    #                 plotter_obj = plotter.Plotter()
    #             plotter_obj.plot_results(
    #                 list(temp_dict.keys()),
    #                 [list(temp_dict.values())],
    #                 ["Num. of transactions"],
    #                 "Num. of transactions",
    #                 "Number of Transactions Per User",
    #                 plot_name + " " + action + " per user.png",
    #                 "Transactions Per User",
    #                 "bar",
    #             )


class AttackerKwargs:
    def __init__(self, attacker_class):  # type:ignore
        self.attacker_kwargs_list = []
        self.reverse_dict: tp.Dict[str, tp.Dict] = dict()  # type:ignore
        self.attacker_metrics = attacker_class.attacker_metrics
        self.default_attacker_kwargs = attacker_class.kwargs_list(
            ["self", "repgraph", "seq"]
        )

    def add_attacker_kwargs(self, attacker_kwargs_dict: dict):  # type:ignore
        # If this dict of attacker kwargs is not in the list
        if attacker_kwargs_dict not in self.attacker_kwargs_list:
            # Give this dict the next available index
            attacker_kwargs_idx = len(self.attacker_kwargs_list)
            # Append dict to list
            self.attacker_kwargs_list.append(attacker_kwargs_dict)
            for key, value in attacker_kwargs_dict.items():
                if key in self.reverse_dict:
                    if value in self.reverse_dict[key]:
                        self.reverse_dict[key][value].add(attacker_kwargs_idx)
                    else:
                        self.reverse_dict[key][value] = {attacker_kwargs_idx}
                else:
                    self.reverse_dict[key] = {value: {attacker_kwargs_idx}}
            # if str(key) == 'back_off':
            #     if value is True:
            #         self.reverse_dict['back_off_true'].add(attacker_kwargs_idx)
            #     else:
            #         self.reverse_dict['back_off_false'].add(attacker_kwargs_idx)
            # elif value in self.reverse_dict:
            #     self.reverse_dict[value].add(attacker_kwargs_idx)
            # else:
            #     self.reverse_dict[value] = {attacker_kwargs_idx}

    def get_attacker_kwargs(  # type:ignore
        self,
        search_list: tp.List = None,  # type:ignore
        search_dict: tp.Dict = None,  # type:ignore
        omit_list: tp.List = None,  # type:ignore
        indices_only=False
        # omit_dict: tp.Dict = None,
    ):
        k_v_list = []
        if omit_list is not None:
            omit_vals: tp.List = []  # type:ignore
            omit_strs: tp.List = []  # type:ignore
            for o in omit_list:
                (omit_strs if isinstance(o, str) else omit_vals).append(o)
        if search_list is not None:
            for v_search in search_list:
                some_attacker = False
                for key_dict, sub_dict in self.reverse_dict.items():
                    if isinstance(v_search, str):
                        for k_sub_dict in sub_dict:
                            if (
                                isinstance(k_sub_dict, str)
                                and v_search in k_sub_dict
                            ):
                                k_v_list.append((key_dict, v_search))
                                some_attacker = True
                    else:
                        if v_search in sub_dict:
                            k_v_list.append((key_dict, v_search))
                            some_attacker = True
                if not some_attacker:
                    return []
        if search_dict is not None:
            for k_search, v_search in search_dict.items():
                some_attacker = False
                if k_search in self.reverse_dict:
                    sub_dict = self.reverse_dict[k_search]
                    if isinstance(v_search, str):
                        for k_sub_dict in sub_dict:
                            if v_search in k_sub_dict:
                                k_v_list.append((k_search, v_search))
                                some_attacker = True
                    else:
                        if v_search in sub_dict:
                            k_v_list.append((k_search, v_search))
                            some_attacker = True
                if not some_attacker:
                    return []
        idx_set_list: tp.List[set] = [set() for _ in k_v_list]  # type:ignore
        for idx, (k, v) in enumerate(k_v_list):
            if omit_list is not None:
                if isinstance(v, str):
                    for k_sub_dict, v_sub_dict in self.reverse_dict[k].items():
                        if isinstance(k_sub_dict, str) and v in k_sub_dict:
                            omit_v = False
                            for o in omit_strs:
                                if o in k_sub_dict:
                                    omit_v = True
                            if not omit_v:
                                idx_set_list[idx].update(v_sub_dict)
                else:
                    if v not in omit_vals:
                        idx_set_list[idx].update(self.reverse_dict[k][v])
            else:
                if isinstance(v, str):
                    for k_sub_dict, v_sub_dict in self.reverse_dict[k].items():
                        if isinstance(k_sub_dict, str) and v in k_sub_dict:
                            idx_set_list[idx].update(v_sub_dict)
                else:
                    idx_set_list[idx].update(self.reverse_dict[k][v])
        idx_set = set.intersection(*idx_set_list)
        if indices_only:
            return idx_set
        else:
            return [
                self.attacker_kwargs_list[idx] for idx in sorted(list(idx_set))
            ]

    def get_attacker_kwargs_combo(  # type:ignore
        self,
        combo_list_in: tp.List = None,  # type:ignore
        all_list: tp.List = None,  # type:ignore
        combo_dicts_in: tp.List[tp.Dict] = None,  # type:ignore
        all_dict: tp.Dict = None,  # type:ignore
        omit_list: tp.List = None,  # type:ignore
    ):
        atk_kwargs_set = set()
        if all_list is None:
            all_list = []
        if all_dict is None:
            all_dict = {}
        if combo_dicts_in is None:
            if combo_list_in is not None:
                combo_list: tp.List = combo_list_in  # type:ignore
            for x in combo_list:
                atk_kwargs_set.update(
                    self.get_attacker_kwargs(
                        all_list + [x],
                        all_dict,
                        omit_list=omit_list,
                        indices_only=True,
                    )
                )
        elif combo_list_in is None:
            if combo_dicts_in is not None:
                combo_dicts: tp.List[tp.Dict] = combo_dicts_in  # type:ignore
            for temp_dict in combo_dicts:
                temp_dict.update(all_dict.copy())
                atk_kwargs_set.update(
                    self.get_attacker_kwargs(
                        all_list,
                        temp_dict,
                        omit_list=omit_list,
                        indices_only=True,
                    )
                )
        else:
            for x, temp_dict in zip(combo_list_in, combo_dicts_in):
                temp_dict.update(all_dict.copy())
                atk_kwargs_set.update(
                    self.get_attacker_kwargs(
                        all_list + [x],
                        temp_dict,
                        omit_list=omit_list,
                        indices_only=True,
                    )
                )
        return [
            self.attacker_kwargs_list[idx]
            for idx in sorted(list(atk_kwargs_set))
        ]

    def get_all_attacker_kwargs(self):  # type:ignore
        return self.attacker_kwargs_list

    def get_reverse_dict(self):  # type:ignore
        return self.reverse_dict

    def gen_attacker_kwargs_from_partial(  # type:ignore
        self,
        partial_attacker_kwargs_dict,
    ):
        return {
            key: partial_attacker_kwargs_dict[key]
            if key in partial_attacker_kwargs_dict
            else None
            for key in self.default_attacker_kwargs
        }

    #
    # def gen_attacker_kwargs_signature(self, attacker_kwargs_dict):
    #     full_kwargs = self.gen_attacker_kwargs()
