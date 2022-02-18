import typing as tp
import typing_extensions as tp_ext
import networkx as nx


# # # Graphs # # #

# NXGraph object and related types.
# TODO networkx needs proper type hinting (stubs)
NXGraph = nx.Graph


RepGraph = nx.Graph


# class RepGraph(NXGraph):
# pass


# Type enforcing dicts defining args for the creation of some nxgraphs.


class NXGraphArgsWS(tp_ext.TypedDict):
    n: tp.Optional[int]
    k: tp.Optional[int]
    p: tp.Optional[float]


nxgraph_keys_ws_type = tp.List[tp_ext.Literal["n", "k", "p"]]
nxgraph_keys_ws: nxgraph_keys_ws_type = ["n", "k", "p"]


class NXGraphArgsBA(tp_ext.TypedDict):
    n: tp.Optional[int]
    m: tp.Optional[int]


nxgraph_keys_ba_type = tp.List[tp_ext.Literal["n", "m"]]
nxgraph_keys_ba = ["n", "m"]


class NXGraphArgsBT(tp_ext.TypedDict):
    r: tp.Optional[int]
    h: tp.Optional[int]


nxgraph_keys_bt_type = tp.List[tp_ext.Literal["r", "h"]]
nxgraph_keys_bt = ["r", "h"]


NXGraphArgs = tp.Union[
    NXGraphArgsWS,
    NXGraphArgsBA,
    NXGraphArgsBT,
]


nxgraph_keys_all_type = tp_ext.Literal[
    "n",
    "k",
    "p",
    "m",
    "r",
    "h",
]
nxgraph_keys_all_list_type = tp.List[
    tp_ext.Literal[
        "n",
        "k",
        "p",
        "m",
        "r",
        "h",
    ]
]
nxgraph_keys_all_list: nxgraph_keys_all_list_type = [
    "n",
    "k",
    "p",
    "m",
    "r",
    "h",
]


class NXGraphArgsListsWS(tp_ext.TypedDict):
    n: tp.List[int]
    k: tp.List[int]
    p: tp.List[float]


class NXGraphArgsListsBA(tp_ext.TypedDict):
    n: tp.List[int]
    m: tp.List[int]


class NXGraphArgsListsBT(tp_ext.TypedDict):
    r: tp.List[int]
    h: tp.List[int]


NXGraphArgsLists = tp.Union[
    NXGraphArgsListsWS,
    NXGraphArgsListsBA,
    NXGraphArgsListsBT,
]


# Type enforcing dict defining args for subsequent "create_reputation_graph"
# method.
class RepGraphArgs(tp_ext.TypedDict):
    # NXGraph generation function.
    nxgraph_function: tp.Callable[..., nx.Graph]
    # Arguments for nx graph gen function.
    nxgraph_args_dict: NXGraphArgs
    # Base user rate or a function that returns a single number.
    rate: float
    # If "rate" is callable, these are the arguments that it is called with.
    # Set to "None" if the function doesn't take arguments.
    rate_args: tp.Optional[tp.Dict[str, tp.Any]]
    # Specifies whether the generated graph should be directed or not.
    directed: bool
    # Specifies whether the whole graph should be returned or just a view of
    # the graph.
    as_view: bool


# class RepGraphArgsWS(RepGraphArgs):
#     # Arguments for nx graph gen function.
#     nxgraph_args_dict: NXGraphArgsWS
#
#
# class RepGraphArgsBA(RepGraphArgs):
#     # Arguments for nx graph gen function.
#     nxgraph_args_dict: NXGraphArgsBA
#
#
# class RepGraphArgsBT(RepGraphArgs):
#     # Arguments for nx graph gen function.
#     nxgraph_args_dict: NXGraphArgsBT


# # # Attacker # # #

attacker_action = tp.Literal["deal", "skip", "cheat"]
attacker_action_list = tp.List[attacker_action]
attacker_action_set = tp.Set[attacker_action]


class FloatPerAction(tp_ext.TypedDict):
    deal: float
    skip: float
    cheat: float


class IntPerAction(tp_ext.TypedDict, total=False):
    deal: int
    skip: int
    cheat: int


class EventsPerAction(tp_ext.TypedDict, total=False):
    deal: tp.List[tp.Tuple[float, tp.Tuple[str, int]]]
    cheat: tp.List[tp.Tuple[float, tp.Tuple[str, int]]]
    skip: tp.List[tp.Tuple[float, tp.Tuple[str, int]]]


class NodesPerAction(tp_ext.TypedDict):
    deal: tp.Optional[tp.Dict[int, int]]
    cheat: tp.Optional[tp.Dict[int, int]]


# # # Results # # #


class BasicStatsDict(tp_ext.TypedDict, total=False):
    minimum: float
    maximum: float
    mean: float
    variance: float
    median: float
    tenth: float
    ninetieth: float
    pop_size: int


class SimDataDict(tp_ext.TypedDict, total=False):
    stop_condition: tp.List[str]
    runtime_list: tp.List[float]
    runtime_stats: tp.Optional[BasicStatsDict]
    attack_ordered: tp.List[bool]
    clairvoyance: tp.List[bool]
    users_trust_histories: tp.List[tp.List[str]]
    terminating_indices: tp.List[int]
    terminating_indices_stats: tp.Optional[BasicStatsDict]


class RepgraphDataDict(tp_ext.TypedDict, total=False):
    name: tp.List[str]
    average_clustering_per_graph: tp.List[float]
    average_clustering_stats: tp.Optional[BasicStatsDict]
    dc: tp.List[tp.Dict[int, float]]
    spbc: tp.List[tp.Dict[int, float]]
    cc: tp.List[tp.Dict[int, float]]
    ec: tp.List[tp.Dict[int, float]]
    enebc: tp.List[tp.Dict[int, float]]
    enbc: tp.List[tp.Dict[int, float]]
    eebc: tp.List[tp.Dict[int, float]]


class UsersDataDict(tp_ext.TypedDict, total=False):
    user_thresholds: tp.List[int]
    user_thresholds_stats: tp.Optional[BasicStatsDict]
    num_weighted_cheats: tp.List[tp.List[int]]
    num_own_weighted_cheats: tp.List[tp.List[int]]
    thing: tp.Any
    oracle_accepts: tp.List[EventsPerAction]
    oracle_accepts_deals: tp.List[int]
    oracle_accepts_cheats: tp.List[int]
    oracle_accepts_deal_stats: tp.Optional[BasicStatsDict]
    oracle_accepts_cheat_stats: tp.Optional[BasicStatsDict]
    oracle_rejects: tp.List[EventsPerAction]
    oracle_rejects_deals: tp.List[int]
    oracle_rejects_cheats: tp.List[int]
    oracle_rejects_deal_stats: tp.Optional[BasicStatsDict]
    oracle_rejects_cheat_stats: tp.Optional[BasicStatsDict]


# TODO consider delistifying and making a list of these
# with one element per attacker
class AttackerDataDict(tp_ext.TypedDict, total=False):
    name: tp.List[str]
    attacker_type_at_creation: tp.List[
        tp.Tuple[
            str,
            tp.Optional[
                tp.Union[
                    tp.Callable[[tp.Sequence[float]], tp.Sequence[float]],
                    str,
                ]
            ],
        ]
    ]
    attacker_type: tp.List[
        tp.Tuple[
            str,
            tp.Optional[
                tp.Union[
                    tp.Callable[[tp.Sequence[float]], tp.Sequence[float]],
                    str,
                ]
            ],
        ]
    ]
    seq_length: tp.List[int]
    seq_length_stats: BasicStatsDict
    attacker_data_last_accept_idcs: tp.List[int]
    last_accept_idcs_stats: BasicStatsDict
    last_accept_action_count: IntPerAction
    last_accept_time_stats: BasicStatsDict
    rate: tp.List[float]
    back_off: tp.List[bool]
    flip_victim_weight: tp.List[bool]
    attack_seqs: tp.Optional[tp.List[tp.List[tp.Tuple[str, int]]]]
    action_probabilities: tp.List[FloatPerAction]
    action_caps: tp.List[IntPerAction]
    profits: tp.List[int]
    profits_stats: tp.Optional[BasicStatsDict]
    expenditure: tp.List[int]
    expenditure_stats: tp.Optional[BasicStatsDict]
    lag_expenditure: tp.List[int]
    lag_expenditure_stats: tp.Optional[BasicStatsDict]
    lag_profits: tp.List[int]
    lag_profits_stats: tp.Optional[BasicStatsDict]
    attempts: tp.List[int]
    attempts_stats: tp.Optional[BasicStatsDict]
    accepts: tp.List[int]
    accepts_stats: tp.Optional[BasicStatsDict]
    lag_accepts: tp.List[int]
    lag_accepts_stats: tp.Optional[BasicStatsDict]
    rejects: tp.List[int]
    rejects_stats: tp.Optional[BasicStatsDict]
    rejects_deals: tp.List[int]
    rejects_cheats: tp.List[int]
    rejects_deals_stats: tp.Optional[BasicStatsDict]
    rejects_cheats_stats: tp.Optional[BasicStatsDict]
    rejects_per_user: tp.List[NodesPerAction]
    lag_rejects: tp.List[int]
    lag_rejects_stats: tp.Optional[BasicStatsDict]
    ratio_rejects_attempts: tp.List[float]
    ratio_rejects_attempts_stats: tp.Optional[BasicStatsDict]
    attempts_deals: tp.List[int]
    attempts_cheats: tp.List[int]
    attempts_deals_stats: BasicStatsDict
    attempts_cheats_stats: BasicStatsDict
    # TODO check actual types of the statements below
    transactions_per_user: tp.List[NodesPerAction]


class ResultsDataDict(tp_ext.TypedDict):
    sim_data: SimDataDict
    repgraph_data: RepgraphDataDict
    users_data: UsersDataDict
    attacker_data: AttackerDataDict


# # # Plotting # # ##

xval_num_type = tp.TypeVar("xval_num_type", int, float, str)
xvals_w_varname = tp.Tuple[tp.List[xval_num_type], str]
xvals_w_varname_any = tp.Union[
    xvals_w_varname[int], xvals_w_varname[float], xvals_w_varname[str]
]
ytriple = tp.Tuple[tp.List[float], tp.List[tp.Tuple[float, float]], str]
prep_dict = tp.Dict[
    str,
    tp.Union[
        str,
        ytriple,
        xvals_w_varname[int],
        xvals_w_varname[float],
        xvals_w_varname[str],
    ],
]
xvals = tp.Union[tp.Sequence[int], tp.Sequence[float], tp.Sequence[str]]
yvals = tp.Sequence[tp.Sequence[float]]
yerrs_pairs = tp.Sequence[tp.Sequence[tp.Tuple[float, float]]]
yerrs_split = tp.Sequence[tp.Tuple[tp.Sequence[float], tp.Sequence[float]]]
