import networkx as nx
import typing as tp
import customtypes as ctp
import attacker
import plotter


# reload = importlib.# reload
# reload(tools)


# Adds weights to edge objects in graph.
class Graphs:
    @staticmethod
    def add_graph_weights(
        rate: float,
        graph: ctp.NXGraph,
        rate_args: tp.Any = None,
        directed: bool = True,
        as_view: bool = False,
    ) -> ctp.NXGraph:
        if as_view:
            raise ValueError(
                "Functionality not implemented for `as_view = True`"
            )
        # If "rate" is a function.
        if callable(rate):
            # If the "rate" function is to be called without variables.
            # Set "rate_args" to an empty list.
            if rate_args is None:
                rate_args = []
            # Set "'rate'" variable and "'period'" variable"
            # # in edge dict objects.
            # noinspection PyTypeChecker
            for u, v in graph.edges:
                graph[u][v]["rate"] = rate(*rate_args)
                graph[u][v]["period"] = 1 / graph[u][v]["rate"]
        else:
            # Set "'rate'" variable and "'period'" variable
            # # in edge dict objects.
            # noinspection PyTypeChecker
            for u, v in graph.edges:
                # print((u, v))
                graph[u][v]["rate"] = rate
                if rate == 0:
                    # graph[u][v]["period"] = float("inf")
                    raise ValueError("Rate must be higher than 0")
                else:
                    graph[u][v]["period"] = 1 / graph[u][v]["rate"]
        if False:
            graph = graph.to_directed()
        else:
            graph = graph.to_undirected()
        return graph

    @staticmethod
    # Adds edge betweenness to edge dict objects in graph.
    def add_graph_edge_betweenness(graph: ctp.NXGraph) -> ctp.NXGraph:
        edge_betweenness = nx.edge_betweenness_centrality(
            graph, weight="period"
        )
        for (i, j), v in edge_betweenness.items():
            graph[i][j]["edge_betweenness"] = v
        return graph

    @staticmethod
    def add_graph_centralities(graph: ctp.NXGraph) -> ctp.NXGraph:
        graph.dc = nx.degree_centrality(graph)
        graph.spbc = nx.betweenness_centrality(
            graph, weight="period", endpoints=True
        )
        graph.cc = nx.closeness_centrality(graph, distance="period")
        graph.ec = nx.eigenvector_centrality(
            graph, weight="period", max_iter=100000
        )
        return graph

    @staticmethod
    # Creates a name for a graph from the graph generator name
    # # and its arguments.
    def create_graph_name(
        nxgraph_function: tp.Callable[..., ctp.NXGraph],
        nxgraph_args: ctp.NXGraphArgs,
    ) -> str:
        return nxgraph_function.__name__ + " " + str(nxgraph_args)

    @classmethod
    def create_nxgraph(
        cls,
        nxgraph_function: tp.Callable[..., nx.Graph],
        nxgraph_args_dict: ctp.NXGraphArgs,
    ) -> ctp.NXGraph:
        nxgraph: ctp.NXGraph = nxgraph_function(**nxgraph_args_dict)
        nxgraph.name = cls.create_graph_name(
            nxgraph_function, nxgraph_args_dict
        )
        return nxgraph

    @classmethod
    # Creates a reputation graph.
    def create_repgraph(cls, repgraph_args: ctp.RepGraphArgs) -> ctp.RepGraph:
        nxgraph_function: tp.Callable[..., ctp.NXGraph] = repgraph_args[
            "nxgraph_function"
        ]
        nxgraph_args_dict: ctp.NXGraphArgs = repgraph_args["nxgraph_args_dict"]
        rate = repgraph_args["rate"]
        rate_args = repgraph_args["rate_args"]
        directed = repgraph_args["directed"]
        as_view = repgraph_args["as_view"]
        # Call nxgraph generator arguments.
        nxgraph = cls.create_nxgraph(nxgraph_function, nxgraph_args_dict)
        # Add weights to graph.
        if rate is not None:
            nxgraph = cls.add_graph_weights(
                rate, nxgraph, rate_args, directed, as_view
            )
        else:
            raise ValueError("No rate specified")
        # nxgraph = cls.add_graph_edge_betweenness(nxgraph)
        nxgraph = cls.add_graph_centralities(nxgraph)
        repgraph: ctp.RepGraph
        repgraph = nxgraph
        return repgraph

    @staticmethod
    def generate_nxgraph_args_ws(
        n: tp.Optional[int],
        k: tp.Optional[int],
        p: tp.Optional[float],
    ) -> ctp.NXGraphArgsWS:
        return {
            "n": n,
            "k": k,
            "p": p,
        }

    @staticmethod
    def generate_nxgraph_args_ba(
        n: tp.Optional[int],
        m: tp.Optional[int],
    ) -> ctp.NXGraphArgsBA:
        return {
            "n": n,
            "m": m,
        }

    @staticmethod
    def generate_nxgraph_args_bt(
        r: tp.Optional[int],
        h: tp.Optional[int],
    ) -> ctp.NXGraphArgsBT:
        return {
            "r": r,
            "h": h,
        }

    @classmethod
    # Generate a list of nxgraph_args_dict for
    # # a Barabasi-Albert Reputation graph.
    def generate_nxgraph_args_multi_ws(
        cls,
        n_list: tp.List[int],
        k_list: tp.List[int],
        p_list: tp.List[float],
    ) -> tp.List[ctp.NXGraphArgsWS]:
        return [
            cls.generate_nxgraph_args_ws(n_arg, k_arg, p_arg)
            for n_arg in n_list
            for k_arg in k_list
            for p_arg in p_list
        ]

    @classmethod
    def generate_nxgraph_args_multi_ba(
        cls,
        n_list: tp.List[int],
        m_list: tp.List[int],
    ) -> tp.List[ctp.NXGraphArgsBA]:
        return [
            cls.generate_nxgraph_args_ba(n_arg, m_arg)
            for n_arg in n_list
            for m_arg in m_list
        ]

    @classmethod
    def generate_nxgraph_args_multi_bt(
        cls,
        r_list: tp.List[int],
        h_list: tp.List[int],
    ) -> tp.List[ctp.NXGraphArgsBT]:
        return [
            cls.generate_nxgraph_args_bt(r_arg, h_arg)
            for r_arg in r_list
            for h_arg in h_list
        ]

    @staticmethod
    # Generate nxgraph_args_dict for Connected Watts-Strogatz Reputation graph.
    def generate_repgraph_args_ws(
        # Dict containg args for underlying NXGraph.
        nxgraph_args_dict: ctp.NXGraphArgsWS,
        # Base user rate or a function that returns a single number.
        rate: float = -1,
        # If "rate" is callable, these are the arguments that it is called with.
        # Set to "None" if "rate" doesn't take arguments.
        rate_args: tp.Any = None,
        # Specifies whether the generated graph should be directed or not.
        directed: bool = True,
        # Specifies whether the whole graph should be returned
        # # or just a view of the graph.
        as_view: bool = False,
    ) -> ctp.RepGraphArgs:
        if rate < 0:
            raise ValueError("Please input rate >= 0")
        return {
            "nxgraph_function": nx.connected_watts_strogatz_graph,
            "nxgraph_args_dict": nxgraph_args_dict.copy(),
            "rate": rate,
            "rate_args": rate_args,
            "directed": directed,
            "as_view": as_view,
        }

    @staticmethod
    # Generate a list of nxgraph_args_dict for
    # # a Barabasi-Albert Reputation graph.
    def generate_repgraph_args_ba(
        # Dict containg args for underlying NXGraph.
        nxgraph_args_dict: ctp.NXGraphArgsBA,
        # Base user rate or a function that returns a single number.
        rate: float = -1,
        # If "rate" is callable, these are the arguments that it is called with.
        # Set to "None" if "rate" doesn't take arguments.
        rate_args: tp.Any = None,
        # Specifies whether the generated graph should be directed or not.
        directed: bool = True,
        # Specifies whether the whole graph should be returned
        # # or just a view of the graph.
        as_view: bool = False,
    ) -> ctp.RepGraphArgs:
        if rate < 0:
            raise ValueError("Please input rate >= 0")
        return {
            "nxgraph_function": nx.barabasi_albert_graph,
            "nxgraph_args_dict": nxgraph_args_dict.copy(),
            "rate": rate,
            "rate_args": rate_args,
            "directed": directed,
            "as_view": as_view,
        }

    @staticmethod
    # Generate a list of nxgraph_args_dict for
    # # a Barabasi-Albert Reputation graph.
    def generate_repgraph_args_bt(
        # Dict containg args for underlying NXGraph.
        nxgraph_args_dict: ctp.NXGraphArgsBT,
        # Base user rate or a function that returns a single number.
        rate: float = -1,
        # If "rate" is callable, these are the arguments that it is called with.
        # Set to "None" if "rate" doesn't take arguments.
        rate_args: tp.Any = None,
        # Specifies whether the generated graph should be directed or not.
        directed: bool = True,
        # Specifies whether the whole graph should be returned
        # # or just a view of the graph.
        as_view: bool = False,
    ) -> ctp.RepGraphArgs:
        if rate < 0:
            raise ValueError("Please input rate >= 0")
        return {
            "nxgraph_function": nx.balanced_tree,
            "nxgraph_args_dict": nxgraph_args_dict.copy(),
            "rate": rate,
            "rate_args": rate_args,
            "directed": directed,
            "as_view": as_view,
        }

    @classmethod
    # Return a list containing the specified property from a list of graphs.
    def graph_properties(
        cls,
        graph: ctp.NXGraph,
        attacker_metrics_filter: tp.List[str],
        plot_bar: bool = True,
        draw_graph: bool = False,
        plotter_object: tp.Optional[plotter.Plotter] = None,
    ) -> tp.List[tp.Dict[int, float]]:
        if plotter_object is None:
            plotter_object = plotter.Plotter()
        graph_metrics: tp.List[tp.Dict[int, float]] = []
        filtered_attacker_metrics = [
            attacker_metric
            for attacker_metric in attacker.Attacker.attacker_metrics
            if attacker_metric not in attacker_metrics_filter
        ]
        for attacker_metric in filtered_attacker_metrics:
            graph_metrics.append(
                attacker.get_victim_weights(  # type: ignore
                    (attacker_metric, None), graph, only_values=False
                )
            )
        for result, name in zip(graph_metrics, filtered_attacker_metrics):
            nodes = []
            weights = []
            for k, v in result.items():
                nodes.append(k)
                weights.append(v)
            if plot_bar:
                raise ValueError("Graph plot bar not ready")
                # plotter_object.plot_bar(
                #     nodes, weights, name, graph_name=graph.name
                # )
            if draw_graph:
                plotter_object.draw_graph(graph, "circo", name, result)
        return graph_metrics
        # if callable(property_functions):
        #     return [property_functions(graph) for graph in graphs]
        # else:
        #     return [
        #         [property_function(graph) for graph in graphs]
        #         for property_function in property_functions
        #     ]

    @classmethod
    def create_graph_list(
        cls,
        repgraph_args: ctp.RepGraphArgs,
        num_graphs: int,
    ) -> None:
        rg_list = []
        for idx in range(num_graphs):
            if not (idx % 10):
                print("On repgraph {}".format(idx))
            rg_list.append(cls.create_repgraph(repgraph_args))
        plotter_object = plotter.Plotter()
        plotter_object.pickle_save(rg_list, "repgraph_list")
        plotter_object.pickle_save(
            (repgraph_args, num_graphs), "repgraph_args_and_num_graphs"
        )
        with open(
            plotter_object.dir_path + "repgraph_args_and_num_graphs.txt", "a"
        ) as f:
            f.write("repgraph_args")
            f.write(str(repgraph_args))
            f.write("")
            f.write("num_graphs")
            f.write(str(num_graphs))
