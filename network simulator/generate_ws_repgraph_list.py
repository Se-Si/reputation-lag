import graphs

# 2021-10-04 02-33-51
# 2021-10-04 07-00-26
repgraph_args_1 = graphs.Graphs.generate_repgraph_args_ws(
    graphs.Graphs.generate_nxgraph_args_ws(400, 20, 0.001),
    rate=1,
)

# 2021-10-04 02-24-14
# 2021-10-04 06-52-25
repgraph_args_2 = graphs.Graphs.generate_repgraph_args_ws(
    graphs.Graphs.generate_nxgraph_args_ws(400, 20, 0.01),
    rate=1,
)

# 2021-10-04 03-02-10
# 2021-10-04 06-36-21
repgraph_args_3 = graphs.Graphs.generate_repgraph_args_ws(
    graphs.Graphs.generate_nxgraph_args_ws(400, 20, 0.1),
    rate=1,
)

graphs.Graphs.create_graph_list(repgraph_args_3, 500)

# ? 06-57-19
