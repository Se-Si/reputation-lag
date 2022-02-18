import graphs

# 2021-09-27 00-04-22
repgraph_args = graphs.Graphs.generate_repgraph_args_ba(
    graphs.Graphs.generate_nxgraph_args_ba(400, 10),
    rate=1,
)

graphs.Graphs.create_graph_list(repgraph_args, 500)
