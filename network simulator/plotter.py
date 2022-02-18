from __future__ import annotations

import os
import tools
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import typing as tp
import customtypes as ctp


class Plotter:
    def __init__(self, name: str = "", output_date_and_time: str = "") -> None:
        # Set object's date and time
        self.date_and_time = tools.create_name_datetime(
            name, output_date_and_time
        )
        # Use date and time to make directory name
        self.dir_name = self.date_and_time + " plots/"
        # Derive path relative to basic-sim folder
        self.dir_path = "./plots/" + self.dir_name
        # Try to make the directory
        try:
            os.mkdir(self.dir_path)
        # If the directory exists then do nothing
        except FileExistsError:
            pass

    # Bar plot of attacker profits for a single (kind of) graph.
    def plot_results(
        self,
        # The xvals being plotted on the x-axis.
        # The xvals are usually attacker xvals.
        xvals: ctp.xvals,
        # Each element of this list is
        # a sequence of values per variable.
        yvals_lst: tp.Sequence[tp.Sequence[float]],
        # TODO
        # ? Is this right
        # Each element of this list is a tuple of
        # the +/- errors for each sequence in `yvals_lst`.
        yvals_errs_split_lst: tp.Sequence[
            tp.Tuple[tp.Sequence[float], tp.Sequence[float]]
        ],
        # Each element of this list is a label
        # for each sequence in `yvals_lst`.
        yvals_labels: tp.Sequence[str],
        xlabel: str,
        ylabel: str,
        # Specifies whether the plot is a "std" or "bar" plot.
        plot_type: str,
        plot_title: str,
        filename_base: str,
        xscale: str = "linear",
        yscale: str = "linear",
        graph_name: str = "",
        # Specifies whether to actually draw the figure or not.
        finish_plot: bool = True,
        # Accepts the undrawn figure of a previous plot.
        fig_ax_in: tp.Optional[tp.Tuple[tp.Any, tp.Any]] = None,
        # font_size: float = 6,
        dpi_value: float = 600,
        legend_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
        ybound: tp.Optional[tp.Dict[str, float]] = None,
        axhlines: tp.Optional[tp.Sequence[tp.Tuple[float, str, str]]] = None,
        axvlines: tp.Optional[tp.Sequence[tp.Tuple[float, str, str]]] = None,
    ) -> tp.Optional[tp.Tuple[tp.Any, tp.Any]]:
        if legend_kwargs is None:
            legend_kwargs = {}
        # error_colours = ("g", "c", "m")
        error_colours = "k"
        # plt.rcParams.update({"font.size": font_size})

        if fig_ax_in is not None:
            fig, ax = fig_ax_in
        else:
            fig, ax = plt.subplots()
        if yvals_errs_split_lst is not None:
            if not (len(yvals_lst) == len(yvals_errs_split_lst)):
                raise ValueError(
                    "Please input one set of errors for each set of values."
                )
            # Loop through the
            for idx in range(len(yvals_lst)):
                if plot_type == "bar":
                    # The elements of this array define the central x-position
                    # of the points/bars for each variable.
                    # It is initialised as an evenly spaced sequence of values
                    # whose length is the number of xvals.
                    # Each variable has `K` points/bars where
                    # `K = len(yvals_lst)`.
                    # `x_pos_array = [0, 1, 2, ... len(yvals_lst)]`
                    x_pos_array = np.arange(len(xvals))

                    # The total width of all the bars for a single variable.
                    # I use `0.8` to leave a bit more spacing between the
                    # clusters of bars.
                    width_per_var = 0.8

                    # The number of sequences of values is the number of bars
                    # per
                    num_bars_per_var = len(yvals_lst)

                    # The width of a single bar.
                    # This is the total width per variable divided by the number
                    # of bars per variable.
                    bar_width = width_per_var / num_bars_per_var

                    offset_array = self.get_offset_array(
                        num_bars_per_var, width_per_var
                    )

                    # Define list where each element
                    # is a list of bar heights per variable.
                    # This equal to the list of values.
                    bars_per_var_lst = yvals_lst
                    if yscale == "log":
                        bar_log_flag = True
                    else:
                        bar_log_flag = False
                    # print(x_pos_array - offset_array[idx])
                    ax.bar(
                        # x-values of
                        x_pos_array - offset_array[idx],
                        bars_per_var_lst[idx],
                        bar_width,
                        label=yvals_labels[idx],
                        log=bar_log_flag,
                    )
                    ax.errorbar(
                        x_pos_array - offset_array[idx],
                        bars_per_var_lst[idx],
                        yerr=yvals_errs_split_lst[idx],
                        fmt="x",
                        color=error_colours[idx % len(error_colours)],
                        elinewidth=3.0,
                    )
                    ax.set_xticks(x_pos_array)
                    ax.set_xticklabels(xvals)
                elif plot_type == "std":
                    ax.plot(
                        xvals,
                        yvals_lst[idx],
                        label=yvals_labels[idx],
                    )
                    ax.errorbar(
                        xvals,
                        yvals_lst[idx],
                        yerr=yvals_errs_split_lst[idx],
                        fmt="x",
                        color=error_colours[idx % len(error_colours)],
                        elinewidth=1,
                    )
                    ax.set_xscale(xscale)
                    ax.set_yscale(yscale)
                else:
                    raise ValueError("Please input type of plot")
        else:
            for values in yvals_lst:
                ax.bar(xvals, values)
        if ybound is not None:
            ax.set_ybound(**ybound)
        if axhlines is not None:
            for axhline, clr, lbl in axhlines:
                ax.axhline(axhline, color=clr, label=lbl)
        if axvlines is not None:
            for axvline, clr, lbl in axvlines:
                ax.axvline(axvline, color=clr, label=lbl)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(plot_title)
        ax.legend(**legend_kwargs)
        if graph_name == "":
            file_path = self.dir_path + filename_base + " " + plot_type + ".png"
        else:
            graph_name = graph_name.replace(":", "_")
            file_path = (
                self.dir_path + graph_name + " " + filename_base + " bar.png"
            )
        file_path = file_path.replace(":", "-")
        if finish_plot:
            # ax.legend(**legend_kwargs)
            plt.savefig(file_path, dpi=dpi_value, bbox_inches="tight")
            plt.close()
            return None
        else:
            return fig, ax

    @staticmethod
    def get_offset_array(
        num_bars_per_var: int, width_per_var: float
    ) -> tp.Sequence[float]:
        """The `offset_array` defines the deviation of each `vs`
        for `vs in yvals_lst`
        i.e. if `vs = yvals_lst[i]` then the offset for all `v in vs`
        is `offset_array[i]`."""

        n = num_bars_per_var

        # Initialise the offset array as `n` evenly-spaced integers.
        # `offset_array = [0, 1, ..., n-1]`
        offset_array: npt.NDArray[np.float_] = np.arange(n, dtype=float)

        # Shift the offset array to centre around 0.
        # So, the shifting value must satisfy the following:
        # `-(offset_array_min + shift) = offset_array_max + shift`
        # `=> max + min = -2 * shift`
        # `shift = -max / 2`
        # `shift = -(n-1) / 2 = (1 - n) / 2`
        offset_array += (1 - n) / 2

        # Scale the offset array so that consecutive bars fit into the desired
        # width.
        # First, scale the array down to fit into a width of 1.
        # Then, scale the array to the appropriate desired width.
        offset_array /= n
        offset_array /= width_per_var
        return tp.cast(tp.List[float], offset_array.tolist())

    def plot_histogram(
        self,
        filename_base: str,
        values: tp.Sequence[float],
        bins: tp.Optional[int] = None,
        error_values: tp.Optional[tp.Sequence[float]] = None,
        graph_name: str = "",
        font_size: float = 6,
        dpi_value: float = 600,
        legend_loc: str = "best",
    ) -> None:
        # fig = plt.figure()
        plt.figure()
        plt.rcParams.update({"font.size": font_size})
        if bins is None:
            plt.hist(values)
        else:
            plt.hist(values, bins=bins)
        # if error_values is not None:
        #     plt.errorbar(
        #         xvals, values, yerr=error_values, fmt="o", color="r"
        #     )
        # if graph_name == "":
        file_path = self.dir_path + filename_base + " hist.png"
        # else:
        #     file_path = (
        #         self.dir_path + graph_name + " " + filename_base + " bar.png"
        #     )
        plt.legend(loc=legend_loc)
        plt.savefig(file_path, dpi=dpi_value)
        plt.close()

    @staticmethod
    def draw_graph(
        G: ctp.NXGraph,
        draw_type: str,
        name: str,
        node_labels_dict: tp.Optional[tp.Dict[int, tp.Any]] = None,
        edge_labels_dict: tp.Optional[
            tp.Dict[tp.Tuple[int, int], tp.Any]
        ] = None,
        font_size: float = 6,
        dpi_value: float = 600,
    ) -> None:
        plt.figure()
        plt.rcParams.update({"font.size": font_size})
        node_idx_pos = nx.circular_layout(G)
        nx.draw_networkx(
            G,
            node_idx_pos,
            with_labels=True,
            node_size=100,
            font_size=font_size,
            width=0.25,
        )
        if node_labels_dict is not None:
            print(node_idx_pos)
            node_label_pos = {
                k: (x, y + 0.1) for k, (x, y) in node_idx_pos.items()
            }
            node_labels_dict = {
                k: "{:.3f}".format(v) for k, v in node_labels_dict.items()
            }
            nx.draw_networkx_labels(
                G,
                pos=node_label_pos,
                labels=node_labels_dict,
                font_size=font_size,
                font_color="r",
            )
        if edge_labels_dict is not None:
            # TODO implement
            pass
            # nx.draw_networkx_edge_labels(
            #     G,
            #     pos=edge_label_pos,
            #     labels=edge_labels_dict,
            #     font_size=font_size,
            # )
        name_with_datetime = tools.create_name_datetime(name + " graph")
        filename = "./plots/" + name_with_datetime
        filename = filename.replace(":", "-")
        plt.axis("off")
        axis = plt.gca()
        axis.set_xlim([1.2 * x for x in axis.get_xlim()])
        axis.set_ylim([1.2 * y for y in axis.get_ylim()])
        plt.tight_layout()
        plt.savefig(filename, dpi=dpi_value)
        plt.close()

    def pickle_save(
        self,
        object: tp.Any,
        name: str,
    ) -> None:
        pickle_path = self.dir_path + name + ".pickle"
        with open(pickle_path, "wb") as handle:
            pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)
