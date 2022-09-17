import multiprocessing
import numpy as np
import os
import sys

from utils import Edge, Agent, Network
from plot import PlotLinesHandler
from args import ArgsConfig


if __name__ == "__main__":
    args_config = ArgsConfig()
    args = args_config.get_args()

    net = Network(
        n_player=args.n_player,
        n_neighbor=args.n_neighbor,
        n_iteration=args.n_iteration,
        measure_itv=args.measure_itv,
        rnd_seed=args.random_seed
    )
    net.simulate()
    print("mean pairwise correlation: {}".format(net.measure_pairwise_corr()))

    fn_suffix = "N_{}_K_{}_iter_{}_rndSeed_{}".format(args.n_player, args.n_neighbor, args.n_iteration, args.random_seed)
    plot_handler = PlotLinesHandler(xlabel="Iterations",
                                    ylabel="Structural Dissonance",
                                    title=None,
                                    fn="figA2",
                                    x_lim=[-50000, args.n_iteration+50000],
                                    x_tick=[0, args.n_iteration, 1000000], 
                                    x_as_million=True,
                                    y_lim=None, y_tick=None,
                                    use_ylim=False,
                                    figure_ratio=635/902)
    plot_handler.plot_line(np.array(net.dissonance_lst), data_log_v=args.measure_itv, linewidth=2, color="black")
    plot_handler.save_fig(fn_suffix=fn_suffix)
    
    
    

                    

