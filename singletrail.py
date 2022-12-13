import multiprocessing
import numpy as np
import os
import sys

from utils import Network, print_std_out_err
from plot import PlotLinesHandler
from args import ArgsConfig


if __name__ == "__main__":
    args_config = ArgsConfig()
    args = args_config.get_args()

    fn_suffix = "N_{}_K_{}_iter_{}_rndSeed_{}".format(args.n_player, args.n_neighbor, args.n_iteration, args.random_seed)
    f = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "res_"+fn_suffix+".txt"), 'w')
    stdout = sys.stdout
    sys.stdout = f

    print_std_out_err(args)
    net = Network(
        n_player=args.n_player,
        n_neighbor=args.n_neighbor,
        n_iteration=args.n_iteration,
        use_prebuild_net=True,
        measure_itv=args.measure_itv,
        n_ttest_sample=args.n_ttest_sample,
        rnd_seed=args.random_seed
    )
    terminate_iter = net.simulate()
    print_std_out_err("mean pairwise correlation: {}".format(net.measure_pairwise_corr()))
    
    f.close()
    sys.stdout = stdout

    plot_handler = PlotLinesHandler(xlabel="Iterations",
                                    ylabel="Structural Dissonance",
                                    title=None,
                                    fn="figA2",
                                    x_lim=[-50000, terminate_iter+50000],
                                    x_tick=[0, terminate_iter, 1000000], 
                                    x_as_million=True,
                                    y_lim=None, y_tick=None,
                                    use_ylim=False,
                                    figure_ratio=635/902)
    plot_handler.plot_line(np.array(net.dissonance_lst), data_log_v=args.measure_itv, linewidth=2, color="black")
    plot_handler.save_fig(fn_suffix=fn_suffix)
    
    
    

                    

