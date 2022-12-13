import multiprocessing
import numpy as np
import os
import sys

from utils import Network, print_std_out_err
from plot import PlotLinesHandler
from args import ArgsConfig


def run_simulation(log_list, args, output_dir):
    fn_suffix = "N_{}_K_{}_iter_{}_rndSeed_{}".format(args.n_player, args.n_neighbor, args.n_iteration, args.random_seed)
    if args.use_prebuild_net:
        fn_suffix = "prebuild_" + fn_suffix
    f = open(os.path.join(output_dir, "res_"+fn_suffix+".txt"), 'w')
    stdout = sys.stdout
    sys.stdout = f

    net = Network(
        n_player=args.n_player,
        n_neighbor=args.n_neighbor,
        n_iteration=args.n_iteration,
        use_prebuild_net=args.use_prebuild_net,
        measure_itv=args.measure_itv,
        n_ttest_sample=args.n_ttest_sample,
        rnd_seed=args.random_seed
    )
    net.simulate()
    print_std_out_err("mean pairwise correlation: {}".format(net.measure_pairwise_corr()))

    f.close()
    sys.stdout = stdout

    log_list.append(net.measure_pairwise_corr())


if __name__ == "__main__":
    args_config = ArgsConfig()
    args = args_config.get_args()

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "res_txt")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    PARAM_N_START = 500
    PARAM_N_END = 5000
    PARAM_N_INTERVAL = 500
    PARAM_N = list(range(PARAM_N_START, PARAM_N_END+1, PARAM_N_INTERVAL))
    log_data = list()
    args_list = list()
    for exp_idx, N in enumerate(PARAM_N):
        manager = multiprocessing.Manager()
        log_list = manager.list()
        log_data.append(log_list)

        for rep_idx in range(args.n_replication):
            args_tmp = args_config.get_args()
            args_tmp.n_player = N
            args_tmp.random_seed += rep_idx 

            args_list.append([log_list, args_tmp, output_dir])
    
    n_cpus = multiprocessing.cpu_count()
    print("cpu count: {}".format(n_cpus))
    pool = multiprocessing.Pool(n_cpus+2)
    pool.starmap(run_simulation, args_list)

    # print result = while 
    res_mean = np.array([np.mean(log_list) for log_list in log_data])
    res_std = np.array([np.std(log_list) for log_list in log_data])
    
    fn_suffix = "N_{}_{}_K_{}_iter_{}_repli_{}_rndSeed_{}".format(
        PARAM_N_START, PARAM_N_END, args.n_neighbor, args.n_iteration, args.n_replication, args.random_seed)
    if args.use_prebuild_net:
        fn_suffix = "prebuild_" + fn_suffix
    plot_handler = PlotLinesHandler(xlabel="Network Size",
                                    ylabel="Mean Pairwise Correlation Magnitude",
                                    title=None,
                                    fn="fig3",
                                    x_lim=[PARAM_N_START, PARAM_N_END], x_tick=[PARAM_N_START, PARAM_N_END, PARAM_N_INTERVAL], 
                                    y_lim=[0.0, 1.0], y_tick=[0.0, 1.0, 0.1],
                                    use_ylim=True,
                                    figure_ratio=853/1090,
                                    figure_size=7.5)
    plot_handler.plot_line_errorbar(res_mean, res_std, data_log_v=PARAM_N_INTERVAL, linewidth=2, color="black")
    plot_handler.save_fig(fn_suffix=fn_suffix)


