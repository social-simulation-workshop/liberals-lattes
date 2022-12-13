import argparse

class ArgsConfig(object):
    
    def __init__(self) -> None:
        super().__init__()
    
        parser = argparse.ArgumentParser()

        parser.add_argument("--n_player", type=int, default=500,
            help="")
        parser.add_argument("--n_neighbor", type=int, default=99,
            help="")
        parser.add_argument("--n_iteration", type=int, default=100000000,
            help="")
        parser.add_argument("--measure_itv", type=int, default=10000,
            help="")
        parser.add_argument("--n_ttest_sample", type=int, default=100,
            help="")
        parser.add_argument("--n_replication", type=int, default=100,
            help="")
        parser.add_argument("--use_prebuild_net", nargs="?", const=True, default=False,
            help="")
        parser.add_argument("--random_seed", type=int, default=8783,
            help="")
        
        self.parser = parser
    

    def get_args(self):
        args = self.parser.parse_args()
        return args
