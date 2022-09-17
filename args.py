import argparse

class ArgsConfig(object):
    
    def __init__(self) -> None:
        super().__init__()
    
        parser = argparse.ArgumentParser()

        parser.add_argument("--n_player", type=int, default=500,
            help="")
        parser.add_argument("--n_neighbor", type=int, default=99,
            help="")
        parser.add_argument("--n_iteration", type=int, default=4000000,
            help="")
        parser.add_argument("--measure_itv", type=int, default=2000,
            help="")
        parser.add_argument("--random_seed", type=int, default=1025,
            help="")
        
        self.parser = parser
    

    def get_args(self):
        args = self.parser.parse_args()
        return args
