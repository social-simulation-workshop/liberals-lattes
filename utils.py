import itertools
import numpy as np
import scipy
import sys
import time


def draw(p) -> bool:
    return True if np.random.uniform() < p else False

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def print_std_out_err(*args, **kwargs):
    print(*args, **kwargs)
    eprint(*args, **kwargs)

class Edge:
    def __init__(self, u: int, v: int, w=None):
        if u > v:
            u, v = v, u
        self.u = u
        self.v = v
        self.w = w
    
    def update_weight(self, w_in: float):
        self.w = w_in

    def get_other_end(self, ag_id: int) -> int:
        return self.u if ag_id == self.v else self.v


class Agent:
    _ids = itertools.count(0)

    def __init__(self) -> None:
        self.id = next(self._ids)
        self.gene = Agent.generate_random_gene()
        self.edges = list()
    

    @staticmethod
    def generate_random_gene() -> str:
        """ Return 25 randomly generated genes with first 20 genes be the dynamic."""
        return "".join([f"{b:0>8b}" for b in np.random.bytes(4)])[:25]
    
    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)
    
    @staticmethod
    def flip_gene(bit) -> str:
        return "1" if bit == "0" else "0"
    
    def alter_gene(self, g_idx, g_new) -> None:
        ag_g = list(self.gene)
        ag_g[g_idx] = g_new
        self.gene = "".join(ag_g)
    
    def to_ndarray(self) -> np.ndarray:
        return np.array([int(g) for g in self.gene])


class Network:
    N_DYNAMIC_GENE = 20
    N_STATIC_GENE = 5
    N_GENE = N_DYNAMIC_GENE + N_STATIC_GENE

    def __init__(self, n_player=500, n_neighbor=99, n_iteration=1000000,
        measure_itv=10000, n_ttest_sample=100, verbose=True, rnd_seed=np.random.randint(10000)) -> None:
        
        np.random.seed(rnd_seed)
        Agent._ids = itertools.count(0)
        self.verbose = verbose
        self.elapsed_time = 0

        self.n_iteration = n_iteration
        self.n_player = n_player
        self.n_neighbor = n_neighbor
        self.n_cave = int(n_player / (n_neighbor + 1))
        assert n_player % (n_neighbor + 1) == 0
        
        self.measure_itv = measure_itv
        self.n_ttest_sample = n_ttest_sample

        start_time = time.time()
        self.ags = [Agent() for _ in range(n_player)]
        print_std_out_err("{} agents initialized".format(n_player))
        self.all_edges = self.build_net()
        print_std_out_err("{} edges initialized".format(len(self.all_edges)))
        self.elapsed_time += time.time() - start_time

        # E(d)
        self.expected_dis = self.eval_expected_dis()
        print_std_out_err("expected dis: theory {} actual {}".format(((0.5**2)*25)**0.5, self.expected_dis))

        self.dissonance_lst = list()

    
    def build_net(self, rewire_ratio=0.1) -> list:
        all_edges = list()
        for cave_idx in range(self.n_cave):
            ag_base_idx = cave_idx * (self.n_neighbor+1)
            for u in range(self.n_neighbor+1):
                for v in range(u+1, self.n_neighbor+1):
                    all_edges.append(Edge(ag_base_idx+u, ag_base_idx+v))
        
        # rewire
        for _ in range(int(len(all_edges)*rewire_ratio)):
            e1, e2 = np.random.choice(all_edges, size=2, replace=False)
            e1.v, e2.v = e2.v, e1.v
        
        # build net
        for e in all_edges:
            self.ags[e.u].add_edge(e)
            self.ags[e.v].add_edge(e)
        
        return all_edges
    

    def eval_expected_dis(self) -> float:
        return np.mean([self.eval_dis(e) for e in self.all_edges])
    

    def eval_dis(self, e: Edge) -> float:
        ag_u, ag_v = self.ags[e.u], self.ags[e.v]
        return np.linalg.norm(ag_u.to_ndarray() - ag_v.to_ndarray())
    

    def eval_weight(self, e: Edge) -> None:
        w = self.expected_dis - self.eval_dis(e)
        e.update_weight(w)
    
    
    def update_ag_weight(self, ag: Agent) -> None:
        for e in ag.edges:
            self.eval_weight(e)

    
    def simulate_iteration(self):
        ag = np.random.choice(self.ags)
        gene_idx = np.random.choice(Network.N_DYNAMIC_GENE)

        for e in ag.edges:
            if e.w is None:
                self.eval_weight(e)

        e_prob = np.array([abs(e.w) for e in ag.edges])
        e_prob = e_prob / np.sum(e_prob)
        e_prob[-1] += 1 - np.sum(e_prob) # make the sum of prob exactly 1.0
        chosen_e = ag.edges[np.random.choice(np.arange(e_prob.shape[0]), p=e_prob)]
        chosen_gene = self.ags[chosen_e.get_other_end(ag.id)].gene[gene_idx]
        if chosen_e.w < 0:
            if draw(0.1):
                ag.alter_gene(gene_idx, Agent.flip_gene(chosen_gene))
                self.update_ag_weight(ag)
        else:
            ag.alter_gene(gene_idx, chosen_gene)
            self.update_ag_weight(ag)    
    
    
    @staticmethod
    def t_test(sample1: np.ndarray, sample2: np.ndarray, conf=0.05) -> bool:
        """ Assume unequal variance. Perform Welch's t-test. """

        t_stat, p_value = scipy.stats.ttest_ind(
            sample1, sample2,
            equal_var=False,
            alternative="two-sided"
        )
        
        return True if p_value <= conf else False
    

    def get_dissonance(self) -> float:
        return self.dissonance_lst[-1]


    def _measure_dissonance(self) -> float:
        dissonance = 0
        for e in self.all_edges:
            if e.w is None:
                self.eval_weight(e)
            o_u = self.ags[e.u].to_ndarray()[:Network.N_DYNAMIC_GENE]
            o_v = self.ags[e.v].to_ndarray()[:Network.N_DYNAMIC_GENE]
            dissonance += e.w * np.sum(2 * np.abs(o_u - o_v) - 1)
        return dissonance / (len(self.all_edges) * Network.N_DYNAMIC_GENE)
    

    def measure_record_print(self, iter_idx) -> None:
        dissonance = self._measure_dissonance()
        self.dissonance_lst.append(dissonance)
        if self.verbose:
            print_std_out_err("iter {:7d} dissonance: {:.5f}".format(iter_idx, self.get_dissonance()))
    

    def measure_pairwise_corr(self) -> float:
        abs_corr_list = list()
        for gene_i in range(Network.N_DYNAMIC_GENE):
            for gene_j in range(gene_i+1, Network.N_DYNAMIC_GENE):
                corr = np.corrcoef(
                    np.array([int(ag.gene[gene_i]) for ag in self.ags]),
                    np.array([int(ag.gene[gene_j]) for ag in self.ags])
                )
                abs_corr_list.append(abs(corr[0, 1]))
        return np.mean(abs_corr_list)
    

    def simulate(self):
        start_time = time.time()
        terminate_iter = self.n_iteration

        for iter_idx in range(1, self.n_iteration+1):
            self.simulate_iteration()

            if iter_idx % self.measure_itv == 0:
                self.measure_record_print(iter_idx)

                # condition 1: a statistically significant drop 
                if self.get_dissonance() < 0.1:
                    # condition 2: no further drop between two consecutive samples of 100 data points 
                    #   taken over 1 million iteration intervals
                    if (iter_idx % (self.measure_itv * self.n_ttest_sample) == 0 and
                        len(self.dissonance_lst) > self.n_ttest_sample * 2):

                        sample1 = np.array(self.dissonance_lst[-self.n_ttest_sample:])
                        sample2 = np.array(self.dissonance_lst[-self.n_ttest_sample*2: -self.n_ttest_sample])
                        # no statistically significant difference
                        if not self.t_test(sample1, sample2):
                            terminate_iter = iter_idx
                            break
                else:
                    if self.dissonance_lst[-2] == self.dissonance_lst[-1]:
                        terminate_iter = iter_idx
                        break

        self.elapsed_time += time.time() - start_time
        if self.verbose:
            print_std_out_err("model terminated at iteration {}".format(terminate_iter))
            print_std_out_err("elapsed time: {:.5f} hrs".format(self.elapsed_time/3600))
    
