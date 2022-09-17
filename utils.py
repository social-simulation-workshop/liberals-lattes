import itertools
import numpy as np
import time


def draw(p) -> bool:
    return True if np.random.uniform() < p else False


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
        """ Return 25 randomly generated genes. """
        return "".join([f"{b:0>8b}" for b in np.random.bytes(4)])[:25]
    
    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)
    
    @staticmethod
    def flip_gene(bit) -> str:
        return "1" if bit == "0" else "0"


class Network:
    N_STATIC_GENE = 5
    N_DYNAMIC_GENE = 20
    N_GENE = N_STATIC_GENE + N_DYNAMIC_GENE

    EXPECTED_DIS_GENE = 0.5
    EXPECTED_DIS = (EXPECTED_DIS_GENE ** 2 * N_GENE) ** 0.5

    def __init__(self, n_player=500, n_neighbor=99, n_iteration=1000000,
        measure_itv=100, verbose=True, rnd_seed=np.random.randint(10000)) -> None:
        
        np.random.seed(rnd_seed)
        Agent._ids = itertools.count(0)
        self.verbose = verbose
        self.measure_itv = measure_itv
        self.elapsed_time = 0

        self.n_iteration = n_iteration
        self.n_player = n_player
        self.n_neighbor = n_neighbor
        self.n_cave = int(n_player / (n_neighbor + 1))
        assert n_player % (n_neighbor + 1) == 0

        start_time = time.time()
        self.ags = [Agent() for _ in range(n_player)]
        print("{} agents initialized".format(n_player))
        self.all_edges = self.build_net()
        print("{} edges initialized".format(len(self.all_edges)))
        self.elapsed_time += time.time() - start_time

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
    

    def eval_weight(self, e: Edge) -> None:
        g_u = self.ags[e.u].gene
        g_v = self.ags[e.v].gene
        
        d = sum([(int(g_u[g_idx]) - int(g_v[g_idx])) ** 2 for g_idx in range(Network.N_GENE)]) ** 0.5
        w = Network.EXPECTED_DIS - d
        e.update_weight(w)

    
    def simulate_iteration(self):
        ag = np.random.choice(self.ags)
        gene_idx = np.random.choice(Network.N_DYNAMIC_GENE)

        for e in ag.edges:
            if e.w is None:
                self.eval_weight(e)

        urn = list()
        w_abs_sum = sum([abs(e.w) for e in ag.edges])
        for e in ag.edges:
            ag_other = self.ags[e.get_other_end(ag.id)]
            if draw(abs(e.w)/w_abs_sum):
                if e.w < 0:
                    if draw(0.1):
                        urn.append(Agent.flip_gene(ag_other.gene[gene_idx]))
                else:
                    urn.append(ag_other.gene[gene_idx])

        if urn:
            ag_g = list(ag.gene)
            ag_g[gene_idx] = np.random.choice(urn)
            ag.gene = "".join(ag_g)
            for e in ag.edges:
                self.eval_weight(e)
    
    
    def measure_dissonance(self) -> float:
        dissonance = 0
        for e in self.all_edges:
            if e.w is None:
                self.eval_weight(e)
            g_u = self.ags[e.u].gene
            g_v = self.ags[e.v].gene
            dissonance += e.w * sum([2 * abs(int(g_u[g_idx]) - int(g_v[g_idx])) - 1 for g_idx in range(Network.N_DYNAMIC_GENE)])
        return dissonance / len(self.all_edges)
    

    def measure_pairwise_corr(self) -> float:
        corr_list = list()
        for ag_a_idx in range(len(self.ags)):
            for ag_b_idx in range(ag_a_idx, len(self.ags)):
                g_a = np.array([int(g) for g in self.ags[ag_a_idx].gene])
                g_b = np.array([int(g) for g in self.ags[ag_b_idx].gene])
                corr_list.append(np.corrcoef(g_a, g_b)[0, 1])
        return sum(corr_list) / len(corr_list)
    

    def simulate(self):
        start_time = time.time()
        for iter_idx in range(1, self.n_iteration+1):
            # print("iter {:7d}".format(iter_idx))
            self.simulate_iteration()

            if iter_idx % self.measure_itv == 0:
                dissonance = self.measure_dissonance()
                self.dissonance_lst.append(dissonance)
                if self.verbose:
                    print("iter {:7d} dissonance: {:.5f}".format(iter_idx, self.dissonance_lst[-1]))
        self.elapsed_time += time.time() - start_time
        print("elapsed time: {:.5f} hrs".format(self.elapsed_time/3600))
    
