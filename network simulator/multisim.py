# TODO
# type: ignore
import numpy as np
import attacker
import simulator

# reload = importlib.# reload
# reload(attacker)
# reload(simulator)

Attacker = attacker.Attacker
Simulator = simulator.Simulator


class MultiSim:
    def __init__(self, sim_num, **kwargs):
        self.simulator_args_default = kwargs
        self.sim_num = sim_num
        self.sims = [
            Simulator(**self.simulator_args_default) for i in range(sim_num)
        ]

    # def init_single_abstract(self, idx=0):
    #     if not self.sims[idx]:
    #         self.sims[idx] = Simulator(**self.simulator_args_default)

    def run_single_abstract(self, idx):
        if not self.sims[idx].abstract_run:
            self.sims[idx].run_sim_abstract()

    def run_all_abstract(self, sim_template=None):
        for idx, sim in enumerate(self.sims):
            # if idx % 25 == 0:
            # print('========')
            # print('Abstract run for sim {}'.format(idx))
            # print('========')
            if not sim.abstract_run:
                sim.run_sim_abstract()

    def duplicate_abstract(self, sim_template_input):
        for idx, sim in enumerate(self.sims):
            if sim is not sim_template_input:
                self.sims[idx] = Simulator(**sim_template_input.get_abstract())

    def run_single_concrete(self, idx):
        if not self.sims[idx].concrete_run:
            self.sims[idx].run_sim_concrete()

    def run_all_concrete(self):
        for idx, sim in enumerate(self.sims):
            # if idx % 25 == 0:
            # print('========')
            # print('Concrete run for sim {}'.format(idx))
            # print('========')
            if not sim.concrete_run:
                sim.run_sim_concrete()

    def get_average_transactions(self, transaction_type):
        return np.average(
            [
                len(sim.attacker.transactions[transaction_type])
                for sim in self.sims
            ]
        )

    # TODO: Why is this not in Tester?
    def order_test(self):
        self.run_all_abstract()
        self.run_all_concrete()
        new_multi_sim = MultiSim(self.sim_num, **self.simulator_args_default)
        for idx, sim in enumerate(self.sims):
            new_multi_sim.sims[idx] = Simulator(**sim.get_abstract())
            seq_attacker = Attacker.construct_from_template_with_seq(
                sim.attacker
            )
            seq_attacker.transactions = {"deal": [], "cheat": []}
            seq_attacker.seq.sort(key=lambda x: Attacker.actions.index(x[0]))
            new_multi_sim.sims[idx].attacker = seq_attacker
        new_multi_sim.run_all_concrete()
        return self, new_multi_sim
