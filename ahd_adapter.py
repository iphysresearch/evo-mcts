from source.evo_mcts import EvoMCTS
from source.getParas import Paras
from source import prob_rank, pop_greedy
from problem_adapter import Problem

from utils.utils import init_client

class AHD:
    def __init__(self, cfg, root_dir, workdir, client) -> None:
        self.cfg = cfg
        self.root_dir = root_dir
        self.problem = Problem(cfg, root_dir)

        self.paras = Paras() 
        self.paras.set_paras(method = "evo_mcts",
                             init_size = self.cfg.init_pop_size,
                             pop_size = self.cfg.pop_size,
                             llm_model = client,
                             ec_fe_max = self.cfg.max_fe,
                             exp_output_path = f"{workdir}/",
                             exp_debug_mode = self.cfg.debug_mode,
                             eva_timeout=self.cfg.timeout,
                             exp_use_seed=self.cfg.use_seed,
                             temperature=self.cfg.llm_client.temperature,
                             exp_use_continue=list(self.cfg.resume_dirs))
        init_client(self.cfg)
    
    def evolve(self):
        print("- Evolution Start -")

        method = EvoMCTS(self.paras, self.problem, prob_rank, pop_greedy)

        # method.run_demo()
        # return

        results = method.run()

        print("> End of Evolution! ")
        print("-----------------------------------------")
        print("---  Evo-MCTS successfully finished!  ---")
        print("-----------------------------------------")

        return results


