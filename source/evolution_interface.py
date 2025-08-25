import copy
import random

import numpy as np
import time

from .evolution import Evolution
import warnings
from joblib import Parallel, delayed
import re
import concurrent.futures


class InterfaceEC():
    def __init__(self, m, api_endpoint, api_key, llm_model, debug_mode, interface_prob, select, n_p, timeout, use_numba,
                 **kwargs):
        assert 'use_local_llm' in kwargs
        assert 'url' in kwargs
        # -----------------------------------------------------------

        # LLM settings
        self.interface_eval = interface_prob
        prompts = interface_prob.prompts
        self.evol = Evolution(api_endpoint, api_key, llm_model, debug_mode, prompts, **kwargs)
        self.m = m
        self.debug = debug_mode

        if not self.debug:
            warnings.filterwarnings("ignore")

        self.select = select
        self.n_p = n_p

        self.timeout = timeout
        self.use_numba = use_numba
        self.elite_offspring = {
            'reflection': None,
            'algorithm': None,
            'thought': None,
            'code': None,
            'objective': None,
            'other_inf': None,
        }

    def update_elite_offspring(self, offspring):
        if offspring is None:
            return
        if self.elite_offspring['objective'] is None or offspring['objective'] < self.elite_offspring['objective']:
            for key in self.elite_offspring:
                self.elite_offspring[key] = offspring[key]

    def code2file(self, code):
        with open("./ael_alg.py", "w") as file:
            # Write the code to the file
            file.write(code)
        return

    def add2pop(self, population, offspring):
        for ind in population:
            if ind['objective'] == offspring['objective']:
                if self.debug:
                    print("duplicated result, retrying ... ")
                return False
        population.append(offspring)
        return True

    def check_duplicate_obj(self, population, obj):
        for ind in population:
            if obj == ind['objective']:
                return True
        return False

    def check_duplicate(self, population, code):
        for ind in population:
            if code == ind['code']:
                return True
        return False

    def population_generation_seed(self, seeds):

        population = []

        fitness = self.interface_eval.batch_evaluate([seed['code'] for seed in seeds])

        for i in range(len(seeds)):
            try:
                seed_alg = {
                    'algorithm': seeds[i]['algorithm'],
                    'code': seeds[i]['code'],
                    'objective': None,
                    'other_inf': None
                }

                obj = np.array(fitness[i])
                seed_alg['objective'] = np.round(obj, 5)
                population.append(seed_alg)

            except Exception as e:
                print("Error in seed algorithm")
                exit()

        print("Initiliazation finished! Get " + str(len(seeds)) + " seed algorithms")

        return population

    def _get_alg(self, pop, operator, father=None, depth=None, rechat=False):
        self.evol.rechat = rechat
        offspring = {
            'reflection': None,
            'algorithm': None,
            'thought': None,
            'code': None,
            'objective': None,
            'other_inf': None,
            'operator': operator,
            'depth': depth,
            'rechat': rechat,
            'user_content': None,
            'system_content': None
        }
        if operator == "i1":
            parents = None
            [offspring['code'], offspring['thought']] = self.evol.i1()
        elif operator in ["i2", "i2_1", "i2_2", "i2_3", "i2_4", "i2_5", "i2_6", "i2_7"]:
            parents = None
            self.evol.i2(offspring) # use seed function as initial algorithm
        elif operator == "i3":
            parents = None
            [offspring['code'], offspring['thought']] = self.evol.i3() # generate algorithms based on seed function
        elif operator == "i4":
            parents = None
            [offspring['code'], offspring['thought']] = self.evol.i4(depth=depth) # generate algorithms based on seed function
        elif operator == "R":
            parents = None
            # Find the seed algorithm in the population
            seed_alg = next((ind for ind in pop if ind.get('other_inf') == 'seed'), None)
            if seed_alg['objective'] > father['objective']:
                code_worse = seed_alg['code']
                code_better = father['code']
            else:
                code_worse = father['code']
                code_better = seed_alg['code']
            reflection, system_content, prompt_content, thinking = self.evol.R(code_worse, code_better, depth=depth) # Select the better (seed vs offspring)
            return reflection, system_content, prompt_content, thinking
            
        elif operator == "e1":
            real_m = random.randint(2, self.m)
            real_m = min(real_m, len(pop))
            parents = self.select.parent_selection_e1(pop, real_m)
            [offspring['code'], offspring['thought']] = self.evol.e1(parents)
        elif operator == "e2":
            other = copy.deepcopy(pop)
            if father in pop:
                other.remove(father)
            real_m = 1
            # real_m = random.randint(2, self.m) - 1
            # real_m = min(real_m, len(other))
            parents = self.select.parent_selection(other, real_m)
            parents.append(father)
            [offspring['code'], offspring['thought']] = self.evol.e2(parents)
        elif operator == "e3":
            other = copy.deepcopy(pop)
            other = [ind for ind in other if ind['objective'] != father['objective']] # DEBUG: 有可能len(other) == 0
            parents = random.choices(other, k=1)
            self.evol.e3(offspring, parents[0], father, depth=depth)
        elif operator == "m1":
            parents = [father]
            [offspring['code'], offspring['thought']] = self.evol.m1(parents[0])
        elif operator == "m2":
            parents = [father]
            [offspring['code'], offspring['thought']] = self.evol.m2(parents[0])
        elif operator == "m3":
            # Copy population and select parents based on objective value weights
            other = copy.deepcopy(pop)
            other = [ind for ind in other if ind['code'] != father['code']]
            # Calculate weights based on objective values (lower is better)
            weights = [1.0 / (ind['objective'] + 1e-10) for ind in other]  # Add small epsilon to avoid division by zero
            # Normalize weights
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            # Select parents based on weights
            parents = random.choices(other, weights=normalized_weights, k=min(self.m, len(other)))
            self.evol.m3(offspring, parents, father, depth=depth)
        elif operator == "m6":
            other = copy.deepcopy(pop)
            # Remove the elite offspring from the population if it exists
            other = [ind for ind in other if ind['objective'] != self.elite_offspring['objective']] # DEBUG: 有可能len(other) == 0
            real_m = 1
            parents = self.select.parent_selection(other, real_m)
            self.evol.m6(offspring, parents[0], self.elite_offspring, depth=depth)
        elif operator == "m7":
            other = copy.deepcopy(pop)
            other = [ind for ind in other if ind['objective'] != self.elite_offspring['objective']]
            parents = self.select.parent_selection(other, self.m, unique=True)
            self.evol.m7(offspring, parents, self.elite_offspring, depth=depth)
        elif operator == "s1":
            parents = pop
            [offspring['code'], offspring['thought']] = self.evol.s1(pop)
        elif operator == "s2":
            parents = pop
            self.evol.s2(offspring, pop, father, depth=depth)
        elif operator == "s3":
            parents = pop
            self.evol.s3(offspring, pop, father, depth=depth)
        else:
            print(f"Evolution operator [{operator}] has not been implemented ! \n")

        self.evol.post_thought(offspring)
        offspring['timestamp'] = time.time()
        return parents, offspring

    def get_offspring(self, pop, operator, father=None, depth=None, rechat=False):
        while True:
            try:
                p, offspring = self._get_alg(pop, operator, father=father, depth=depth, rechat=rechat)
                code = offspring['code']
                n_retry = 1
                while self.check_duplicate(pop, offspring['code']):
                    n_retry += 1
                    if self.debug:
                        print("duplicated code, wait 1 second and retrying ... ")
                    p, offspring = self._get_alg(pop, operator, father=father, depth=depth, rechat=True)
                    code = offspring['code']
                    if n_retry > 1:
                        break
                break                
            except Exception as e:
                print(e)
                if e == 'list index out of range':
                    breakpoint()
                breakpoint()
        self.evol.rechat = False
        return p, offspring

    def get_algorithm(self, eval_times, pop, operator, iteration, depth=None, log_info=None):
        rechat = False
        num = 3
        for i in range(num):
            eval_times += 1
            parents, offspring = self.get_offspring(pop, operator, depth=depth, rechat=rechat)
            objs, runid = self.interface_eval.batch_evaluate([offspring['code']], eval_times=eval_times, iteration=iteration, depth=depth, operator=operator, log_info=log_info, reflections=[offspring['reflection']])
            if operator not in ["i2", "i2_1", "i2_2", "i2_3", "i2_4", "i2_5", "i2_6", "i2_7"] and (objs == 'timeout' or objs[0] == float('inf') or objs[0] == 0.0 or self.check_duplicate_obj(pop, np.round(objs[0], 5))):
                print(f"Evaluation failed for offspring {objs}. Retrying... (num: {i+1}/{num})")
                self.interface_eval.cleanup_evaluation_files(runid)
                rechat = True
                continue
            offspring['objective'] = np.round(objs[0], 5)
            if operator in ["i3", "i4"] and any(ind['other_inf'] == 'seed' for ind in pop):
                offspring['reflection'], offspring['R_system_content'], offspring['R_user_content'], offspring['R_thinking'] = self._get_alg(pop, "R", father=offspring, depth=depth) # Add reflectionr (seed vs offspring)

            # Cache offspring
            offspring['iteration'] = iteration
            offspring['eval_times'] = eval_times
            self.cache_offspring(offspring, runid)
            return eval_times, pop, offspring
        return eval_times, pop, None

    def evolve_algorithm(self, eval_times, pop, node, brother_nodes=None, operator=None, iteration=None, depth=None, log_info=None):
        rechat = False
        num = 3
        for i in range(num):
            eval_times += 1
            if brother_nodes:
                _, offspring = self.get_offspring(brother_nodes, operator, father=node, depth=depth, rechat=rechat)
            else:
                _, offspring = self.get_offspring(pop, operator, father=node, depth=depth, rechat=rechat)
            objs, runid = self.interface_eval.batch_evaluate([offspring['code']], eval_times=eval_times, iteration=iteration, depth=depth, operator=operator, log_info=log_info, reflections=[offspring['reflection']])
            if objs == 'timeout':
                print(f"Evaluation failed for offspring {objs}. Retrying... (num: {i+1}/{num})")
                self.interface_eval.cleanup_evaluation_files(runid)
                rechat = True
                return eval_times, None
            if objs[0] == float('inf') or objs[0] == 0.0 or self.check_duplicate(pop, offspring['code']) or self.check_duplicate_obj(pop, np.round(objs[0], 5)):
                print(f"Evaluation failed for offspring {objs}. Retrying... (num: {i+1}/{num})")
                self.interface_eval.cleanup_evaluation_files(runid)
                rechat = True
                continue
            offspring['objective'] = np.round(objs[0], 5)
            
            # Cache offspring
            offspring['iteration'] = iteration
            offspring['eval_times'] = eval_times
            self.cache_offspring(offspring, runid)
            return eval_times, offspring
        return eval_times, None

    def cache_offspring(self, offspring, runid):
        import os
        import json
        outdir = './evaluations/'
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        # Write response to file
        file_name = outdir + f"problem_eval{runid}_offspring.json"
        with open(file_name, 'w', encoding='utf-8') as file:
            json.dump(offspring, file)
