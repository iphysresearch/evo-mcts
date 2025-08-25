import copy

import json
import random
import time

from .mcts import MCTS, MCTSNode
from .evolution_interface import InterfaceEC


# main class for evo_mcts
class EvoMCTS:
    def __init__(self, paras, problem, select, manage, **kwargs):

        self.prob = problem
        self.select = select
        self.manage = manage
        # LLM settings
        self.use_local_llm = paras.llm_use_local
        self.url = paras.llm_local_url
        self.api_endpoint = paras.llm_api_endpoint  # currently only API2D + GPT
        self.api_key = paras.llm_api_key
        self.llm_model = paras.llm_model
        self.temperature = paras.temperature

        # ------------------ RZ: use local LLM ------------------
        self.use_local_llm = kwargs.get('use_local_llm', False)
        assert isinstance(self.use_local_llm, bool)
        if self.use_local_llm:
            assert 'url' in kwargs, 'The keyword "url" should be provided when use_local_llm is True.'
            assert isinstance(kwargs.get('url'), str)
            self.url = kwargs.get('url')
        # -------------------------------------------------------

        # Experimental settings       
        self.init_size = paras.init_size  # popopulation size, i.e., the number of algorithms in population
        self.pop_size = paras.pop_size  # popopulation size, i.e., the number of algorithms in population
        self.fe_max = paras.ec_fe_max  # function evaluation times
        self.eval_times = 0  # number of populations

        self.operators = paras.ec_operators
        self.operator_weights = paras.ec_operator_weights
        # paras.ec_m = 5
        self.m = paras.ec_m

        self.debug_mode = paras.exp_debug_mode  # if debug
        self.ndelay = 1  # default

        self.use_seed = paras.exp_use_seed
        self.seed_path = paras.exp_seed_path
        self.load_pop = paras.exp_use_continue
        self.load_pop_path = paras.exp_continue_path
        self.load_pop_id = paras.exp_continue_id

        self.output_path = paras.exp_output_path
        self.resume_dirs = paras.exp_use_continue

        self.exp_n_proc = paras.exp_n_proc

        self.timeout = paras.eva_timeout

        self.use_numba = paras.eva_numba_decorator

        print("- Evo-MCTS parameters loaded -")

        # Set a random seed
        random.seed(2024)
        self.build_evolution_cache()

    # add new individual to population
    def add2pop(self, population, offspring):
        for ind in population:
            if ind['algorithm'] == offspring['algorithm']:
                if (self.debug_mode):
                    print("duplicated result, retrying ... ")
        population.append(offspring)

    def expand(self, mcts, cur_node, nodes_set, option, iteration=None, log_info=None):
        if self.check_exist_file(self.eval_times, iteration, cur_node.depth+1, option):
            offsprings = self.check_exist_file(self.eval_times, iteration, cur_node.depth+1, option)
            self.eval_times = offsprings['eval_times']

            print(f"(Loaded) Action: {option}, Father Obj: {cur_node.raw_info['objective']}, Now Obj: {offsprings['objective']}, Depth: {cur_node.depth + 1}")
            self.add2pop(nodes_set, offsprings)  # Check duplication, and add the new offspring
            size_act = min(len(nodes_set), self.pop_size)
            nodes_set = self.manage.population_management(nodes_set, size_act)
            nownode = MCTSNode(offsprings['algorithm'], offsprings['code'], offsprings['objective'],
                               parent=cur_node, depth=cur_node.depth + 1,
                               visit=1, Q=-1 * offsprings['objective'], raw_info=offsprings)
            if option == 'e1':
                nownode.subtree.append(nownode)
            cur_node.add_child(nownode)
            cur_node.children_info.append(offsprings)
            mcts.backpropagate(nownode)
            self.interface_ec.update_elite_offspring(offsprings)            
            return nodes_set

        if option in ['s1', 's2', 's3']:
            path_set = []
            now = copy.deepcopy(cur_node)
            while now.code != "Root":
                path_set.append(now.raw_info)
                now = copy.deepcopy(now.parent)
            path_set = self.manage.population_management_s1(path_set, len(path_set))
            if len(path_set) == 1:
                return nodes_set
            self.eval_times, offsprings = self.interface_ec.evolve_algorithm(self.eval_times, path_set,
                                                                             cur_node.raw_info,
                                                                             cur_node.children_info, option, iteration=iteration, depth=cur_node.depth+1, log_info=log_info)
        elif option == 'e1':
            e1_set = [copy.deepcopy(children.subtree[random.choices(range(len(children.subtree)), k=1)[0]].raw_info) for
                      children in mcts.root.children]
            self.eval_times, offsprings = self.interface_ec.evolve_algorithm(self.eval_times, e1_set,
                                                                             cur_node.raw_info,
                                                                             cur_node.children_info, option, iteration=iteration, depth=cur_node.depth+1, log_info=log_info)
        elif option == 'e3':
            self.eval_times, offsprings = self.interface_ec.evolve_algorithm(self.eval_times, nodes_set,
                                                                             cur_node.raw_info,
                                                                             cur_node.parent.children_info, option, iteration=iteration, depth=cur_node.depth+1, log_info=log_info)
        else:
            self.eval_times, offsprings = self.interface_ec.evolve_algorithm(self.eval_times, nodes_set,
                                                                             cur_node.raw_info,
                                                                             cur_node.children_info, option, iteration=iteration, depth=cur_node.depth+1, log_info=log_info)
        if offsprings == None:
            print(f"Timeout emerge, no expanding with action {option}.")
            return nodes_set

        if option != 'e1':
            print(
                f"Action: {option}, Father Obj: {cur_node.raw_info['objective']}, Now Obj: {offsprings['objective']}, Depth: {cur_node.depth + 1}")
        else:
            if self.interface_ec.check_duplicate_obj(mcts.root.children_info, offsprings['objective']):
                print(f"Duplicated e1, no action, Father is Root, Abandon Obj: {offsprings['objective']}")
                return nodes_set
            else:
                print(f"Action: {option}, Father is Root, Now Obj: {offsprings['objective']}")
        if offsprings['objective'] != float('inf'):
            self.add2pop(nodes_set, offsprings)  # Check duplication, and add the new offspring
            size_act = min(len(nodes_set), self.pop_size)
            nodes_set = self.manage.population_management(nodes_set, size_act)
            nownode = MCTSNode(offsprings['algorithm'], offsprings['code'], offsprings['objective'],
                               parent=cur_node, depth=cur_node.depth + 1,
                               visit=1, Q=-1 * offsprings['objective'], raw_info=offsprings)
            if option == 'e1':
                nownode.subtree.append(nownode)
            cur_node.add_child(nownode)
            cur_node.children_info.append(offsprings)
            mcts.backpropagate(nownode)
            self.interface_ec.update_elite_offspring(offsprings)
        return nodes_set

 
    def run(self):
        print("- Initialization Start (w. reflection) -")

        interface_prob = self.prob

        # 创建MCTS树
        mcts = MCTS('Root')
        # 设置最大深度
        mcts.max_depth = 10  # 设置一个合理的最大深度值
        
        # main loop
        self.operators = ['e3']*5 + ['s2','s3','m3','m6','m7'] # ['e1', 'e2', 'm1', 'm2', 's1']
        n_op = len(self.operators)  # ['e1', 'e2',f 'm1', 'm2', 's1']

        # interface for ec operators
        self.interface_ec = InterfaceEC(self.m, self.api_endpoint, self.api_key, self.llm_model,
                                        self.debug_mode, interface_prob, use_local_llm=self.use_local_llm, url=self.url,
                                        select=self.select, n_p=self.exp_n_proc,
                                        timeout=self.timeout, use_numba=self.use_numba,
                                        max_depth=mcts.max_depth,
                                        temperature=self.temperature
                                        )

        # 检查是否有缓存的树结构可以恢复
        last_eval_times = self.get_last_eval_times()
        if last_eval_times > 0:
            print(f"Found cached data with eval_times={last_eval_times}, attempting to restore MCTS tree...")
            nodes_set, restored_mcts, iteration, parent_node = self.restore_mcts_tree(mcts)
            if nodes_set:
                mcts = restored_mcts
                print(f"Successfully restored MCTS tree at eval_times={self.eval_times}, iteration={iteration}")
                iteration -= 1
            else:
                print("Failed to restore MCTS tree, starting from scratch")
                iteration = 0
                nodes_set = []
                parent_node = None
        else:
            iteration = 0
            nodes_set = []
            parent_node = None

        # 如果没有恢复树，则初始化种群
        if self.eval_times == 0:
            brothers = []
            # Init population
            # init_operators = ['i2'] + ['i4'] * self.init_size + ['m6', 'm7']
            init_operators = ['i2', 'i2_1', 'i2_2', 'i2_3', 'i2_4', 'i2_5', 'i2_6', 'i2_7'] + ['m6', 'm7']
            for i in range(len(init_operators)):
                depth = mcts.root.depth + 1
                if self.check_exist_file(self.eval_times, iteration, depth, init_operators[i]):
                    offsprings = self.check_exist_file(self.eval_times, iteration, depth, init_operators[i])
                    self.eval_times = offsprings['eval_times']
                else:
                    self.eval_times, brothers, offsprings = self.interface_ec.get_algorithm(self.eval_times, brothers, init_operators[i],
                                                                                            iteration=iteration, depth=depth,
                                                                                            log_info=f'Iteration: {iteration}, Depth: {depth}/{mcts.max_depth}, OP: {init_operators[i]}')
                    if offsprings is None:
                        continue
                self.interface_ec.update_elite_offspring(offsprings)
                brothers.append(offsprings)
                nownode = MCTSNode(offsprings['algorithm'], offsprings['code'], offsprings['objective'], parent=mcts.root,
                                depth=1, visit=1, Q=-1 * offsprings['objective'], raw_info=offsprings)
                mcts.root.add_child(nownode)
                mcts.root.children_info.append(offsprings)
                mcts.backpropagate(nownode)
                nownode.subtree.append(nownode)
            nodes_set = brothers
            size_act = min(len(nodes_set), self.pop_size)
            nodes_set = self.manage.population_management(nodes_set, size_act)
            print("- Initialization Finished - Evolution Start -")
        
        # 主循环
        while self.eval_times < self.fe_max:
            iteration += 1
            print(f"Current performances of MCTS nodes: {mcts.rank_list}")
            # print([len(node.subtree) for node in mcts.root.children])
            cur_node = mcts.root if (parent_node is None or parent_node.depth+1 == mcts.max_depth) else parent_node  # 从根节点开始(当来自初始化树或缓存达到最大深度)，从当前节点开始(当来自缓存)
            while len(cur_node.children) > 0 and cur_node.depth < mcts.max_depth:
                uct_scores = [mcts.uct(node, max(1 - self.eval_times / self.fe_max, 0)) for node in cur_node.children]
                selected_pair_idx = uct_scores.index(max(uct_scores))
                cur_node = cur_node.children[selected_pair_idx]
                # breakpoint()
                # 只有在未达到最大深度时才进行扩展操作
                if cur_node.depth < mcts.max_depth:
                    for i in range(n_op):
                        op = self.operators[i]
                        print(f"Iter: {self.eval_times}/{self.fe_max} OP: {op}")
                        nodes_set = self.expand(mcts, cur_node, nodes_set, op, iteration=iteration,
                                                log_info=f'Iteration: {iteration}, Depth: {cur_node.depth+1}/{mcts.max_depth}, OP: {op}')
                        # breakpoint()
                        assert len(cur_node.children) == len(cur_node.children_info)
                else:
                    print(f"Iter: {self.eval_times}/{self.fe_max} - Maximum depth reached, skipping expansion")
            
            # 保存population到文件
            filename = self.output_path + "population_generation_" + str(self.eval_times) + ".json"
            with open(filename, 'w') as f:
                json.dump(nodes_set, f, indent=5)

            # 保存MCTS树结构信息
            mcts_tree_info = self.serialize_mcts_tree(mcts.root)
            mcts_filename = self.output_path + "mcts_tree_" + str(self.eval_times) + ".json"
            with open(mcts_filename, 'w') as f:
                json.dump(mcts_tree_info, f, indent=5)

            # Save the best one to a file
            filename = self.output_path + "best_population_generation_" + str(self.eval_times) + ".json"
            with open(filename, 'w') as f:
                json.dump(nodes_set[0], f, indent=5)

        return nodes_set[0]["code"], filename

    def get_last_eval_times(self):
        """获取缓存中最后一次评估的eval_times"""
        if not hasattr(self, 'evolution_cache') or not self.evolution_cache:
            return 0
            
        # 找到缓存中最大的eval_times
        max_eval_times = 0
        for key in self.evolution_cache.keys():
            eval_times = key[0]
            if eval_times > max_eval_times:
                max_eval_times = eval_times
                
        return max_eval_times
        
    def restore_mcts_tree(self, mcts):
        """从缓存中恢复MCTS树结构"""
        if not hasattr(self, 'evolution_cache') or not self.evolution_cache:
            return None, None, 0
            
        # 按照eval_times排序所有缓存项
        sorted_cache = sorted(self.evolution_cache.items(), key=lambda x: x[0][0])
        
        # 创建节点映射表，用于快速查找父节点
        node_map = {0: mcts.root}  # 根节点ID为0
        
        # 用于存储所有个体的集合
        nodes_set = []
        max_iteration = 0
        
        # 遍历所有缓存项，按顺序重建树
        for (eval_times, iteration, depth, op), data in sorted_cache:
            if iteration > max_iteration:
                max_iteration = iteration
                
            # 更新当前的eval_times
            self.eval_times = eval_times
            
            # 对于深度为1的节点，它们是根节点的直接子节点
            if depth == 1:
                parent_node = mcts.root
                
                # 创建新节点
                new_node = MCTSNode(
                    data['algorithm'], 
                    data['code'], 
                    data['objective'],
                    parent=parent_node,
                    depth=depth,
                    visit=1,
                    Q=-1 * data['objective'],
                    raw_info=data
                )
                
                # 添加到父节点
                parent_node.add_child(new_node)
                parent_node.children_info.append(data)
                
                # 更新节点映射
                node_id = len(node_map)
                node_map[node_id] = new_node
                
                # 添加到subtree（如果是e1操作）
                if op == 'e1':
                    new_node.subtree.append(new_node)
                
                # 添加到nodes_set
                nodes_set.append(data)
                
                # 反向传播
                mcts.backpropagate(new_node)
                
            # 对于深度大于1的节点，需要找到合适的父节点
            elif depth > 1:
                # 尝试找到合适的父节点（深度为depth-1的节点）
                potential_parents = [node for node_id, node in node_map.items() if node.depth == depth - 1]
                
                if potential_parents:
                    # 选择最近添加的父节点（简单策略）
                    parent_node = potential_parents[-1]
                    
                    # 创建新节点
                    new_node = MCTSNode(
                        data['algorithm'], 
                        data['code'], 
                        data['objective'],
                        parent=parent_node,
                        depth=depth,
                        visit=1,
                        Q=-1 * data['objective'],
                        raw_info=data
                    )
                    
                    # 添加到父节点
                    parent_node.add_child(new_node)
                    parent_node.children_info.append(data)
                    
                    # 更新节点映射
                    node_id = len(node_map)
                    node_map[node_id] = new_node
                    
                    # 添加到subtree（如果是e1操作）
                    if op == 'e1':
                        new_node.subtree.append(new_node)
                    
                    # 添加到nodes_set
                    nodes_set.append(data)
                    
                    # 反向传播
                    mcts.backpropagate(new_node)
        
        # 对nodes_set进行管理
        if nodes_set:
            size_act = min(len(nodes_set), self.pop_size)
            nodes_set = self.manage.population_management(nodes_set, size_act)
            
            # 更新interface_ec的精英后代
            for node in nodes_set:
                self.interface_ec.update_elite_offspring(node)
                
            return nodes_set, mcts, max_iteration, parent_node
            
        return None, None, 0, None

    def build_evolution_cache(self):
        """Preload all evolution JSON files into memory cache"""
        import os
        import json
        import glob
        
        # Initialize the evolution cache
        self.evolution_cache = {}

        # Collect json files from all resume directories if available
        json_files = []
        if self.resume_dirs and isinstance(self.resume_dirs, list) and len(self.resume_dirs) > 0:
            # Collect files from all resume directories
            for resume_dir in self.resume_dirs:
                evolutions_dir = os.path.join(resume_dir, "evaluations")
                if os.path.exists(evolutions_dir):
                    dir_json_files = glob.glob(os.path.join(evolutions_dir, "**", "*.json"), recursive=True)
                    json_files.extend(dir_json_files)
                    print(f"Found {len(dir_json_files)} json files in {evolutions_dir}")
                else:
                    print(f"Evolutions directory {evolutions_dir} does not exist")
        else:
            # Check the current output directory
            evolutions_dir = os.path.join(self.output_path, "evaluations")
            if os.path.exists(evolutions_dir):
                json_files = glob.glob(os.path.join(evolutions_dir, "**", "*.json"), recursive=True)
                print(f"Found {len(json_files)} json files in {evolutions_dir}")
            else:
                print(f"Evolutions directory {evolutions_dir} does not exist")
        
        # Sort all collected json files by modification time
        json_files = sorted(json_files, key=os.path.getmtime)
        # Load each file and store in the dictionary using iteration, depth, op as keys
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and 'iteration' in data and 'depth' in data and 'operator' in data and 'eval_times' in data:
                        key = (data['eval_times'], data['iteration'], data['depth'], data['operator'])
                        self.evolution_cache[key] = data
                        print(f"Loaded evolution data for eval_times {data['eval_times']}, iteration {data['iteration']}, depth {data['depth']} and operator {data['operator']}")
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading file {file_path}: {e}")
    
    def check_exist_file(self, eval_times, iteration, depth, op):
        """Check if a specific evolution result exists in the cache"""
        
        # Initialize cache if it doesn't exist
        if not hasattr(self, 'evolution_cache'):
            self.build_evolution_cache()
                
        # Check if the specific combination exists in our cache
        if eval_times is not None and iteration is not None and depth is not None and op is not None:
            key = (eval_times, iteration, depth, op)
            return self.evolution_cache.get(key)
            
        return None


    def serialize_mcts_tree(self, node, node_id=0):
        """递归序列化MCTS树，保存节点信息和关联关系"""
        tree_info = {
            'nodes': [],
            'edges': []
        }
        
        # 辅助函数：递归构建树结构
        def build_tree_recursive(current_node, parent_id, current_id):
            # 添加当前节点信息
            node_info = {
                'id': current_id,
                'depth': current_node.depth,
                'visits': current_node.visits,
                'Q': float(current_node.Q),
                'is_terminal': len(current_node.children) == 0,
                'timestamp': current_node.timestamp if hasattr(current_node, 'timestamp') else time.time(),
            }
            
            # 添加额外的节点特定信息(如果有)
            if hasattr(current_node, 'code') and current_node.code is not None:
                node_info['code'] = current_node.code
            if hasattr(current_node, 'raw_info') and current_node.raw_info is not None:
                node_info['raw_info'] = current_node.raw_info
            tree_info['nodes'].append(node_info)
            
            # 如果不是根节点，添加与父节点的边
            if parent_id != -1:
                edge_info = {
                    'source': parent_id,
                    'target': current_id
                }
                
                # 添加操作符信息
                if hasattr(current_node, 'raw_info') and current_node.raw_info is not None and 'operator' in current_node.raw_info:
                    edge_info['operator'] = current_node.raw_info['operator']
                
                tree_info['edges'].append(edge_info)
            
            # 递归处理所有子节点
            child_id = current_id + 1
            for i, child in enumerate(current_node.children):
                # 递归处理子节点
                next_id = build_tree_recursive(child, current_id, child_id)
                child_id = next_id
            
            return child_id
        
        # 从根节点开始构建树
        build_tree_recursive(node, -1, node_id)
        return tree_info
