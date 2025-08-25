import logging
import os
import subprocess
import re
from typing import List, Any

from utils.utils import block_until_running, file_to_string, filter_traceback


class Prompts:
    def __init__(self, problem_cfg, root_dir: str):
        self.cfg = problem_cfg
        self.problem = problem_cfg.problem_name
        self.root_dir = root_dir
        self.prompt_dir = f"{self.root_dir}/prompts"

        problem_prompt_path = f'{self.prompt_dir}/{self.problem}'
        self.func_signature = file_to_string(f'{problem_prompt_path}/func_signature.txt').format(version=2).strip()
        self.func_desc = file_to_string(f'{problem_prompt_path}/func_desc.txt')
        seed_func_path = f'{problem_prompt_path}/seed_func.txt'
        self.seed_func = file_to_string(seed_func_path) if os.path.exists(seed_func_path) else ""
        self.seed_func_1 = file_to_string(f'{problem_prompt_path}/seed_func_1.txt') if os.path.exists(f'{problem_prompt_path}/seed_func_1.txt') else ""
        self.seed_func_2 = file_to_string(f'{problem_prompt_path}/seed_func_2.txt') if os.path.exists(f'{problem_prompt_path}/seed_func_2.txt') else ""
        self.seed_func_3 = file_to_string(f'{problem_prompt_path}/seed_func_3.txt') if os.path.exists(f'{problem_prompt_path}/seed_func_3.txt') else ""
        self.seed_func_4 = file_to_string(f'{problem_prompt_path}/seed_func_4.txt') if os.path.exists(f'{problem_prompt_path}/seed_func_4.txt') else ""
        self.seed_func_5 = file_to_string(f'{problem_prompt_path}/seed_func_5.txt') if os.path.exists(f'{problem_prompt_path}/seed_func_5.txt') else ""
        self.seed_func_6 = file_to_string(f'{problem_prompt_path}/seed_func_6.txt') if os.path.exists(f'{problem_prompt_path}/seed_func_6.txt') else ""
        self.seed_func_7 = file_to_string(f'{problem_prompt_path}/seed_func_7.txt') if os.path.exists(f'{problem_prompt_path}/seed_func_7.txt') else ""

        match = re.match(r'^def +(.+?)\((.*)\) *-> *(.*?) *:', self.func_signature)
        assert match is not None
        self.prompt_func_name = match.group(1)
        self.prompt_func_inputs = [txt.split(":")[0].strip() for txt in match.group(2).split(",")]
        if self.prompt_func_name.startswith('select_next_node'):
            self.prompt_func_outputs = ['next_node']
        elif self.prompt_func_name.startswith('priority'):
            self.prompt_func_outputs = ['priority']
        elif self.prompt_func_name.startswith('heuristics'):
            self.prompt_func_outputs = ['heuristics_matrix']
        elif self.prompt_func_name.startswith('crossover'):
            self.prompt_func_outputs = ['offsprings']
        elif self.prompt_func_name.startswith('utility'):
            self.prompt_func_outputs = ['utility_value']
        elif self.prompt_func_name.startswith('pipeline'):
            self.prompt_func_outputs = ['peak_times', 'peak_heights', 'peak_deltat']
        else:
            self.prompt_func_outputs = ['result']

    def get_task(self):
        return self.cfg.description

    def get_func_name(self):
        return self.prompt_func_name

    def get_func_inputs(self):
        return self.prompt_func_inputs

    def get_func_outputs(self):
        return self.prompt_func_outputs

    def get_inout_inf(self):
        return self.func_desc

    def get_other_inf(self):
        return ""
    
    def get_seed_func(self):
        return self.seed_func
    
    def get_seed_func_1(self):
        return self.seed_func_1
    
    def get_seed_func_2(self):
        return self.seed_func_2
    
    def get_seed_func_3(self):
        return self.seed_func_3
    
    def get_seed_func_4(self):
        return self.seed_func_4
    
    def get_seed_func_5(self):
        return self.seed_func_5
    
    def get_seed_func_6(self):
        return self.seed_func_6
    
    def get_seed_func_7(self):
        return self.seed_func_7
    
    def get_template_i2(self):
        return file_to_string(f'{self.prompt_dir}/common/user_generator_i2.txt')

    def get_template_R(self):
        return file_to_string(f'{self.prompt_dir}/common/user_generator_R.txt')
    
    def get_template_R_1(self):
        return file_to_string(f'{self.prompt_dir}/common/user_generator_R_1.txt')
    
    def get_template_i3(self):
        return file_to_string(f'{self.prompt_dir}/common/user_generator_i3.txt')
    
    def get_template_i4(self):
        return file_to_string(f'{self.prompt_dir}/common/user_generator_i4.txt')

    def get_template_e3(self):
        return file_to_string(f'{self.prompt_dir}/common/user_generator_e3.txt')
    
    def get_template_e3_1(self):
        return file_to_string(f'{self.prompt_dir}/common/user_generator_e3_1.txt')
    
    def get_template_m31(self):
        return file_to_string(f'{self.prompt_dir}/common/user_generator_m31.txt')
    
    def get_template_m31_1(self):
        return file_to_string(f'{self.prompt_dir}/common/user_generator_m31_1.txt')
    
    def get_template_m32(self):
        return file_to_string(f'{self.prompt_dir}/common/user_generator_m32.txt')
    
    def get_template_m32_1(self):
        return file_to_string(f'{self.prompt_dir}/common/user_generator_m32_1.txt')
    
    def get_template_m6(self):
        return file_to_string(f'{self.prompt_dir}/common/user_generator_m6.txt')
    
    def get_template_m6_1(self):
        return file_to_string(f'{self.prompt_dir}/common/user_generator_m6_1.txt')
    
    def get_template_m71(self):
        return file_to_string(f'{self.prompt_dir}/common/user_generator_m71.txt')
    
    def get_template_m71_1(self):
        return file_to_string(f'{self.prompt_dir}/common/user_generator_m71_1.txt')
    
    def get_template_m72(self):
        return file_to_string(f'{self.prompt_dir}/common/user_generator_m72.txt')
    
    def get_template_m72_1(self):
        return file_to_string(f'{self.prompt_dir}/common/user_generator_m72_1.txt')
    
    def get_template_s21(self):
        return file_to_string(f'{self.prompt_dir}/common/user_generator_s21.txt')
    
    def get_template_s21_1(self):
        return file_to_string(f'{self.prompt_dir}/common/user_generator_s21_1.txt')
    
    def get_template_s22(self):
        return file_to_string(f'{self.prompt_dir}/common/user_generator_s22.txt')
    
    def get_template_s22_1(self):
        return file_to_string(f'{self.prompt_dir}/common/user_generator_s22_1.txt')

    def get_template_s31(self):
        return file_to_string(f'{self.prompt_dir}/common/user_generator_s31.txt')
    
    def get_template_s31_1(self):
        return file_to_string(f'{self.prompt_dir}/common/user_generator_s31_1.txt')
    
    def get_external_knowledge(self):
        return file_to_string(f'{self.prompt_dir}/common/user_generator_external_knowledge.txt')
    
class Problem:
    def __init__(self, cfg, root_dir):
        self.config = cfg
        self.root_dir = root_dir

        self.problem = self.config.problem.problem_name
        self.problem_description = self.config.problem.description
        self.problem_size = self.config.problem.get('problem_size', 22.5)  # Default value
        self.obj_type = self.config.problem.obj_type
        import time
        self.runid = int(time.time())
        self.runid = f"{self.config.label}_{self.runid}"
        self.output_file = f"{self.root_dir}/problems/{self.problem}/gpt_{self.runid}.py"

        self.prompts = Prompts(self.config.problem, root_dir)

    def response_to_individual(self, code, response_id, file_name=None, reflection=None) -> dict:
        """
        Convert response to individual
        """
        outdir = './evaluations/'
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        # runid = hash(code)
        import time
        runid = int(time.time())
        # Write response to file
        file_name = outdir + f"problem_eval{runid}.txt" if file_name is None else file_name + ".txt"
        with open(file_name, 'w', encoding='utf-8') as file:
            file.writelines(code + '\n')

        if reflection is not None:
            reflection_file_name = outdir + f"problem_eval{runid}_reflection.txt"
            with open(reflection_file_name, 'w', encoding='utf-8') as file:
                file.writelines(reflection + '\n')
        else:
            reflection_file_name = None

        # Extract code and description from response
        std_out_filepath = outdir + f"problem_eval{runid}_stdout.txt" if file_name is None else file_name.rstrip(
            ".txt") + "_stdout.txt"

        individual = {
            "stdout_filepath": std_out_filepath,
            "code_path": outdir + f"problem_eval{runid}_code.py",
            "code": code,
            "response_id": response_id,
            "reflection_path": reflection_file_name,
            "reflection": reflection,
            "runid": runid
        }
        return individual

    def mark_invalid_individual(self, individual: dict, traceback_msg: str) -> dict:
        """
        Mark an individual as invalid.
        """
        individual["exec_success"] = False
        individual["obj"] = float("inf")
        individual["traceback_msg"] = traceback_msg
        return individual

    def batch_evaluate(self, codes: list[str], eval_times: int, iteration: int, depth: int = None, operator: str = None, reflections: list[str] = None, log_info: str = "") -> str | list[Any]:
        """
        Evaluate population by running code in parallel and computing objective values and fitness.
        """
        self.iteration = iteration
        if reflections is None:
            reflections = [None] * len(codes)
        population = [self.response_to_individual(resp, index, reflection=reflection) for index, (resp, reflection) in enumerate(zip(codes, reflections))]
        inner_runs = []
        # Run code to evaluate
        for response_id in range(len(population)):
            # runid = hash(population[response_id]["code"])
            runid = population[response_id]["runid"]
            # Skip if response is invalid
            if population[response_id]["code"] is None:
                population[response_id] = self.mark_invalid_individual(population[response_id], "Invalid response!")
                inner_runs.append(None)
                continue

            logging.info(f"Eval_times: {eval_times}, Iteration {self.iteration}, Depth {depth}, Operator {operator}: Running Code {runid}")
            logging.info(f"stdout_filepath: {os.path.abspath(population[response_id]['stdout_filepath'])}")
            individual = population[response_id]

            try:
                logging.debug(f"Eval_times: {eval_times}, Iteration {self.iteration}, Depth {depth}, Operator {operator}: Processing Code Run {runid}")

                with open(self.output_file, 'w', encoding = 'utf-8') as file:
                    file.writelines(individual["code"] + '\n')

                # Execute the python file with flags
                with open(individual["stdout_filepath"], 'w') as f:
                    file_path = f'{self.root_dir}/problems/{self.problem}/eval.py'
                    inner_run = process = subprocess.Popen(
                        ['python', '-u', file_path, f'{self.problem_size}', self.root_dir, "train", f"./evaluations/", f"problem_eval{runid}", log_info, f"{self.runid}"], stdout=f, stderr=f)

                block_until_running(individual["stdout_filepath"], log_status=True)
                inner_runs.append(process)
            except Exception as e:  # If code execution fails
                print(e)
                logging.info(f"Error for response_id {response_id} (process): {e}")
                population[response_id] = self.mark_invalid_individual(population[response_id], str(e))
                inner_runs.append(None) 

            if inner_run is None:  # If code execution fails, skip
                continue
            try:
                inner_run.communicate(timeout=self.config.timeout)  # Wait for code execution to finish
            except subprocess.TimeoutExpired as e:
                logging.info(f"Error for response_id {response_id} (communicate): {e}")
                population[response_id] = self.mark_invalid_individual(population[response_id], str(e))
                inner_run.kill()
                return 'timeout', runid

            individual = population[response_id]
            stdout_filepath = individual["stdout_filepath"]
            with open(stdout_filepath, 'r') as f:  # read the stdout file
                stdout_str = f.read()
            traceback_msg = filter_traceback(stdout_str)

            # Store objective value and fitness for each individual
            if traceback_msg == '':  # If execution has no error
                try:
                    individual["obj"] = float(stdout_str.split('\n')[-2])
                    assert individual["obj"] > 0, "Objective value <= 0 is not supported."
                    individual["obj"] = -individual["obj"] if self.obj_type == "max" else individual["obj"]
                    # individual["fitness"] = 1 / individual["obj"] if self.obj_type == "min" else individual["obj"]
                    individual["exec_success"] = True
                    if individual["obj"] == float('inf'): # 打印出来结果异常的报错信息
                        logging.info(f"Error for response_id {response_id} (obj={individual['obj']}): {stdout_str}")
                except Exception as e:
                    print(e)
                    logging.info(f"Error for response_id {response_id} (traceback_msg): {e}")
                    population[response_id] = self.mark_invalid_individual(population[response_id],
                                                                           "Invalid std out / objective value!")
                finally:
                    # Delete the output file if it exists
                    if hasattr(self, 'output_file') and os.path.exists(self.output_file):
                        try:
                            os.remove(self.output_file)
                            logging.debug(f"Deleted output file: {self.output_file}")
                        except Exception as e:
                            logging.warning(f"Failed to delete output file {self.output_file}: {e}")
            else:  # Otherwise, also provide execution traceback error feedback
                population[response_id] = self.mark_invalid_individual(population[response_id], traceback_msg)

            logging.info(f"Eval_times: {eval_times}, Iteration {self.iteration}, Depth {depth}, Operator {operator}, response_id {response_id}: Objective value: {individual['obj']}")

        return [indiv["obj"] for indiv in population], runid

    def cleanup_evaluation_files(self, runid: int) -> None:
        """
        Cleans up temporary files created during evaluation for a specific run ID.
        
        Args:
            runid (int): The hash ID of the run to clean up files for.
        
        This method removes the stdout file and any other temporary files 
        created during the evaluation process for the specified run ID.
        The actual file removal happens with a 10-second delay in the background.
        """
        import glob
        import threading
        import time
        
        n = 60 * 30
        def delayed_removal():
            time.sleep(n)  # 延迟n秒
            try:
                # Define the pattern for files to be cleaned up
                file_pattern = f"problem_eval{runid}*"
                
                # Look for files in the evaluations directory
                eval_dir = "./evaluations/"
                for file_path in glob.glob(f"{eval_dir}{file_pattern}"):
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logging.info(f"Removed evaluation file: {file_path}")
                    
                logging.info(f"Cleanup completed for run ID: {runid}")
            except Exception as e:
                logging.warning(f"Error during cleanup for run ID {runid}: {e}")
        
        # 启动后台线程进行延迟删除
        cleanup_thread = threading.Thread(target=delayed_removal)
        cleanup_thread.daemon = True
        cleanup_thread.start()
        
        # 立即返回，让函数完成操作
        logging.info(f"Scheduled cleanup for run ID: {runid} (will execute in {n} seconds)")
