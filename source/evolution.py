import re
import os
import time
from .interface_LLM import InterfaceAPI as InterfaceLLM
import re
import logging

input = lambda: ...


class Evolution():

    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode, prompts, **kwargs):
        assert 'use_local_llm' in kwargs
        assert 'url' in kwargs
        self._use_local_llm = kwargs.get('use_local_llm')
        self._url = kwargs.get('url')
        self.temp = kwargs.get('temperature')
        # -----------------------------------------------------------

        # set prompt interface
        # getprompts = GetPrompts()
        self.prompt_task = prompts.get_task()
        self.prompt_func_name = prompts.get_func_name()
        self.prompt_func_inputs = prompts.get_func_inputs()
        self.prompt_func_outputs = prompts.get_func_outputs()
        self.prompt_inout_inf = prompts.get_inout_inf()
        self.prompt_other_inf = prompts.get_other_inf()
        self.prompt_seed_func = prompts.get_seed_func()
        self.prompt_seed_func_pos = {
            "i2": prompts.get_seed_func(),
            "i2_1": prompts.get_seed_func_1(),
            "i2_2": prompts.get_seed_func_2(),
            "i2_3": prompts.get_seed_func_3(),
            "i2_4": prompts.get_seed_func_4(),
            "i2_5": prompts.get_seed_func_5(),
            "i2_6": prompts.get_seed_func_6(),
            "i2_7": prompts.get_seed_func_7()
        }
        self.prompt_template_i2 = prompts.get_template_i2()
        self.prompt_template_R = prompts.get_template_R()
        self.prompt_template_R_1 = prompts.get_template_R_1()
        self.prompt_template_i3 = prompts.get_template_i3()
        self.prompt_template_i4 = prompts.get_template_i4()
        self.prompt_template_e3 = prompts.get_template_e3()
        self.prompt_template_e3_1 = prompts.get_template_e3_1()
        self.prompt_template_m31 = prompts.get_template_m31()
        self.prompt_template_m31_1 = prompts.get_template_m31_1()
        self.prompt_template_m32 = prompts.get_template_m32()
        self.prompt_template_m32_1 = prompts.get_template_m32_1()
        self.prompt_template_m6 = prompts.get_template_m6()
        self.prompt_template_m6_1 = prompts.get_template_m6_1()
        self.prompt_template_m71 = prompts.get_template_m71()
        self.prompt_template_m71_1 = prompts.get_template_m71_1()
        self.prompt_template_m72 = prompts.get_template_m72()
        self.prompt_template_m72_1 = prompts.get_template_m72_1()
        self.prompt_s21_template = prompts.get_template_s21()
        self.prompt_s21_template_1 = prompts.get_template_s21_1()
        self.prompt_s22_template = prompts.get_template_s22()
        self.prompt_s22_template_1 = prompts.get_template_s22_1()
        self.prompt_s31_template = prompts.get_template_s31()
        self.prompt_s31_template_1 = prompts.get_template_s31_1()

        self.prompt_external_knowledge = prompts.get_external_knowledge()

        if len(self.prompt_func_inputs) > 1:
            self.joined_inputs = ", ".join("'" + s + "'" for s in self.prompt_func_inputs)
        else:
            self.joined_inputs = "'" + self.prompt_func_inputs[0] + "'"

        if len(self.prompt_func_outputs) > 1:
            self.joined_outputs = ", ".join("'" + s + "'" for s in self.prompt_func_outputs)
        else:
            self.joined_outputs = "'" + self.prompt_func_outputs[0] + "'"

        # set LLMs
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode  # close prompt checking

        # -------------------- RZ: use local LLM --------------------
        if self._use_local_llm:
            self.interface_llm = LocalLLM(self._url)
        else:
            self.interface_llm = InterfaceLLM(self.api_endpoint, self.api_key, self.model_LLM, self.debug_mode)

        self.depth_dict = {1: "first", 2: "second", 3: "third",
                           4: "fourth", 5: "fifth", 6: "sixth",
                           7: "seventh", 8: "eighth", 9: "ninth",
                           10: "tenth", 11: "eleventh", 12: "twelfth",
                           13: "thirteenth", 14: "fourteenth", 15: "fifteenth",
                           16: "sixteenth", 17: "seventeenth", 18: "eighteenth",
                           19: "nineteenth", 20: "twentieth"}

        self.max_depth = kwargs.get('max_depth')
        self.response = None
        self.rechat = False
        # self.prompt_post_check = "CRITICAL CHECKS before submitting your answer:\n"\
        #     "1. Make sure your code STARTS with the complete function declaration: def pipeline_v2(strain_h1: np.ndarray, strain_l1: np.ndarray, times: np.ndarray)\n"\
        #     "2. Verify proper indentation throughout your code\n"\
        #     "3. All variables used in the return statement are properly defined\n"\
        #     "4. The final return statement is exactly: return peak_times, peak_heights, peak_deltat\n"\
        #     "5. There are no undefined variables or syntax errors in your code"
        # self.prompt_post_check += "\nAfter writing your code, review it carefully for syntax errors, especially in the final return statement. Make sure you're returning the exact variables (peak_times, peak_heights, peak_deltat) that were requested."
    def get_prompt_post(self, code, algorithm):

        prompt_content = self.prompt_task + "\n" + "Following is the a Code implementing a heuristic algorithm with function name " + self.prompt_func_name + " to solve the above mentioned problem.\n"
        prompt_content += self.prompt_inout_inf + " " + self.prompt_other_inf
        prompt_content += "\n\nCode:\n" + code
        prompt_content += "\n\nNow you should describe the Design Idea of the algorithm using less than 5 sentences.\n"
        prompt_content += "Hint: You should highlight every meaningful designs in the provided code and describe their ideas. You can analyse the code to see which variables are given higher values and which variables are given lower values, the choice of parameters or the total structure of the code."
        return prompt_content

    def get_prompt_refine(self, code, algorithm):
        timestamp_prefix = f"(Timestamp: {time.time()})\n" if self.model_LLM.model.startswith("deepseek") else ""
        system_content = f"{timestamp_prefix}You are an expert in gravitational wave signal detection algorithms. Your task is to analyze and refine descriptions of heuristic algorithms that solve optimization problems. {self.prompt_task}"
        
        prompt_content = "Following is the Design Idea of a heuristic algorithm for the problem and the code with function name '" + self.prompt_func_name + "' for implementing the heuristic algorithm.\n"
        prompt_content += self.prompt_inout_inf + " " + self.prompt_other_inf
        prompt_content += "\nDesign Idea:\n" + algorithm
        prompt_content += "\n\nCode:\n```python\n" + code + "\n```"
        prompt_content += "\n\nThe content of the Design Idea idea cannot fully represent what the algorithm has done informative. So, now you should re-describe the algorithm using less than 3 sentences.\n"
        prompt_content += "Hint: You should reference the given Design Idea and highlight the most critical design ideas of the code. You can analyse the code to describe which variables are given higher priorities and which variables are given lower priorities, the parameters and the structure of the code."
        
        return prompt_content, system_content

    def get_prompt_i1(self):

        prompt_content = self.prompt_task + "\n" + "First, describe the design idea and main steps of your algorithm in one sentence. " + "The description must be inside a brace outside the code implementation. Next, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations."
        return prompt_content

    def get_prompt_i2(self, operator):
        timestamp_prefix = f"(Timestamp: {time.time()})\n" if self.model_LLM.model.startswith("deepseek") else ""
        system_content = f"{timestamp_prefix}You are an expert in gravitational wave signal detection algorithms. Your task is to design heuristics that can effectively solve optimization problems. {self.prompt_task}"
        prompt_content = (
            self.prompt_template_i2.format(
                prompt_seed_func=self.prompt_seed_func_pos[operator].replace("_v1", "_v2"),
                prompt_inout_inf=self.prompt_inout_inf,
                prompt_other_inf=self.prompt_other_inf,
                external_knowledge=self.prompt_external_knowledge,
            )
        )
        return prompt_content, system_content

    def get_prompt_R(self, code_worse, code_better, depth=None):
        timestamp_prefix = f"(Timestamp: {time.time()})\n" if self.model_LLM.model.startswith("deepseek") else ""
        system_content = f"{timestamp_prefix}You are an expert in gravitational wave signal detection algorithms. Your task is to design heuristics that can effectively solve optimization problems. {self.prompt_task}"
        prompt_content = self.prompt_template_R_1.format(
            depth=depth,
            max_depth=self.max_depth,
            prompt_task=self.prompt_task,
            code_worse=code_worse,
            code_better=code_better
        )
        return prompt_content, system_content

    def get_prompt_i3(self):
        prompt_content = self.prompt_template_i3.format(
            seed_func=self.prompt_seed_func.replace("_v1", "_v2"),
            func_name=self.prompt_func_name,
            input_count=len(self.prompt_func_inputs),
            joined_inputs=self.joined_inputs,
            output_count=len(self.prompt_func_outputs),
            joined_outputs=self.joined_outputs,
            inout_inf=self.prompt_inout_inf,
            other_inf=self.prompt_other_inf
        )
        return prompt_content

    def get_prompt_i4(self, depth=None):
        # system_content = "You are an expert in the domain of optimization heuristics. Your task is to design heuristics that can effectively solve optimization problems." # follow the ReEvo system_generator.txt
        timestamp_prefix = f"(Timestamp: {time.time()})\n" if self.model_LLM.model.startswith("deepseek") else ""
        system_content = f"{timestamp_prefix}You are an expert in gravitational wave signal detection algorithms. Your task is to design heuristics that can effectively solve optimization problems. {self.prompt_task}"
        prompt_content = self.prompt_template_i4.format(
            depth=self.depth_dict[depth],
            seed_func=self.prompt_seed_func.replace("_v1", "_v2"),
            func_name=self.prompt_func_name,
            input_count=len(self.prompt_func_inputs),
            joined_inputs=self.joined_inputs,
            output_count=len(self.prompt_func_outputs),
            joined_outputs=self.joined_outputs,
            inout_inf=self.prompt_inout_inf,
            other_inf=self.prompt_other_inf
        )
        return prompt_content, system_content

    def get_prompt_e1(self, indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            # print(indivs[i]['algorithm'] + f"Objective value: {indivs[i]['objective']}")
            prompt_indiv = prompt_indiv + "No." + str(
                i + 1) + " algorithm's description, its corresponding code and its objective value are: \n" + \
                           indivs[i]['algorithm'] + "\n" + indivs[i][
                               'code'] + "\n" + f"Objective value: {indivs[i]['objective']}" + "\n\n"

        prompt_content = self.prompt_task + "\n" \
                                            "I have " + str(
            len(indivs)) + " existing algorithms with their codes as follows: \n\n" \
                         + prompt_indiv + \
                         "Please create a new algorithm that has a totally different form from the given algorithms. Try generating codes with different structures, flows or algorithms. The new algorithm should have a relatively low objective value. \n" \
                         "First, describe the design idea and main steps of your algorithm in one sentence. The description must be inside a brace outside the code implementation. Next, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations."
        return prompt_content

    def get_prompt_e2(self, indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            # print(indivs[i]['algorithm'] + f"Objective value: {indivs[i]['objective']}")
            prompt_indiv = prompt_indiv + "No." + str(
                i + 1) + " algorithm's description, its corresponding code and its objective value are: \n" + \
                           indivs[i]['algorithm'] + "\n" + indivs[i][
                               'code'] + "\n" + f"Objective value: {indivs[i]['objective']}" + "\n\n"

        prompt_content = self.prompt_task + "\n" \
                                            "I have " + str(
            len(indivs)) + " existing algorithms with their codes and objective values as follows: \n\n" \
                         + prompt_indiv + \
                         f"Please create a new algorithm that has a similar form to the No.{len(indivs)} algorithm and is inspired by the No.{1} algorithm. The new algorithm should have a objective value lower than both algorithms.\n" \
                         f"Firstly, list the common ideas in the No.{1} algorithm that may give good performances. Secondly, based on the common idea, describe the design idea based on the No.{len(indivs)} algorithm and main steps of your algorithm in one sentence. \
The description must be inside a brace. Thirdly, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations."
        return prompt_content

    def get_prompt_e3(self, reflection, worse_code, better_code, depth=None):
        timestamp_prefix = f"(Timestamp: {time.time()})\n" if self.model_LLM.model.startswith("deepseek") else ""
        system_content = f"{timestamp_prefix}You are an expert in gravitational wave signal detection algorithms. Your task is to design heuristics that can effectively solve optimization problems. {self.prompt_task}"
        prompt_content = self.prompt_template_e3_1.format(
            depth=depth,
            max_depth=self.max_depth,
            worse_code=worse_code,
            better_code=better_code,
            reflection=reflection,
            func_name=self.prompt_func_name,
            input_count=len(self.prompt_func_inputs),
            joined_inputs=self.joined_inputs,
            output_count=len(self.prompt_func_outputs),
            joined_outputs=self.joined_outputs,
            inout_inf=self.prompt_inout_inf,
            other_inf=self.prompt_other_inf,
            external_knowledge=self.prompt_external_knowledge
        )
        return prompt_content, system_content

    def get_prompt_m1(self, indiv1):
        prompt_content = self.prompt_task + "\n" \
                                            "I have one algorithm with its code as follows. \n\n\
Algorithm's description: " + indiv1['algorithm'] + "\n\
Code:\n\
" + indiv1['code'] + "\n\
Please create a new algorithm that has a different form but can be a modified version of the provided algorithm. Attempt to introduce more novel mechanisms and new equations or programme segments.\n" \
                     "First, describe the design idea based on the provided algorithm and main steps of the new algorithm in one sentence. \
The description must be inside a brace outside the code implementation. Next, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations."
        return prompt_content

    def get_prompt_m2(self, indiv1):
        prompt_content = self.prompt_task + "\n" \
                                            "I have one algorithm with its code as follows. \n\n\
Algorithm's description: " + indiv1['algorithm'] + "\n\
Code:\n\
" + indiv1['code'] + "\n\
Please identify the main algorithm parameters and help me in creating a new algorithm that has different parameter settings to equations compared to the provided algorithm. \n" \
                     "First, describe the design idea based on the provided algorithm and main steps of the new algorithm in one sentence. \
The description must be inside a brace outside the code implementation. Next, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations."
        return prompt_content
    
    def get_prompt_m31(self, parents, father, depth=None):
        # Collect reflections from all parents
        sorted_parents = sorted(parents, key=lambda x: x['objective'])
        parent_reflections = ""
        for i, parent in enumerate(sorted_parents):
            if 'reflection' in parent:
                parent_reflections += f"[No.{i+1} Brother Reflection | Score: {parent['objective']}]\n```text\n{parent['reflection']}\n```\n"
        
        # Add father reflection if available
        father_reflection = ""
        if 'reflection' in father:
            father_reflection = f"[Father Reflection | Score: {father['objective']}]\n```text\n{father['reflection']}\n```\n"
        
        timestamp_prefix = f"(Timestamp: {time.time()})\n" if self.model_LLM.model.startswith("deepseek") else ""
        system_content = f"{timestamp_prefix}You are an expert in gravitational wave signal detection algorithms. Your task is to design heuristics that can effectively solve optimization problems. {self.prompt_task}"

        # Apply the template
        prompt_content = self.prompt_template_m31_1.format(
            max_depth=self.max_depth,
            parent_depth=depth,
            father_depth=depth-1,
            parent_reflections=parent_reflections,
            father_reflection=father_reflection
        )
        
        return prompt_content, system_content

    def get_prompt_m32(self, reflection, father, depth=None):
        prompt_content = self.prompt_template_m32_1.format(
            depth=depth,
            max_depth=self.max_depth,
            reflection=reflection,
            algorithm_description=father['algorithm'],
            algorithm_code=father['code'],
            func_name=self.prompt_func_name,
            input_count=len(self.prompt_func_inputs),
            joined_inputs=self.joined_inputs,
            output_count=len(self.prompt_func_outputs),
            joined_outputs=self.joined_outputs,
            inout_inf=self.prompt_inout_inf,
            other_inf=self.prompt_other_inf,
            external_knowledge=self.prompt_external_knowledge,
        )
        
        return prompt_content

    # def get_prompt_m6(self, indiv, elite_offspring):
    #     # system_content = "You are an expert in the domain of optimization heuristics. Your task is to design heuristics that can effectively solve optimization problems." # follow the ReEvo system_generator.txt
    #     prompt_content = self.prompt_template_m6.format(
    #         prompt_task=self.prompt_task,
    #         original_algorithm_description=indiv['algorithm'],
    #         original_algorithm_code=indiv['code'],
    #         original_objective_value=indiv['objective'],
    #         better_algorithm_description=elite_offspring['algorithm'],
    #         better_algorithm_code=elite_offspring['code'],
    #         better_objective_value=elite_offspring['objective'],
    #         better_algorithm_reflection=elite_offspring['reflection'],
    #         func_name=self.prompt_func_name,
    #         input_count=len(self.prompt_func_inputs),
    #         joined_inputs=self.joined_inputs,
    #         output_count=len(self.prompt_func_outputs),
    #         joined_outputs=self.joined_outputs,
    #         inout_inf=self.prompt_inout_inf,
    #         other_inf=self.prompt_other_inf
    #     )
        
    #     return prompt_content
    
    def get_prompt_m6_1(self, indiv, elite_offspring, depth=None):
        timestamp_prefix = f"(Timestamp: {time.time()})\n" if self.model_LLM.model.startswith("deepseek") else ""
        system_content = f"{timestamp_prefix}You are an expert in gravitational wave signal detection algorithms. Your task is to design heuristics that can effectively solve optimization problems. {self.prompt_task}"

        # system_content = "You are an expert in gravitational wave signal detection algorithms. Your task is to analyze algorithms for processing dual-channel gravitational wave data from H1 and L1 detectors, and create improved versions. Focus on enhancing data conditioning, time-frequency transformations, and signal identification techniques. Your goal is to develop algorithms that more accurately identify potential gravitational wave signals with precise GPS time, ranking statistics, and timing accuracy. Provide thoughtful analysis of algorithm differences and implement improvements that address limitations in previous versions. Ensure your solutions strictly follow the specified function signature and formatting requirements."

        prompt_content = self.prompt_template_m6_1.format(
            original_algorithm_description=indiv['algorithm'],
            original_algorithm_code=indiv['code'],
            original_objective_value=indiv['objective'],
            better_algorithm_description=elite_offspring['algorithm'],
            better_algorithm_code=elite_offspring['code'],
            better_objective_value=elite_offspring['objective'],
            better_algorithm_reflection=elite_offspring['reflection'],
            func_name=self.prompt_func_name,
            input_count=len(self.prompt_func_inputs),
            joined_inputs=self.joined_inputs,
            output_count=len(self.prompt_func_outputs),
            joined_outputs=self.joined_outputs,
            inout_inf=self.prompt_inout_inf,
            other_inf=self.prompt_other_inf,
            external_knowledge=self.prompt_external_knowledge,
        )
        
        return prompt_content, system_content

    def get_prompt_m71_1(self, parents, elite_offspring):
        timestamp_prefix = f"(Timestamp: {time.time()})\n" if self.model_LLM.model.startswith("deepseek") else ""
        system_content = f"{timestamp_prefix}You are an expert in gravitational wave signal detection algorithms. Your task is to design heuristics that can effectively solve optimization problems. {self.prompt_task}"

        # Collect reflections from all parents
        # Sort parents by objective (newer=lower) and include scores
        sorted_parents = sorted(parents, key=lambda x: x['objective'])
        parent_reflections = ""
        for i, parent in enumerate(sorted_parents):
            if 'reflection' in parent:
                parent_reflections += f"[Parent {i+1} Reflection | Score: {parent['objective']}]\n```text\n{parent['reflection']}\n```\n"
        
        # Add elite offspring reflection with score if available
        elite_reflection = ""
        if 'reflection' in elite_offspring:
            elite_reflection = f"[Elite Offspring Reflection | Score: {elite_offspring['objective']}]\n```text\n{elite_offspring['reflection']}\n```\n"
        
        # Apply the template
        prompt_content = self.prompt_template_m71_1.format(
            parent_reflections=parent_reflections,
            elite_reflection=elite_reflection
        )
        
        return prompt_content, system_content

    def get_prompt_m72_1(self, reflection, elite_offspring):
        timestamp_prefix = f"(Timestamp: {time.time()})\n" if self.model_LLM.model.startswith("deepseek") else ""
        system_content = f"{timestamp_prefix}You are an expert in gravitational wave signal detection algorithms. Your task is to design heuristics that can effectively solve optimization problems. {self.prompt_task}"

        prompt_content = self.prompt_template_m72_1.format(
            reflection=reflection,
            elite_algorithm_description=elite_offspring['algorithm'],
            elite_algorithm_code=elite_offspring['code'],
            func_name=self.prompt_func_name,
            input_count=len(self.prompt_func_inputs),
            joined_inputs=self.joined_inputs,
            output_count=len(self.prompt_func_outputs),
            joined_outputs=self.joined_outputs,
            inout_inf=self.prompt_inout_inf,
            other_inf=self.prompt_other_inf,
            external_knowledge=self.prompt_external_knowledge,
        )
        
        return prompt_content, system_content

    def get_prompt_s1(self, indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv = prompt_indiv + "No." + str(
                i + 1) + " algorithm's description, its corresponding code and its objective value are: \n" + \
                           indivs[i]['algorithm'] + "\n" + indivs[i][
                               'code'] + "\n" + f"Objective value: {indivs[i]['objective']}" + "\n\n"

        prompt_content = self.prompt_task + "\n" \
                                            "I have " + str(
            len(indivs)) + " existing algorithms with their codes and objective values as follows: \n\n" \
                         + prompt_indiv + \
                         f"Please help me create a new algorithm that is inspired by all the above algorithms with its objective value lower than any of them.\n" \
                         "Firstly, list some ideas in the provided algorithms that are clearly helpful to a better algorithm. Secondly, based on the listed ideas, describe the design idea and main steps of your new algorithm in one sentence. \
The description must be inside a brace. Thirdly, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations."
        return prompt_content

    def get_prompt_s21(self, indivs, depth=None):
        # Compile reflections from all individuals
        algorithm_reflections = ""
        
        # Since indivs represents a route from deep to shallow,
        # we need to track the current depth for each algorithm
        current_depth = len(indivs)  # Start from deepest level
        
        for i in range(len(indivs)):
            if 'reflection' in indivs[i]:
                # Add depth information to the reflection
                algorithm_reflections += (
                    f"[No.{i + 1} algorithm's reflection (depth: {current_depth})]\n"
                    f"```text\n"
                    f"{indivs[i]['reflection']}\n"
                    f"```\n\n"
                )
            
            current_depth -= 1  # Move to shallower depth
        
        timestamp_prefix = f"(Timestamp: {time.time()})\n" if self.model_LLM.model.startswith("deepseek") else ""
        system_content = f"{timestamp_prefix}You are an expert in gravitational wave signal detection algorithms. Your task is to design heuristics that can effectively solve optimization problems. {self.prompt_task}"
        # Apply the template
        prompt_content = self.prompt_s21_template_1.format(
            depth=depth,
            max_depth=self.max_depth,
            num_algorithms=len(indivs),
            algorithm_reflections=algorithm_reflections
        )
        
        return prompt_content, system_content

    def get_prompt_s22(self, reflection, father):
        timestamp_prefix = f"(Timestamp: {time.time()})\n" if self.model_LLM.model.startswith("deepseek") else ""
        system_content = f"{timestamp_prefix}You are an expert in gravitational wave signal detection algorithms. Your task is to design heuristics that can effectively solve optimization problems. {self.prompt_task}"
        prompt_content = self.prompt_s22_template_1.format(
            reflection=reflection,
            algorithm_description=father['algorithm'],
            algorithm_code=father['code'],
            func_name=self.prompt_func_name,
            input_count=len(self.prompt_func_inputs),
            joined_inputs=self.joined_inputs,
            output_count=len(self.prompt_func_outputs),
            joined_outputs=self.joined_outputs,
            inout_inf=self.prompt_inout_inf,
            other_inf=self.prompt_other_inf,
            external_knowledge=self.prompt_external_knowledge,
        )
        
        return prompt_content, system_content

    def get_prompt_s31(self, parents, father, depth=None):

        # Collect information from all parents with depth tracking
        parent_info = ""
        
        # Since parents represents a route from deep to shallow,
        # we need to track the current depth for each parent
        current_depth = len(parents)  # Start from deepest level
        
        for i, parent in enumerate(parents):
            parent_info += (
                f"[No.{i + 1} algorithm's reflection (depth: {current_depth})]\n"
                f"Description: {parent['algorithm']}\n"
                f"Current Performance: {parent['objective']}\n"
                f"Current Code:\n"
                f"```python\n"
                f"{parent['code']}\n"
                f"```\n\n"
            )
            
            current_depth -= 1  # Move to shallower depth
        timestamp_prefix = f"(Timestamp: {time.time()})\n" if self.model_LLM.model.startswith("deepseek") else ""
        system_content = f"{timestamp_prefix}You are an expert in gravitational wave signal detection algorithms. Your task is to design heuristics that can effectively solve optimization problems. {self.prompt_task}"
        # Apply the template
        prompt_content = self.prompt_s31_template_1.format(
            depth=depth,
            max_depth=self.max_depth,
            parent_info=parent_info,
            num_algorithms=len(parents),
            current_algorithm_description=father['algorithm'],
            current_algorithm_code=father['code'],
            current_objective_value=father['objective']
        )
        
        return prompt_content, system_content

    def _get_thought(self, prompt_content, system_content=None):

        response = self.interface_llm.get_response(system_content=system_content, prompt_content=prompt_content, temp=self.temp)

        # algorithm = response.split(':')[-1]
        return response

    def _get_alg(self, prompt_content, system_content=None, require_code=True, require_algorithm=True, n=1, deepseek_thinking=False, force_no_rechat=False):
        valid_results = []
        thinking = None
        if n > 1:
            responses = self.interface_llm.multi_get_response(system_content=system_content, user_content=prompt_content, n=n, temp=self.temp)
        else:
            if deepseek_thinking:
                logging.info(f"Using deepseek-reasoner thinking.")
                # Add random timestamp to avoid triggering context disk cache
                unique_prompt = f"[Request ID: {time.time()}]\n\n{system_content}"
                responses = [self.interface_llm.get_response(system_content=unique_prompt, prompt_content=prompt_content, temp=self.temp,
                                                         rechat_response=self.response if self.rechat and self.response is not None and not force_no_rechat else None,
                                                         model=os.environ.get('DEEPSEEK_MODEL', 'deepseek-r1-250120'),
                                                         api_key=os.environ.get('DEEPSEEK_API_KEY', 'your-deepseek-api-key'),
                                                         base_url=os.environ.get('DEEPSEEK_BASE_URL', 'https://'),
                                                         )]
                thinking = responses[0][1]
                responses = [responses[0][0]]
            else:
                responses = [self.interface_llm.get_response(system_content=system_content, prompt_content=prompt_content, temp=self.temp,
                                                         rechat_response=self.response if self.rechat and self.response is not None else None)]
        
        for response in responses:
            if self.debug_mode:
                logging.info(f"Response: \n{response}")
            
            algorithm = re.search(r"\{(.*?)\}", response, re.DOTALL).group(1) if require_algorithm else ""
            if len(algorithm) == 0:
                if 'python' in response:
                    algorithm = re.findall(r'^.*?(?=python)', response, re.DOTALL)
                elif 'import' in response:
                    algorithm = re.findall(r'^.*?(?=import)', response, re.DOTALL)
                else:
                    algorithm = re.findall(r'^.*?(?=def)', response, re.DOTALL)

            code = re.findall(r"import.*return", response, re.DOTALL)
            if require_code and len(code) == 0:
                code = re.findall(r"def.*return", response, re.DOTALL)
                
            if (not require_algorithm or len(algorithm) > 0) and (not require_code or len(code) > 0):
                code_str = code[0] if code else ""
                code_all = code_str + " " + ", ".join(s for s in self.prompt_func_outputs)
                valid_results.append([code_all, algorithm])
        
        n_retry = 1
        while len(valid_results) < n:
            if self.debug_mode:
                logging.info(f"Error: only got {len(valid_results)}/{n} valid results, retrying...")

            response = self.interface_llm.get_response(system_content=system_content, prompt_content=prompt_content, temp=self.temp,
                                                        rechat_response=self.response if self.rechat and self.response is not None else None)

            algorithm = re.search(r"\{(.*?)\}", response, re.DOTALL).group(1) if require_algorithm else ""
            if len(algorithm) == 0:
                if 'python' in response:
                    algorithm = re.findall(r'^.*?(?=python)', response, re.DOTALL)
                elif 'import' in response:
                    algorithm = re.findall(r'^.*?(?=import)', response, re.DOTALL)
                else:
                    algorithm = re.findall(r'^.*?(?=def)', response, re.DOTALL)

            code = re.findall(r"import.*?return", response, re.DOTALL)
            if (require_code and len(code) == 0) or ('def pipeline_v2' not in response):
                code = re.findall(r"def.*?return", response, re.DOTALL)

            if (not require_algorithm or len(algorithm) > 0) and (not require_code or len(code) > 0):
                code_str = code[0] if code else ""
                code_all = code_str + " " + ", ".join(s for s in self.prompt_func_outputs)
                valid_results.append([code_all, algorithm])

            if n_retry > n * 2:
                # Fill remaining slots with the last valid result if we have at least one
                if valid_results:
                    while len(valid_results) < n:
                        valid_results.append(valid_results[-1])
                break
            n_retry += 1

        if not force_no_rechat:
            self.response = response  # DEBUG: only works for n=1
        # print(f"Valid results: \n{valid_results}")``
        # Return a single result or a list of results based on n
        if deepseek_thinking and thinking is not None:
            return thinking, valid_results[0] if n == 1 else valid_results
        else:
            return valid_results[0] if n == 1 else valid_results

    def post_thought(self, offspring):
        prompt_content, system_content = self.get_prompt_refine(offspring['code'], offspring['thought'])
        offspring['algorithm'] = self._get_thought(prompt_content, system_content=system_content)
        offspring['post_user_content'] = prompt_content
        offspring['post_system_content'] = system_content

    def i1(self):
        logging.info("Executing initialization method i1...")
        prompt_content = self.get_prompt_i1()

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ i1 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def I2(self):
        """
        This function is used to create a new algorithm based on the provided seed function.
        """
        logging.info("Executing initialization method 【I2】...")
        prompt_content = self.get_prompt_I2()

        if self.debug_mode:
            print("\n >>> check prompt for creating idea description using [ I2 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        _ , algorithm = self._get_alg(prompt_content, require_code=False)
        code_all = self.prompt_seed_func.replace("_v1", "_v2")
        
        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed seed function code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def R(self, code_worse, code_better, depth=None):
        """
        This function is used to create a new algorithm based on the provided seed function.
        """
        logging.info("Executing initialization method R...")
        prompt_content, system_content = self.get_prompt_R(code_worse, code_better, depth=depth)

        if self.debug_mode:
            print("\n >>> check system prompt for creating idea description using [ R ] : \n", system_content)
            print("\n >>> check user prompt for creating idea description using [ R ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        thinking, (_, reflection) = self._get_alg(prompt_content, system_content=system_content, require_code=False,
                                                  deepseek_thinking=True, force_no_rechat=True)

        if self.debug_mode:
            print("\n >>> check designed reflection: \n", reflection)
            print(">>> Press 'Enter' to continue")
            input()

        return reflection, system_content, prompt_content, thinking

    def i2(self, offspring):
        """
        This function is used to create a new algorithm based on the provided seed function.
        """
        operator = offspring['operator']
        logging.info(f"Executing initialization method i2 with operator: {operator}...")
        prompt_content, system_content = self.get_prompt_i2(operator)

        if self.debug_mode:
            print(f"\n >>> check system prompt for creating idea description using [ {operator} ] : \n", system_content)
            print(f"\n >>> check user prompt for creating idea description using [ {operator} ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        thinking, (_, algorithm) = self._get_alg(prompt_content, system_content=system_content, require_code=False, deepseek_thinking=True)
        code_all = self.prompt_seed_func_pos[operator].replace("_v1", "_v2")
        
        if self.debug_mode:
            print(f"\n >>> check designed algorithm for [ {operator} ] : \n", algorithm)
            print(f"\n >>> check designed seed function code for [ {operator} ] : \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()
        offspring['thinking'] = thinking
        offspring['thought'] = algorithm
        offspring['code'] = code_all
        offspring['other_inf'] = 'seed'
        offspring['reflection'] = algorithm
        offspring['user_content'] = prompt_content
        offspring['system_content'] = system_content


    def i3(self):
        """
        This function is used to create a new algorithm based on the provided seed function.
        """
        logging.info("Executing initialization method i3...")
        prompt_content = self.get_prompt_i3()

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ i3 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def i4(self, depth=None):
        """
        This function is used to create a new algorithm based on the provided seed function.
        """
        logging.info("Executing initialization method i4...")
        prompt_content, system_content = self.get_prompt_i4(depth=depth)

        if self.debug_mode:
            print("\n >>> check system prompt for creating algorithm using [ i4 ] : \n", system_content)
            print("\n >>> check user prompt for creating algorithm using [ i4 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content, system_content=system_content, n=1)

        if self.debug_mode:
            print("\n >>> check designed algorithm from [ i4 ] : \n", algorithm)
            print("\n >>> check designed code from [ i4 ] : \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]


    def e1(self, parents):
        logging.info("Executing initialization method e1...")
        prompt_content = self.get_prompt_e1(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e1 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def e2(self, parents):
        logging.info("Executing initialization method e2...")
        prompt_content = self.get_prompt_e2(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e2 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def e3(self, offspring, parent, father, depth=None):

        logging.info("Executing initialization method e3...")
        if father['objective'] < parent['objective']:
            worse_code = parent['code']
            better_code = father['code']
        else:
            worse_code = father['code']
            better_code = parent['code']

        prompt_content, system_content = self.get_prompt_R(worse_code, better_code, depth=depth)
        offspring['R_user_content'] = prompt_content
        offspring['R_system_content'] = system_content

        if self.debug_mode:
            print("\n >>> check system prompt for creating reflection using [ R (e3) ] : \n", system_content)
            print("\n >>> check user prompt for creating reflection using [ R (e3) ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        thinking, (_, reflection) = self._get_alg(prompt_content, system_content=system_content, require_code=False,
                                                  deepseek_thinking=True, force_no_rechat=True)
        offspring['R_thinking'] = thinking
        offspring['reflection'] = reflection

        if self.debug_mode:
            print("\n >>> check designed reflection for [ R (e3) ] : \n", reflection)
            print(">>> Press 'Enter' to continue")
            input()

        prompt_content, system_content = self.get_prompt_e3(reflection, worse_code, better_code, depth=depth)
        offspring['user_content'] = prompt_content
        offspring['system_content'] = system_content

        if self.debug_mode:
            print("\n >>> check system prompt for creating improved code using [ e3 ] : \n", system_content)
            print("\n >>> check user prompt for creating improved code using [ e3 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content, system_content=system_content, n=1)

        if self.debug_mode:
            print("\n >>> check designed algorithm from [ e3 ] : \n", algorithm)
            print("\n >>> check designed code from [ e3 ] : \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        offspring['code'] = code_all
        offspring['thought'] = algorithm

    def m1(self, parents):
        logging.info("Executing initialization method m1...")
        prompt_content = self.get_prompt_m1(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m1 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()
    
        return [code_all, algorithm]

    def m2(self, parents):
        logging.info("Executing initialization method m2...")
        prompt_content = self.get_prompt_m2(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m2 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def m3(self, offspring, parents, father, depth=None):
        logging.info("Executing initialization method m3...")
        prompt_content, system_content = self.get_prompt_m31(parents, father, depth=depth)

        if self.debug_mode:
            print("\n >>> check system prompt for creating algorithm using [ m3 ] : \n", system_content)
            print("\n >>> check user prompt for creating algorithm using [ m3 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        thinking, (_, reflection) = self._get_alg(prompt_content, system_content=system_content, require_code=False,
                                                  deepseek_thinking=True, force_no_rechat=True)
        offspring['thinking'] = thinking
        offspring['reflection'] = reflection
        offspring['user_content_reflection'] = prompt_content
        offspring['system_content_reflection'] = system_content

        if self.debug_mode:
            print("\n >>> check designed reflection from [ m3 ] : \n", reflection)
            print(">>> Press 'Enter' to continue")
            input()
            
        prompt_content = self.get_prompt_m32(reflection, father, depth=depth)
        offspring['user_content'] = prompt_content
        offspring['system_content'] = system_content

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m3 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content, system_content=system_content, n=1)
        offspring['code'] = code_all
        offspring['thought'] = algorithm

        if self.debug_mode:
            print("\n >>> check designed algorithm from [ m3 ] : \n", algorithm)
            print("\n >>> check designed code from [ m3 ] : \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()
    
    # def m6(self, parent, elite_offspring):
    #     logging.info("Executing initialization method m6...")
    #     prompt_content = self.get_prompt_m6(parent, elite_offspring)

    #     if self.debug_mode:
    #         print("\n >>> check prompt for creating algorithm using [ m6 ] : \n", prompt_content)
    #         print(">>> Press 'Enter' to continue")
    #         input()

    #     [code_all, algorithm] = self._get_alg(prompt_content)

    #     if self.debug_mode:
    #         print("\n >>> check designed algorithm: \n", algorithm)
    #         print("\n >>> check designed code: \n", code_all)
    #         print(">>> Press 'Enter' to continue")
    #         input()

    #     return [code_all, algorithm]

    def m6(self, offspring, parent, elite_offspring, depth=None):
        logging.info("Executing initialization method m6_1...")
        prompt_content, system_content = self.get_prompt_m6_1(parent, elite_offspring, depth=depth)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m6_1 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content, system_content=system_content, n=1)

        if self.debug_mode:
            print("\n >>> check designed algorithm from [ m6_1 ] : \n", algorithm)
            print("\n >>> check designed code from [ m6_1 ] : \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        offspring['code'] = code_all
        offspring['thought'] = algorithm
        offspring['reflection'] = algorithm
        offspring['system_content'] = system_content
        offspring['user_content'] = prompt_content
        
    def m7(self, offspring, parents, elite_offspring, depth=None):
        logging.info("Executing initialization method m7...")
        prompt_content, system_content = self.get_prompt_m71_1(parents, elite_offspring)

        if self.debug_mode:
            print("\n >>> check system prompt for creating algorithm using [ m71_1 ] : \n", system_content)
            print("\n >>> check user prompt for creating algorithm using [ m71_1 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()
        thinking, (_, reflection) = self._get_alg(prompt_content, system_content=system_content, require_code=False,
                                                  deepseek_thinking=True, force_no_rechat=True)

        if self.debug_mode:
            print("\n >>> check designed reflection from [ m71_1 ] : \n", reflection)
            print(">>> Press 'Enter' to continue")
            input()

        offspring['system_content_reflection'] = system_content
        offspring['user_content_reflection'] = prompt_content
        offspring['thinking'] = thinking

        prompt_content, system_content = self.get_prompt_m72_1(reflection, elite_offspring)

        if self.debug_mode:
            print("\n >>> check system prompt for creating algorithm using [ m72_1 ] : \n", system_content)
            print("\n >>> check user prompt for creating algorithm using [ m72_1 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()
        [code_all, algorithm] = self._get_alg(prompt_content, system_content=system_content, n=1)

        if self.debug_mode:
            print("\n >>> check designed algorithm from [ m72_1 ] : \n", algorithm)
            print("\n >>> check designed code from [ m72_1 ] : \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        offspring['code'] = code_all
        offspring['thought'] = algorithm
        offspring['reflection'] = reflection
        offspring['system_content'] = system_content
        offspring['user_content'] = prompt_content

    def s1(self, parents):
        logging.info("Executing initialization method s1...")
        prompt_content = self.get_prompt_s1(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ s1 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def s2(self, offspring, parents, father, depth=None):
        logging.info("Executing initialization method s2...")
        prompt_content, system_content = self.get_prompt_s21(parents, depth=depth)

        if self.debug_mode:
            print("\n >>> check prompt for creating reflection using [ s2 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        thinking, (_, reflection) = self._get_alg(prompt_content, require_code=False, system_content=system_content, 
                                        deepseek_thinking=True, force_no_rechat=True)
        offspring['reflection'] = reflection
        offspring['thinking'] = thinking
        offspring['user_content_reflection'] = prompt_content
        offspring['system_content_reflection'] = system_content

        if self.debug_mode:
            print("\n >>> check designed reflection from [ s2 ] : \n", reflection)
            print(">>> Press 'Enter' to continue")
            input()
        
        prompt_content, system_content = self.get_prompt_s22(reflection, father)
        offspring['user_content'] = prompt_content
        offspring['system_content'] = system_content

        if self.debug_mode:
            print("\n >>> check system prompt for creating algorithm using [ s2 ] : \n", system_content)
            print("\n >>> check user prompt for creating algorithm using [ s2 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content, system_content=system_content)
        offspring['code'] = code_all
        offspring['thought'] = algorithm

        if self.debug_mode:
            print("\n >>> check designed algorithm from [ s2 ] : \n", algorithm)
            print("\n >>> check designed code from [ s2 ] : \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()
    
    def s3(self, offspring, parents, father, depth=None):
        logging.info("Executing initialization method s3...")
        prompt_content, system_content = self.get_prompt_s31(parents, father, depth=depth)

        if self.debug_mode:
            print("\n >>> check prompt for creating reflection using [ s3 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        thinking, (_, reflection) = self._get_alg(prompt_content, require_code=False, system_content=system_content, 
                                        deepseek_thinking=True, force_no_rechat=True)
        offspring['reflection'] = reflection
        offspring['thinking'] = thinking
        offspring['user_content_reflection'] = prompt_content
        offspring['system_content_reflection'] = system_content

        if self.debug_mode:
            print("\n >>> check designed reflection from [ s3 ] : \n", reflection)
            print(">>> Press 'Enter' to continue")
            input()

        prompt_content, system_content = self.get_prompt_s22(reflection, father)
        offspring['user_content'] = prompt_content
        offspring['system_content'] = system_content

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ s3 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()
        
        [code_all, algorithm] = self._get_alg(prompt_content, system_content=system_content)
        offspring['code'] = code_all
        offspring['thought'] = algorithm

        if self.debug_mode:
            print("\n >>> check designed algorithm from [ s3 ] : \n", algorithm)
            print("\n >>> check designed code from [ s3 ] : \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()