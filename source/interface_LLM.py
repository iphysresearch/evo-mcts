import logging

class InterfaceAPI:
    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.client = model_LLM
        self.debug_mode = debug_mode
        self.n_trial = 5

    def get_response(self, system_content=None, prompt_content=None, temp=1., rechat_response=None, 
                     model=None, api_key=None, base_url=None):
        if system_content is None:
            messages = [{"role": "user", "content": prompt_content}]
            if rechat_response is not None:
                logging.info(f"Rechat response!")
                messages.append({"role": "assistant", "content": rechat_response})
                messages.append({"role": "user", "content": "Your previous code had execution errors, couldn't find signals, or timed out. Please debug and fix the issues:\n\n" + prompt_content})
        elif self.client.model in ["deepseek-reasoner"]:
            logging.info(f"Using {self.client.model} (temp={temp}) with system content.")
            messages = [{"role": "user", "content": system_content + "\n\n" + prompt_content}]
            if rechat_response is not None:
                logging.info(f"Rechat response!")
                messages.append({"role": "assistant", "content": rechat_response})
                messages.append({"role": "user", "content": system_content + "\n\n" + "Your previous code had execution errors, couldn't find signals, or timed out. Please debug and fix the issues:\n\n" + prompt_content})
        else:
            messages = [{"role": "system", "content": system_content}, {"role": "user", "content": prompt_content}]
            if rechat_response is not None:
                logging.info(f"Rechat response!")
                messages.append({"role": "assistant", "content": rechat_response})
                messages.append({"role": "user", "content": "Your previous code had execution errors, couldn't find signals, or timed out. Please debug and fix the issues:\n\n" + prompt_content})


        response = self.client.chat_completion(1, messages, temperature=temp, model=model, api_key=api_key, base_url=base_url)
        ret = response[0].message.content
        if model is not None and api_key is not None and base_url is not None:
            return ret, response[0].message.reasoning_content
        else:
            return ret
    
    def multi_get_response(self, system_content=None, user_content=None, n=1, temp=1.):
        if system_content is None:
            messages = [{"role": "user", "content": user_content}]
        elif self.client.model in ["deepseek-reasoner"]:
            logging.info(f"Using {self.client.model} with system content.")
            messages = [{"role": "user", "content": system_content + "\n\n" + user_content}]
        else:
            messages = [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]
        response = self.client.multi_chat_completion([messages], n=n, temperature=temp)
        # logging.info(f"Response: {response}")
        ret = [response[i] for i in range(n)] if n>1 else [response[0]]
        return ret
    