class HumanMessage:
    def __init__(self, msg='', token='<|user|>'):
        self.msg = msg
        self.token = token

    def __str__(self):
        return f'Human: {self.msg}'

    def __repr__(self):
        return f'{self.token}\n{self.msg}</s>'


class AIMessage:
    def __init__(self, msg='', token='<|assistant|>'):
        self.msg = msg
        self.token = token

    def __str__(self):
        return f'AI: {self.msg}'

    def __repr__(self):
        return f'{self.token}\n{self.msg}</s>'


class SystemMessage:
    def __init__(self, msg='', token='<|system|>'):
        self.msg = msg
        self.token = token

    def __str__(self):
        return f'System: {self.msg}'

    def __repr__(self):
        return f'{self.token}\n{self.msg}</s>'


class ChatChain:
    def __init__(self, chain):
        self.chain = chain

    def __str__(self):
        return '\n'.join(tuple(map(str, self.chain)))
    
    def __repr__(self):
        return '\n'.join(tuple(map(repr, self.chain)))

    def generate_prompt(self):
        return f'{self.__repr__()}\n<|assistant|>\n'


class ChatModel:
    def __init__(self, client=None, chatChain=None, model='HuggingFaceH4/zephyr-7b-beta'):
        self.client = client
        self.model = model

        self.chatChain = chatChain
        if chatChain is None:
            self.chatChain = ChatChain([SystemMessage('You are a helpful AI that always gives right answers')])

    def invoke(self, msg, stream=False):
        self.chatChain.chain.append(HumanMessage(msg))

        output = ''
        for token in self.client.text_generation(prompt=self.chatChain.generate_prompt(),
                                                 model=self.model,
                                                 max_new_tokens=512,
                                                 stream=stream):
            if token == '</s>':
                break
            output += token

        self.chatChain.chain.append(AIMessage(output))

        return output