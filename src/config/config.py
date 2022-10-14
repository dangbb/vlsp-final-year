class Config:
    def __init__(self, name: str, result_path: str):
        self.name: str = name
        self.source: str = 'raw_text'  # define which source to get sentences from
        self.model_config = {
            'name': 'textrank',
            'params': [8, .1]
        }
        self.eval_config = {
            'name': 'pip_rouge'
        }
        self.reset = True
        self.result_path = result_path
        self.DEBUG = False
