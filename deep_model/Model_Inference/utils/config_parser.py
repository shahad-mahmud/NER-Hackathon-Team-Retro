class ConfigParser:
    def __init__(self,config_dict):
        self.config_dict = config_dict
        for k,v in config_dict.items():
            setattr(self, k, v)        