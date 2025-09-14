import yaml

def read_config(config_path):
    with open(config_path) as stream:
        config = yaml.safe_load(stream)
    
    print("Successfully read config file!")
    print(config)
    
    return config