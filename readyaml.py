import yaml

config_files = ['systemconfig.yml', 'jobconfig.yml']

config = {}
for this_config_file in config_files:
    with open(this_config_file, 'r') as yamlfile:
        this_config = yaml.safe_load(yamlfile)
        config.update(this_config)

print(config)