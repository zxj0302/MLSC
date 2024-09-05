import argparse
import os
import json
import logging
from rwdq import run_query
import dataset_converter as dc

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def sample_oracle(config):
    oracle_config = config['oracle_sampling']
    
    config_path = oracle_config['config']
    database_path = oracle_config['database']
    output_path = oracle_config['output']
    
    run_query(config_path, database_path, output_path)
    
    logger.info(f"Oracle sampling completed. Output saved to {output_path}")
    return output_path

def convert_datasets(input_file, conversion_config):
    ratio = conversion_config['ratio']
    random_shuffle = conversion_config['random_shuffle']
    # TODO: add seed for reproducibility
    
    for format_config in conversion_config['formats']:
        format_type = format_config['format']
        output_file = format_config['output_file']
        
        logger.info(f"Converting dataset to {format_type} format")
        dc.load_and_convert_dataset(
            input_file=input_file,
            output_file=output_file,
            ratio=ratio,
            random_shuffle=random_shuffle,
            format=format_type
        )
        logger.info(f"Conversion to {format_type} completed. Output saved to {output_file}")

def main(config_path):
    config = load_config(config_path)
    
    # Execute the pipeline
    oracle_data = sample_oracle(config)
    
    # Convert and split the data for all specified formats
    convert_datasets(oracle_data, config['data_conversion'])
    
    # TODO: Implement the rest of the pipeline steps
    # ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master script for running experiments")
    parser.add_argument("config", help="Path to the configuration file")
    args = parser.parse_args()
    main(args.config)