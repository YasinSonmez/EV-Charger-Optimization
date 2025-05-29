#!/usr/bin/env python3
"""
EV Charger Optimization
Main entry point for running optimization with parameters from a JSON configuration file
Now runs all three optimization methods for comparison
"""

import os
import sys
import json
import pickle
import pandas as pd
import time

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.model_fitter import TrafficModelFitter, convert_string_to_array
from src.utils import outer_optimization

# Set up paths for results
os.makedirs("results", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Cache file for fitted model
CACHE_FILE = "data/cached_results.pkl"

def load_or_fit_model(data_path="data/traffic_data.csv"):
    """Load cached model fit results or fit a new model"""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            pandas_df, model_fitter = pickle.load(f)
        print("Loaded cached pandas_df and model_fitter.")
    else:
        print("No cache found. Reading and processing data...")
        pandas_df = pd.read_csv(data_path)
        convert_string_to_array(pandas_df, 'x_vector')
        convert_string_to_array(pandas_df, 'y_vector')

        model_fitter = TrafficModelFitter(pandas_df=pandas_df)
        model_fitter.parallel_fit_and_evaluate()
        model_fitter.fill_missing_link_ids()
        pandas_df = model_fitter.df

        with open(CACHE_FILE, "wb") as f:
            pickle.dump((pandas_df, model_fitter), f)
        print("Cached pandas_df and model_fitter.")

    return pandas_df, model_fitter

def parse_arguments():
    """Parse command line arguments"""
    import argparse
    parser = argparse.ArgumentParser(description='Run EV Charger Optimization')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to the configuration JSON file')
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        sys.exit(1)

def convert_od_demand(od_config):
    """Convert OD demand from config format to dictionary format"""
    od_demand = {}
    for key, value in od_config.items():
        origin, dest = map(int, key.split(','))
        od_demand[(origin, dest)] = tuple(value)
    return od_demand

if __name__ == "__main__":
    # Parse command line arguments to get config file path
    args = parse_arguments()
    
    # Load configuration from JSON file
    config = load_config(args.config)
    
    print("🚗 EV Charger Optimization")
    print("======================")
    print(f"Configuration file: {args.config}")
    print(f"Coordinates: {config['coordinates']}")
    print(f"Number of chargers: {config['num_chargers']}")
    print(f"Possible positions: {config['possible_charger_positions']}")
    print(f"OD demand: {config['od_demand']}")
    
    # Load model
    pandas_df, model_fitter = load_or_fit_model()
    
    # Convert OD demand from config format to dictionary format
    od_demand = convert_od_demand(config['od_demand'])
    
    # Run regular outer optimization (original behavior)
    print(f"\n🔍 Running outer optimization to find best charger placements...")
    outer_optimization(
        coordinates=config['coordinates'],
        num_chargers=config['num_chargers'],
        possible_charger_positions=config['possible_charger_positions'],
        calculate_on_all_possible_positions=config['calculate_on_all_possible_positions'],
        parameter_fit_results=pandas_df,
        max_iter=config['max_iter'],
        use_derivatives=config['use_derivatives'],
        single_swap=config['single_swap'],
        use_cvxpy=config['use_cvxpy'],
        od_demand=od_demand,
        plot_info=config['plot_info'],
        config_filepath=args.config
    )