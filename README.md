# EV Charger Optimization

This project implements an optimization framework for determining the optimal placement of electric vehicle charging stations in a road network. The framework minimizes travel time across the network while accounting for vehicles that need to detour to charging stations.

## Structure

The codebase is organized into the following modules:

- `src/road_network.py`: Contains the `RoadNet` class for downloading and processing road network data from OpenStreetMap
- `src/traffic_optimizer.py`: Contains the main optimization model (`Network` class)
- `src/model_fitter.py`: Contains the `TrafficModelFitter` class for fitting traffic flow models to data
- `src/utils.py`: Utility functions for optimization and visualization
- `main.py`: Main entry point that loads configuration from a JSON file
- `config.json`: Default configuration file with optimization parameters

## Project Layout

```
EV-Charger-Optimization/
│
├── data/                 # Data files and cached results
│
├── results/              # Optimization results and visualizations
│
├── src/                  # Source code
│   ├── __init__.py
│   ├── model_fitter.py
│   ├── road_network.py
│   ├── traffic_optimizer.py
│   └── utils.py
│
├── README.md             # Project documentation
├── requirements.txt      # Dependencies
├── config.json           # Configuration file
└── main.py               # Main execution script
```

## Installation

This project requires several Python dependencies including optimization libraries. The recommended way to set up the environment is using Conda with the provided `environment.yml` file.

1.  **Create and activate Conda environment:**
    ```bash
    # Create a conda environment from the provided file
    conda env create -f environment.yml

    # Activate the environment
    conda activate evopt
    ```

2.  **Verify the installation (optional):**
    You can check if CVXPY and its solvers are correctly installed by running:
    ```bash
    python -c "import cvxpy; print('Available solvers:', cvxpy.installed_solvers())"
    ```

## Usage

You can run the optimizer with the default configuration:

```bash
python main.py
```

Or specify a custom configuration file:

```bash
python main.py --config my_config.json
```

### Configuration File

The configuration is specified in a JSON file with the following structure:

```json
{
    "coordinates": [38.98211, 38.975, -76.93006, -76.93704],
    "num_chargers": 2,
    "possible_charger_positions": [15, 20, 28, 6],
    "od_demand": {
        "0,26": [120, 60]
    },
    "max_iter": 1000,
    "use_derivatives": false,
    "single_swap": true,
    "use_cvxpy": true,
    "plot_info": false,
    "calculate_on_all_possible_positions": true
}
```

#### Configuration Parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `coordinates` | Bounding box coordinates [north, south, east, west] | [38.98211, 38.975, -76.93006, -76.93704] |
| `num_chargers` | Number of chargers to place | 2 |
| `possible_charger_positions` | List of possible charger positions | [15, 20, 28, 6] |
| `od_demand` | OD pairs and demands in format {"origin,dest": [demand1, demand2]} | {"0,26": [120, 60]} |
| `max_iter` | Maximum iterations for optimization | 1000 |
| `use_derivatives` | Use derivatives in optimization | false |
| `single_swap` | Use single swap optimization | true |
| `use_cvxpy` | Use CVXPY solver | true |
| `plot_info` | Plot detailed information | false |
| `calculate_on_all_possible_positions` | Calculate all possible combinations | true |

## API Usage

You can also use the optimizer programmatically:

```python
import sys
import os
import json
# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.model_fitter import TrafficModelFitter, convert_string_to_array
from src.utils import outer_optimization

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Load or fit traffic model parameters
# ... (load model code)

# Convert OD demand format
od_demand = {}
for key, value in config['od_demand'].items():
    origin, dest = map(int, key.split(','))
    od_demand[(origin, dest)] = tuple(value)

# Run optimization to find best charger placements
grids, time_history = outer_optimization(
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
)

# Get best solution
best_grid = grids[np.argmin([grid.travel_time_obj for grid in grids])]
print(f"Best charger configuration: {best_grid.chargers}")
print(f"Best travel time objective: {best_grid.travel_time_obj:.4f}")
```

## Features

- Download road networks from OpenStreetMap
- Fit traffic delay models to data
- Optimize charger placement using:
  - Greedy algorithm
  - Single swap optimization
  - Exhaustive search
- Visualize traffic flows and optimization results
- Detect Braess paradoxes in charger placements

## Requirements

- Python 3.7+
- Dependencies are listed in `environment.yml` (for Conda setup).

## Output Structure (`all_optimization_results.pkl`)

The primary output of an optimization run is a single pickle file named `all_optimization_results.pkl`, located in the run-specific results directory (e.g., `results/YYYY-MM-DD_HH-MM-SS_.../`). This file contains a Python dictionary with the following top-level keys:

-   `'network_link_connectivity'`: A list of dictionaries, where each dictionary describes a link in the road network. Each link's dictionary contains:
    -   `'link_id'`: (integer) The unique identifier for the link.
    -   `'start_node_id'`: (integer) The identifier of the node where the link originates.
    -   `'end_node_id'`: (integer) The identifier of the node where the link terminates.
    This information is stored once per run, as the underlying network is consistent across different charger configurations within that run.

-   `'run_configuration'`: A dictionary holding the input parameters and settings used for the optimization run. This includes details such as:
    -   `coordinates`: The geographical coordinates defining the map area.
    -   `num_chargers`: The number of chargers to place.
    -   `possible_charger_positions`: A list of node IDs where chargers could potentially be placed.
    -   `use_cvxpy`: (boolean) Whether the CVXPY optimization framework was used.
    -   Other relevant parameters from the input `config.json` or `main.py` arguments.

-   `'configurations'`: A dictionary where each key represents a specific charger placement combination evaluated during the run. 
    -   The *key* is node ID representing the charger locations for that specific configuration (e.g., `frozenset({10, 25, 30})`).
    -   The *value* is another dictionary containing the results for that particular charger combination, with the following keys:
        -   `'charger_combination'`: (list) A list of node IDs representing the chargers in this specific configuration.
        -   `'objective_value'`: (float) The calculated objective function value (e.g., total travel time) for this configuration.
        -   `'link_flows'`: (dict) A dictionary where each key is an integer `link_id`.
            -   The value for each `link_id` is another dictionary with:
                -   `'start_node_id'`: (integer) The starting node of the link.
                -   `'end_node_id'`: (integer) The ending node of the link.
                -   `'flow'`: (float) The traffic flow on this link. For charger self-links (where start\_node\_id == end\_node\_id and the node is a charger), this value represents the total demand serviced by that charger in the CVXPY case.
        -   `'method'`: (string) Indicates the optimization method used for this result (e.g., 'cvxpy' or 'scipy').

This structure allows for a comprehensive analysis of the optimization process, providing details about the network, the run parameters, and the performance of each evaluated charger configuration. 
