
import random
import numpy as np
import pandas as pd
from functions import generate_vns_design, discrete_numeric_factor, continuous_factor, categorical_factor

if __name__ == '__main__':
    # Factor inputs
    all_factors = {'A':discrete_numeric_factor(cost_control=True,
                        levels=np.array([-1,0,1]),
                        cost_per_level=[2,8,14],
                        budget=142
                        ),
                    'B':continuous_factor(cost_control=False, 
                        minimum=6, 
                        maximum=36, 
                        step_size=3),
                    'C':continuous_factor(cost_control=False, 
                        minimum=12, 
                        maximum=36, 
                        step_size=3)
                    }

    # Build a dictionary with all following parameters:
    parameters = {}
    parameters['prng'] = random.Random(2999)                # for reproducibility, seed number can be changed
    parameters['all_factors'] = all_factors                 
    parameters['model'] = pd.read_csv('new_model.csv')      # Model definition in csv file, see example file 'new_model.csv' or README for details
    parameters['no_starts'] = 10                            # Number of random starts
    parameters['max_neighborhood'] = 2                      # Maximum neighborhood structure to be used; default neighborhoods are '01', '11', '12', '22'
    parameters['criterion'] = 'D'                           # Optimality criterion; Supported = D & A
    parameters['run_size_limit'] = None                     # Optional limit on the number of runs in the design. If None, no limit is applied.
    parameters['constraints'] = ['B <= C']                  # List of constraint strings. Should be defined in terms of factor names. E.g., 'A + B <= 10', means the sum of factor A and B should be at most 10. Note: for non-numeric factor levels, use single quotes around the level label. E.g., "C != 'level_1'"


    # Generate VNS design using the specified parameters
    generate_vns_design(parameters)