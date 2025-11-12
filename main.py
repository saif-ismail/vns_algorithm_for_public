
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
    parameters['prng'] = random.Random(2999)# for reproducibility, seed number can be changed
    parameters['all_factors'] = all_factors
    parameters['model'] = pd.read_csv('new_model.csv')
    parameters['no_starts'] = 10
    parameters['max_neighborhood'] = 2
    parameters['criterion'] = 'D'#D,A
    parameters['run_size_limit'] = None#None, or an integer number
    # parameters['linear_constraints'] = list_of_contraints

    generate_vns_design(parameters)

    #TODO:
    # add a requirements file
    # convert design back to uncoded levels
    # implement constraints
    # add I-optimality