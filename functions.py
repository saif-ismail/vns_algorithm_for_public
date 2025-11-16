import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class factor:
    cost_control: bool

@dataclass
class continuous_factor(factor):
    minimum: float
    maximum: float
    step_size: float
    budget: Optional[float] = field(default=None, repr=False)
    min_cost: Optional[float] = field(default=None, repr=False)
    step_cost: Optional[float] = field(default=None, repr=False)
    def __post_init__(self):
        self.factor_type = 'continuous'
        self.uncoded_levels = np.arange(start=self.minimum, stop=self.maximum + self.step_size, step=self.step_size)
        self.coded_levels = (2*((self.uncoded_levels-self.minimum)/(self.maximum-self.minimum)))-1
        if self.cost_control:
            missing = [name for name in ['budget', 'min_cost', 'step_cost'] if getattr(self, name) is None]
            assert not missing, f"Missing required arguments: {', '.join(missing)}"
            self.cost_per_level = [self.min_cost + (itr*self.step_cost) for itr in range(len(self.coded_levels))]

@dataclass
class discrete_numeric_factor(factor):
    levels: np.ndarray
    cost_per_level: Optional[list] = field(default=None, repr=False)
    budget: Optional[float] = field(default=None, repr=False)
    def __post_init__(self):
        self.uncoded_levels = np.array(self.levels)
        self.coded_levels = (2*((self.uncoded_levels-self.uncoded_levels.min())/(self.uncoded_levels.max()-self.uncoded_levels.min())))-1
        self.factor_type = 'discrete'
        if self.cost_control:
            missing = [name for name in ['cost_per_level', 'budget'] if getattr(self, name) is None]
            assert not missing, f"Missing required arguments: {', '.join(missing)}"

@dataclass
class categorical_factor(factor):
    labels: list[str]
    cost_per_level: Optional[list] = field(default=None, repr=False)
    budget: Optional[float] = field(default=None, repr=False)
    def __post_init__(self):
        self.factor_type = 'categorical'
        self.uncoded_levels = np.array(self.labels)
        self.coded_levels = np.eye(len(self.labels))[:,1:]
        self.coded_levels[0,:] = -1
        self.coded_levels = [row for row in self.coded_levels]
        if self.cost_control:
            missing = [name for name in ['cost_per_level', 'budget'] if getattr(self, name) is None]
            assert not missing, f"Missing required arguments: {', '.join(missing)}"

def get_locations_for_all_factors(dict_of_factors:dict):
    locations = {}
    start_ix = 0
    for factor_name, factor_instance in dict_of_factors.items():
        if factor_instance.factor_type == 'categorical':
            no_cols = len(factor_instance.coded_levels)-1
        else:
            no_cols = 1
        stop_ix = start_ix + no_cols
        locations[factor_name] = list(range(start_ix, stop_ix))
        start_ix += no_cols

    return locations

def expand_combinations(levels_dict, flatten_arrays=True):
    """Generate all combinations of dictionary values (like itertools.product),
    optionally flattening any numpy arrays or lists inside."""
    combos = itertools.product(*levels_dict.values())
    result = []
    
    for combo in combos:
        if flatten_arrays:
            flat = []
            for x in combo:
                # Flatten numpy arrays or lists
                if isinstance(x, np.ndarray):
                    flat.extend([xi.item() if isinstance(xi, np.generic) else xi for xi in x])
                elif isinstance(x, (list, tuple)) and all(isinstance(i, (int, float, np.generic)) for i in x):
                    flat.extend([i.item() if isinstance(i, np.generic) else i for i in x])
                elif isinstance(x, np.generic):
                    flat.append(x.item())
                else:
                    flat.append(x)
            result.append(tuple(flat))
        else:
            # Keep arrays as-is, but convert np.generic scalars to Python types
            result.append(tuple(x.item() if isinstance(x, np.generic) else x for x in combo))
    
    return pd.DataFrame(result, columns=levels_dict.keys())

def filter_constraints(all_combs:pd.DataFrame, constraints:list):
    '''
    THIS FUNCTION FILTERS THE DATAFRAME BASED ON THE PROVIDED CONSTRAINTS
    '''    
    if not constraints:
        return all_combs

    query_str = ' & '.join(constraints)
    candidate_set = all_combs.query(query_str)
    return candidate_set

def generate_candidate_set(params:dict):
    '''
    THIS FUNCTION GENERATES THE CANDIDATE SET OF COMBINATIONS TAKING INTO ACCOUNT
    LINEAR CONSTRAINTS FOR CODED VARIABLES 
    '''    

    # get uncoded levels for every factor
    levels_dict = {factor_name:factor_instance.uncoded_levels for factor_name, factor_instance in params['all_factors'].items()}

    # Generate all combinations of uncoded levels
    all_combs = expand_combinations(levels_dict)

    # Filter out combinations that do not satisfy constraints
    filtered_combs = filter_constraints(all_combs, params.get('constraints', [])).copy()

    # Convert from uncoded to coded levels
    column_arrays = []
    for factor_name, factor_instance in params['all_factors'].items():
        mapping = dict(zip(factor_instance.uncoded_levels, factor_instance.coded_levels))
        if factor_instance.factor_type == 'categorical':
            mapped_cols = np.array(filtered_combs[factor_name].map(mapping).to_list())
            column_arrays.append(mapped_cols)
            
        else:
            mapped_col = filtered_combs[factor_name].map(mapping).to_numpy().reshape(-1, 1)
            column_arrays.append(mapped_col)

    candidate_set = np.hstack(column_arrays)

    
    print(f'Candidate set ready with {candidate_set.shape[0]} points')
    return candidate_set

def create_model_matrix(design:np.ndarray, model:pd.DataFrame, dict_all_factor_col_locations:dict):
    model_mat = np.ones((design.shape[0],1))
    for row in model.iterrows():
        concerning_effects = row[1][row[1]>0]        
        temp1 = np.ones((design.shape[0],1))
        for itr_factor_name, order in concerning_effects.items():
            col_ixs = dict_all_factor_col_locations[itr_factor_name]
            temp2 = design[:,col_ixs]**order
            if temp1.shape[1]==1 or len(col_ixs)==1:#continuous or categorical two-level factor
                temp1 = temp1 * temp2
            else:#categorical factor with more than two levels
                temp1 = (temp1[:,:,None] * temp2[:,None,:]).reshape(temp1.shape[0], -1)
        model_mat = np.hstack((model_mat, temp1))
    return model_mat

def calculate_cost_per_row_per_factor(design:np.ndarray, dict_all_factor_col_locations:dict, dict_of_factors:dict):

    cost_related_factors = [factor_name for factor_name, factor_instance in dict_of_factors.items() if factor_instance.cost_control == True]
    total_budgets = []
    cost_array = np.zeros((design.shape[0],len(cost_related_factors)))
    for col_ix, factor_name in enumerate(cost_related_factors):
        cols_in_design = dict_all_factor_col_locations[factor_name]
        factor_instance = dict_of_factors[factor_name]
        total_budgets.append(factor_instance.budget)
        if len(cols_in_design) == 1:
            mapping = dict(zip(factor_instance.coded_levels, factor_instance.cost_per_level))
            cost_array[:,col_ix] = np.vectorize(mapping.get)(design[:,cols_in_design[0]])
        else:
            # Build a mapping from tuple(row) -> value
            mapping = {tuple(row): val for row, val in zip(factor_instance.coded_levels, factor_instance.cost_per_level)}
            cost_array[:,col_ix] = np.array([mapping[tuple(row)] for row in design[:,cols_in_design]])
    return cost_array, total_budgets

def get_start_design(candidate_set_expanded:np.ndarray, no_start_points:int, cost_array:np.ndarray, total_budgets:list, criterion:str, prng):
    for itr in range(10000):# attempts to generate starting design for each start
        potential_candidate_points = prng.sample(range(candidate_set_expanded.shape[0]), no_start_points)
        #cost check
        current_cost_for_all_vars = cost_array[potential_candidate_points]
        low_cost = True
        if np.any(current_cost_for_all_vars.sum(axis=0) > total_budgets):
            low_cost = False
            continue
        
        if low_cost:
            full_mat = candidate_set_expanded[potential_candidate_points,:]
            info_mat_i = full_mat.T @ full_mat
            det = np.linalg.det(info_mat_i)
            if det > 0:
                if criterion == 'D':
                    return full_mat, current_cost_for_all_vars, det
                # elif criterion_i == 'I':
                #     tmp_full_avg_var = np.trace(np.linalg.inv(info_mat_i)@moment)/moment[0,0]
                #     return full_mat, current_cost_for_all_vars, tmp_full_avg_var
                elif criterion == 'A':
                    return full_mat, current_cost_for_all_vars, np.trace(np.linalg.inv(info_mat_i))
            else:
                continue

    return None

def evaluation_func(criterion:str):
    if criterion == 'D':
        def calc(model_mat:np.ndarray, current_score:float):
            val = np.linalg.det(model_mat.T @ model_mat)
            return val > current_score, val
    elif criterion == 'A':
        def calc(model_mat:np.ndarray, current_score:float):
            info_mat = model_mat.T @ model_mat
            val = np.trace(np.linalg.inv(info_mat))
            return val < current_score, val
    return calc

def neigborhood_search(candidate_set_expanded:np.ndarray, start_design:np.ndarray, start_des_criterion_value:float, current_cost:np.ndarray, cost_array:np.ndarray, total_budgets:list, relation:str, search_style:str, evaluation_calculation, prng):
    
    # inputs (default settings)
    replication = 'y'

    available_budget = total_budgets - current_cost.sum(axis=0)

    no_rows_to_drop = int(relation[0])
    no_rows_to_add = int(relation[1])
    list1 = list(itertools.combinations(range(start_design.shape[0]), no_rows_to_drop)) # to remove
    if replication == 'y':
        list2 = list(itertools.product(range(candidate_set_expanded.shape[0]), repeat=no_rows_to_add)) # to append
    elif replication == 'n':
        list2 = list(itertools.combinations(range(candidate_set_expanded.shape[0]), no_rows_to_add))
    list_ixs = list(itertools.product(list1,list2))
    if search_style == 'random':
        prng.shuffle(list_ixs)

    #initialize
    change_made = False
    current_best_des = start_design.copy()
    current_best_criterion_value = start_des_criterion_value
    current_best_cost = current_cost
    latest_cost = current_cost

    for row_to_drop, row_to_add in list_ixs:
        new_runs_costs = cost_array[row_to_add,:]
        cost_new = new_runs_costs.sum(axis=0)
        cost_reduction = current_best_cost[row_to_drop,:].sum(axis=0)
        additional_cost = cost_new - cost_reduction
        if np.any(additional_cost > available_budget):
            continue
        tmp_full = np.r_[np.delete(start_design, list(row_to_drop), 0), candidate_set_expanded[list(row_to_add),:]]
        quality_check, criterion_value = evaluation_calculation(tmp_full, current_best_criterion_value)
        if quality_check:
            change_made = True
            removed_old_run_costs = np.delete(current_best_cost, row_to_drop, axis=0)
            latest_cost = np.r_[removed_old_run_costs, new_runs_costs]
            current_best_criterion_value = criterion_value
            current_best_des = tmp_full.copy()
            if search_style != 'best': # first improvement strategy (random or sequential)
                break

    return current_best_criterion_value, change_made, current_best_des, latest_cost

def convert_design_to_uncoded_levels(design:np.ndarray, dict_of_factors:dict, model:pd.DataFrame, dict_all_factor_col_locations:dict):
    main_eff_locs = {}
    idx  = 1
    for eff in model.values:
        if eff.sum() == 1:
            factor_name = model.columns[np.where(eff > 0)[0][0]]
            no_cols = len(dict_all_factor_col_locations[factor_name])
            main_eff_locs[factor_name] = [col for col in range(idx, idx + no_cols)]
            idx += no_cols
        else:
            idx += 1

    uncoded_design_columns = []
    for factor_name, factor_instance in dict_of_factors.items():
        if factor_name in main_eff_locs:
            cols_in_design = main_eff_locs[factor_name] 
            if factor_instance.factor_type == 'categorical':
                mapping = {tuple(v): k for k, v in zip(factor_instance.coded_levels, factor_instance.uncoded_levels)}
                uncoded_cols = np.array([mapping[tuple(row)] for row in design[:,cols_in_design]])
                uncoded_design_columns.append(uncoded_cols.reshape(-1, 1))
            else:
                mapping = dict(zip(factor_instance.coded_levels, factor_instance.uncoded_levels))
                uncoded_col = np.vectorize(mapping.get)(design[:,cols_in_design[0]])
                uncoded_design_columns.append(uncoded_col.reshape(-1, 1))
    uncoded_design = np.hstack(uncoded_design_columns)
    return uncoded_design
                                   
def generate_vns_design(params:dict):

    # defaults:
    neighbors = ['01', '11', '12', '22']
    search_style = 'random'#'best'
    prng = params['prng']
    limit = params['run_size_limit']
    criterion = params['criterion']
    # get evaluation function
    evaluation_calculation = evaluation_func(criterion)

    # get individual locations for each factor associated columns in the model matrix (intercept included in position 0)
    dict_locations = get_locations_for_all_factors(params['all_factors'])

    # generate all combinations (taking constraints into account)
    candidate_set = generate_candidate_set(params)

    # get costs for every row for every factor
    cost_array, total_budgets = calculate_cost_per_row_per_factor(candidate_set, dict_locations, params['all_factors'])

    # add cost constraint for run size limit
    if limit != None:
        cost_array = np.c_[cost_array, np.ones((cost_array.shape[0],1))]
        total_budgets.append(limit)

    # generate the model matrix
    candidate_set_expanded = create_model_matrix(candidate_set, params['model'], dict_locations)

    for start_itr in tqdm(range(params['no_starts'])):
        # select the number of points in the initial design
        no_start_points = prng.randrange(candidate_set_expanded.shape[1]+2, candidate_set_expanded.shape[1]+3)
        try:
            des, cost, des_criterion_value = get_start_design(candidate_set_expanded, no_start_points, cost_array, total_budgets, criterion, prng)
        except:
            continue
        # here des includes model matrix columns by default and cost columns
        path = [1]
        for neighborhood_option in path:
            relation = neighbors[neighborhood_option-1]
            des_criterion_value, change_made, des, cost = neigborhood_search(candidate_set_expanded, des, des_criterion_value, cost, cost_array, total_budgets, relation, search_style, evaluation_calculation, prng)
            if change_made:# continue to next neighborhood
                path.append(1)
                continue
            else:
                if neighborhood_option == params['max_neighborhood']:
                    break
                else:
                    path.append(neighborhood_option+1)
        
        if start_itr == 0:
            check = True
        else:
            if criterion == 'D':
                check = des_criterion_value > best_criterion_value
            elif criterion == 'A' or criterion == 'I':
                check = des_criterion_value < best_criterion_value

        if check:
            best_criterion_value = des_criterion_value
            best_design = des.copy()
            best_cost = cost

    best_design = convert_design_to_uncoded_levels(best_design, params['all_factors'], params['model'], dict_locations)

    print('Cost \n',best_cost.sum(axis=0))
    print('Design \n',best_design)
    print('Criterion value \n',best_criterion_value)
    print('Shape of design \n', best_design.shape)
    print('Number of unique combinations \n', len(np.unique(best_design, axis = 0)))

    # convert back to uncoded levels
    np.savetxt(f'vns_design.csv', best_design)