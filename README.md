# Cost-Constrained Optimal Experimental Design using VNS

This repository contains a Python-based tool for generating optimal experimental designs (DOE) with a primary focus on **cost-constrained problems**. It uses a **Variable Neighborhood Search (VNS)** algorithm to find D-optimal or A-optimal designs that respect complex budget limitations and linear constraints.

> 💡 **Project Motivation**
>
> While standard statistical software (like JMP) is powerful for unconstrained experimental design, it often lacks the flexibility to handle complex, real-world constraints. This tool is built specifically to solve problems where:
> * Different factor levels have different costs.
> * There is a hard, non-negotiable budget for the experiment (or for specific factors).
> * The experimental region is limited by linear constraints (e.g., `Factor_A + Factor_B <= 10`).

This algorithm finds the best possible design *within* your budget.

---

## 🌟 Key Features

* **Cost-Constrained Optimization:** The VNS algorithm explicitly checks budget constraints *during* the search, ensuring the final design is financially feasible.
* **Flexible Factor Types:**
    * `continuous_factor`
    * `discrete_numeric_factor`
    * `categorical_factor`
* **Granular Cost Control:** Define costs per-level and set a total **budget** for any factor.
* **Model-Based:** Generates a design optimized for a specific statistical model (e.g., main effects, interactions, quadratics) defined in a simple CSV.
* **Multiple Optimality Criteria:**
    * **'D'**-Optimality
    * **'A'**-Optimality
* **Constraint Handling:** Supports linear constraints between factors (e.g., `'B <= C'`) and can enforce a total **run size limit**.
* **Reproducibility:** Uses a seeded random number generator for reproducible results.

---

## 📦 Project Structure
```
.
├── main.py                # Main entry point: Configure and run your design
├── functions.py           # Core logic, VNS algorithm, and factor dataclasses
├── model.csv              # INPUT: Your model definition file
├── requirements.txt       # Project dependencies
└── vns_design.csv         # OUTPUT: The final, generated experimental design
```

## ⚙️ Installation

### 1.  Clone the repository:
```Bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Create and activate a virtual environment:

 - Windows:
```PowerShell

python -m venv venv
.\venv\Scripts\Activate.ps1
```

 - macOS / Linux:
```Bash
python3 -m venv venv
source venv/bin/activate
```
### 3. Install dependencies:
We use the full path to the virtual environment's Python executable to ensure permissions are handled correctly.
 - Windows:
```Powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```
 - macOS / Linux:
```Bash
python -m pip install -r requirements.txt
```
#### 📝 Notes & Troubleshooting
 - **Prerequisites**: Ensure you have Python 3.9+ installed before proceeding.
 - **Windows Permissions**: If you encounter a "script is running disabled" error when activating the environment (Step 2), run this command in PowerShell:
```PowerShell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
 - **Why the full path?** In Step 3, we invoke the Python executable directly (.\venv\Scripts\python.exe) to avoid path conflicts and permission issues common in restricted IT environments.
 - **File Not Found**: If pip cannot find requirements.txt, ensure you are in the root directory of the project where the file is located.

## 🚀 How to Use

To generate a design, follow these three steps.

### Step 1: Define Your Model (`new_model.csv`)

Create a CSV file (e.g., `new_model.csv`) that defines the model terms you want to estimate.
* **Columns:** Factor names (e.g., `A`, `B`, `C`).
* **Rows:** Model terms (main effects, interactions, etc.).
* **Values:** The order of the term.

**Example: `new_model.csv`**
This model includes main effects for A, B, and C, one `A*B` interaction, and one quadratic term (`A*A`).

```csv
A,B,C
1,0,0
0,1,0
0,0,1
1,1,0
2,0,0
```

### Step 2: Configure Your Experiment (`main.py`)

Open main.py and edit the all_factors and parameters dictionaries. This is the main control panel.

#### A) Define Factors (`all_factors`)
Define each of your experimental factors. This is where you set up your cost constraints.

**Example of a Cost-Constrained Factor**: Here, Factor 'A' is discrete, and each level has a specific cost. The algorithm will not be allowed to generate a design where the total cost for Factor 'A' exceeds 142.
Python
```Python
'A': discrete_numeric_factor(
    cost_control=True,        # <-- Enable cost control for this factor
    levels=np.array([-1, 0, 1]),
    cost_per_level=[2, 8, 14],  # <-- Cost for level -1, 0, and 1
    budget=142                # <-- Total budget for Factor A
)
```

**Example of a Non-Cost-Constrained Factor**: Factor 'B' is a standard continuous factor with no budget.
Python
```Python
'B': continuous_factor(
    cost_control=False, 
    minimum=6, 
    maximum=36, 
    step_size=3
)
```

**🔬 Factor Configuration Examples**

When defining factors in `main.py`'s `all_factors` dictionary, the `cost_control` flag significantly changes the required arguments.
| Factor Type | `cost_control=False` (Standard DOE) | `cost_control=True` (Cost-Constrained DOE) |
| :--- | :--- | :--- |
| **Discrete Numeric** | Requires: `levels` (np.ndarray) | Requires: `levels`, `cost_per_level`, `budget` |
| **Continuous** | Requires: `minimum`, `maximum`, `step_size` | Requires: `minimum`, `maximum`, `step_size`, `min_cost`, `step_cost`, `budget` |
| **Categorical** | Requires: `labels` (list[str]) | Requires: `labels`, `cost_per_level`, `budget` |

1.  `discrete_numeric_factor`

A factor with distinct numeric levels, like temperature set points or number of steps.
| Scenario	| Example Code	| Explanation | 
| :--- | :--- | :--- |
| **Cost-Constrained** (`True`)	| ```'A': discrete_numeric_factor(cost_control=True, levels=np.array([1, 2, 3]), cost_per_level=[2, 8, 14], budget=142)``` | The cost for levels 1, 2, and 3 are 2, 8, and 14, respectively. The total cost for Factor A across the whole design **must not exceed 142**. | 
| **Standard** (`False`)	| ```'A': discrete_numeric_factor(cost_control=False, levels=np.array([-1, 0, 1]))``` |	Standard discrete factor. No cost is considered in the optimization for this factor. | 

2. `continuous_factor`

A factor that can take any value between a minimum and maximum, with a defined step size (e.g., concentration, pressure).
| Scenario	| Example Code	| Explanation | 
| :--- | :--- | :--- |
| **Cost-Constrained** (`True`)	| ```'B': continuous_factor(cost_control=True, minimum=10, maximum=50, step_size=5, min_cost=50, step_cost=10, budget=600)``` | Cost is a linear function of the factor level. A level of 10 costs 50, a level of 10+5 costs 50+10=60. The total cost for Factor B across the design **must not exceed 600**. | 
| **Standard** (`False`)	| ```'B': continuous_factor(cost_control=False, minimum=10, maximum=50, step_size=5)``` | Standard continuous factor. The design points will be selected from the possible levels (10, 15, 20, ..., 50). | 

3. `categorical_factor`

A factor with non-numeric labels, such as different vendors or material types.
| Scenario	| Example Code	| Explanation |
| :--- | :--- | :--- |
| **Cost-Constrained** (`True`)	| ```'C': categorical_factor(cost_control=True, labels=['VendorX', 'VendorY', 'VendorZ'], cost_per_level=[50, 80, 120], budget=500)```	| The cost for using 'VendorX', 'VendorY', and 'VendorZ' in a single run is 50, 80, and 120, respectively. The total cost for Factor C across the whole design **must not exceed 500**. | 
| **Standard** (`False`)	| ```'C': categorical_factor(cost_control=False, labels=['VendorX', 'VendorY', 'VendorZ'])```	| Standard categorical factor. No cost is considered in the optimization for this factor. | 

#### B) Set Algorithm Parameters (`parameters`)

This dictionary controls the VNS algorithm and constraints.
```Python
parameters = {}

# --- Core Setup ---
# For reproducible results, change the seed number
parameters['prng'] = random.Random(2999) 
# The dictionary of factors you defined above
parameters['all_factors'] = all_factors
# Load the model definition
parameters['model'] = pd.read_csv('new_model.csv')

# --- VNS Algorithm Controls ---
# Number of random starts for the algorithm
parameters['no_starts'] = 10 
# Max neighborhood size to explore
parameters['max_neighborhood'] = 2 
# 'D', 'A'
parameters['criterion'] = 'D' 

# --- Constraints ---
# Optional hard limit on the total number of runs
parameters['run_size_limit'] = None 
# Linear constraints between factors (using uncoded factor names)
# If there are no constraints, this line should be commented out
parameters['constraints'] = ['B <= C'] 
```
### Step 3: Run the Script

Execute `main.py` from your terminal:
```Bash
python main.py
```
## 📄 Output

The script will run the VNS algorithm, showing a progress bar (`tqdm`). When finished, it will print the results to the console:
```
Candidate set ready with 120 points
100%|██████████| 10/10 [00:01<00:00,  8.25it/s]
Cost 
 [142.   9.   0.]
Design 
     A   B   C
0   1  36  36
1  -1   6  12
2   1  33  36
3   1  12  12
4  -1   6  36
...
Criterion value 
 1234567.89
Shape of design 
 (15, 3)
Number of unique combinations 
 14
```
 - `Cost`: The final total cost used for each cost-controlled factor.

 - `Design`: A `pandas.DataFrame` of the final design (in uncoded levels).

 - `Criterion value`: The final optimality score (higher is better for 'D', lower is better for 'A').

 - `vns_design.csv`: A new CSV file will be created in your directory containing the optimal design found.
