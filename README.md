# Cost-Constrained Optimal Experimental Design using VNS

This repository contains a Python-based tool for generating optimal experimental designs (DOE) with a primary focus on **cost-constrained problems**. It uses a **Variable Neighborhood Search (VNS)** algorithm to find D-optimal, A-optimal, or I-optimal designs that respect complex budget limitations and linear constraints.

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
    * **'D'**-Optimality (default)
    * **'A'**-Optimality
    * **'I'**-Optimality
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

1.  Clone this repository:
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  Install the required Python packages from `requirements.txt`:
    *(You should list `numpy`, `pandas`, and `tqdm` in this file).*

    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 How to Use

To generate a design, follow these three steps.

### Step 1: Define Your Model (`new_model.csv`)

Create a CSV file (e.g., `new_model.csv`) that defines the model terms you want to estimate.
* **Columns:** Factor names (e.g., `A`, `B`, `C`).
* **Rows:** Model terms (main effects, interactions, etc.).
* **Values:** The order of the term.

**Example: `new_model.csv`**
This model includes main effects for A, B, and C, and one `A*B` interaction.

```csv
A,B,C
1,0,0
0,1,0
0,0,1
1,1,0
```

### Step 2: Configure Your Experiment (`main.py`)

Open main.py and edit the all_factors and parameters dictionaries. This is the main control panel.

#### A) Define Factors (all_factors)

Define each of your experimental factors. This is where you set up your cost constraints.

**Example of a Cost-Constrained Factor**: Here, Factor 'A' is discrete, and each level has a specific cost. The algorithm will not be allowed to generate a design where the total cost for Factor 'A' exceeds 142.

```Python
'A': discrete_numeric_factor(
    cost_control=True,        # <-- Enable cost control for this factor
    levels=np.array([-1, 0, 1]),
    cost_per_level=[2, 8, 14],  # <-- Cost for level -1, 0, and 1
    budget=142                # <-- Total budget for Factor A
)
```
**Example of a Non-Cost-Constrained Factor**: Factor 'B' is a standard continuous factor with no budget.

```Python
'B': continuous_factor(
    cost_control=False, 
    minimum=6, 
    maximum=36, 
    step_size=3
)
```

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
# 'D', 'A', or 'I'
parameters['criterion'] = 'D' 

# --- Constraints ---
# Optional hard limit on the total number of runs
parameters['run_size_limit'] = None 
# Linear constraints between factors (using uncoded factor names)
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

 - `Criterion value`: The final optimality score (higher is better for 'D', lower is better for 'I' and 'A').

 - `vns_design.csv`: A new CSV file will be created in your directory containing the optimal design found.
