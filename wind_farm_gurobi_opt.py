import gurobipy as gp
from gurobipy import GRB
from gurobi_ml import add_predictor_constr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import datetime
import time
import itertools
from ml_wake_model import create_trained_ml_model, generate_coordinates, create_wind_farm_model  # type: ignore
from input_mngmt import read_input_data # type: ignore
import os
import shutil

def create_micrositing_gurobi_model(input_data, trained_ML_model_import, input_file_path):
    """
    Creates a Gurobi Model based on all the input specifications for the micrositing modelling.

    Parameters:
    input_data: Input data dictionary which is ideally read and imported using read_input_data function
    trained_ML_model_import(str): Takes a string input to determine if a wake ML model needs to be trained from scratch or an existing ML model can be imported
                                'filename_ml_model' - Imports a pre-saved .pkl file with the wake ML model
                                '' - Generates a wake ML model from scratch (Default)
    input_file_path(str): Passing location of the input file to create a user copy to store with results

    Returns:
    Gurobi Model(object): Gurobi model generated based on the input specs.
    Output folder (str): Location of results folder
    Time stamp (str): Timestamp when script was initiated
    """
    # Create output folder with timestamp inside the parent folder
    parent_folder = 'wind_opt_results'
    os.makedirs(parent_folder, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_folder = os.path.join(parent_folder, f'test_{timestamp}')
    os.makedirs(output_folder)

    # Copy input file to the output folder with timestamp
    output_input_file = os.path.join(output_folder, f'input_{timestamp}.xlsx')
    shutil.copy2(input_file_path, output_input_file)


    ### Organising variables and parameters ###

    # Assigning constant parameters from the input file
    n_turbs = input_data['training_data_params'].loc['Grid Size'].item() # Grid Size of the model
    D = input_data['turbine_data']['Diameter [m]'] # Diameter of the Turbine
    xD = input_data['micrositing_opt'].loc['xD'].item() # Inter-Turbine Distance factor
    maxTurbines = input_data['training_data_params'].loc['Max Turbines'].item() # Maximum turbines the model needs to place in a grid of size n_turbs
    minTurbines = input_data['training_data_params'].loc['Min Turbines'].item()
    maxTurbine_cond = input_data['micrositing_opt'].loc['Max Turbines Condition'].item()
    slack = 1 + input_data['micrositing_opt'].loc['Speed bounds slack'].item() # Wind speed upper boundary slack

    # Creates or Imports a trained wake ML model to use with the optimization model 
    if trained_ML_model_import == '':
        # Calls the function form the ml_wake_model.py
        trained_model, _, _ = create_trained_ml_model(input_data)
    else:
        # Importing an existing trained model instead from folder 'Trained ML models' 
        print('Importing trained ML Model')
        trained_model = joblib.load(f'examples/Trained ML models/{trained_ML_model_import}.pkl')

    # Generates coordinates based on turbine diameter and inte-turbine distance factor
    coordinates = generate_coordinates(D=D, xD=xD, grid_size=n_turbs)
    # Extract x and y coordinates of the grid
    x0, y0 = zip(*coordinates)

    # Turbine wind speed and power array generation as per turbine specs (Lookup table- PWL constraint of the model depends on this input)

    power_scale = input_data['micrositing_opt'].loc['Scale Power Factor'].item()

    speed_x = input_data['turbine_data']['Wind Speed [m/s]']
    # Importing power values of the turbine at the respective wind speeds as per turbine specs
    power_y   = input_data['turbine_data']['Power Output [MW]']
    power_y = [value*power_scale for value in power_y] # Scale the power values by the giving power scaling factor
    # Assert the array size of turbine wind speed and power specs is the same for the PWL constraint to function correctly
    assert len(speed_x) == len(power_y), 'Turbine Wind Speeds and Power arrays are not of the same size'

    # Adding a -1 as speed to provide a buffer for our ML model as it can predict small negative values. Our model will return 0W power for any small negative values passed to it.
    speed_x.insert(0, -1)
    power_y.insert(0, 0)

    ### Main Design variables of the model. Freestream Wind Speed and Wind Direction ###
    windspeeds = input_data['micrositing_opt'].loc['Wind Speeds'].item()
    winddirections = input_data['micrositing_opt'].loc['Wind Directions'].item() 
    
    # Handling of singular or multiple values of wind speeds and directions
    if isinstance(windspeeds, str) and  isinstance(winddirections, str):
        windspeeds      = [float(value.strip()) for value in input_data['micrositing_opt'].loc['Wind Speeds'].iloc[0].split(',')]
        winddirections  = [float(value.strip()) for value in input_data['micrositing_opt'].loc['Wind Directions'].iloc[0].split(',')]

    # Validating if the number of wind speeds and directions are same if multiple values are passed
    if isinstance(windspeeds, list) and isinstance(winddirections, list):
        assert len(windspeeds) == len(winddirections), 'Wind Speeds and Wind Directions arrays are not of the same size'
        # Length of the wind speed/direction inputted determines the number of timesteps of the model
        time_steps = len(windspeeds)
        # Creating an upper bound for the windspeeds to be assigned in the model 
        ws_bounds = [value * slack for value in windspeeds]
    else:
        # Time step is 1 if only a single value of wind speed/direction is passed to the model
        time_steps = 1
        ws_bounds = windspeeds * slack
    
    # Turbine rated Power
    P_nom = input_data['turbine_data']['Rated Power [MW]'] * power_scale # Scale the P_nom if required

    ### Creating the Gurobi Model based on the above specifications and parameters assigned ###
    
    grb_mdl = gp.Model('Micrositing Model') # Creating a Gurobi model object

    ws_var      = grb_mdl.addMVar(time_steps,               vtype=GRB.CONTINUOUS, lb=windspeeds,        ub=windspeeds,          name="ws"       )   # wind speed variable for the wind speeds input from the file
    wd_var      = grb_mdl.addMVar(time_steps,               vtype=GRB.CONTINUOUS, lb=winddirections,    ub=winddirections,      name="wd"       )   # wind directions variable for the wind directions input from the file
    x_coord     = grb_mdl.addMVar(n_turbs,                  vtype=GRB.CONTINUOUS,                                               name="x_coord"  )   # x coordinates of all the turbines placed by the model
    y_coord     = grb_mdl.addMVar(n_turbs,                  vtype=GRB.CONTINUOUS,                                               name="y_coord"  )   # y coordinates of all the turbines placed by the model
    n           = grb_mdl.addMVar(n_turbs,                  vtype=GRB.BINARY,                                                   name=f"locbin"  )   # Main binary decision variable of which turbine will be selected by the model
    Ve          = grb_mdl.addMVar((n_turbs, time_steps),    vtype=GRB.CONTINUOUS, lb=-1,                ub=ws_bounds,           name='Ve'       )   # Effective wind speed variable that will be estimated by the ML wake model
    Pe          = grb_mdl.addMVar((n_turbs, time_steps),    vtype=GRB.CONTINUOUS, lb=0,                 ub=P_nom,               name=f"Pe"      )   # Effective turbine power that will be determined using the PWL Lookup constraint


    grb_mdl.addConstrs((x_coord[i] == x0[i] * n[i] for i in range(n_turbs) ), name='xcon') # binary * x coordinate
    grb_mdl.addConstrs((y_coord[i] == y0[i] * n[i] for i in range(n_turbs) ), name='ycon') # binary * y coordinate

    # Constraint to ensure the number of turbines placed are fewer than or equal to the maximum turbines specified by user
    if maxTurbine_cond == '=':
        grb_mdl.addConstr((sum(n) == maxTurbines),        name="maxturbines")
    elif maxTurbine_cond == '<=':
        grb_mdl.addConstr((sum(n) <= maxTurbines),        name="maxturbines")
        grb_mdl.addConstr((sum(n) >= minTurbines),        name="minturbines")

    for i in range(n_turbs):
        for t in range(time_steps):
            # Constraints to add wake ML model for each time step for the given pair of wind speed and direction
            add_predictor_constr(grb_mdl, trained_model, ([ws_var[t].tolist(), wd_var[t].tolist()] + list(itertools.chain(*[x_coord.tolist(), y_coord.tolist()]))), Ve[:,t])

            # Constraints to add the PWL (lookup table) constraint for each effective wind speed Ve estimated by the wake model constraint
            grb_mdl.addGenConstrPWL(Ve[i,t], Pe[i,t], speed_x, power_y, name=f'PWL{[i,t]}')

    # Update the model
    grb_mdl.update()

    # Setting Model solving parameters. MIP Gap for how close a solution to the optimal solution is acceptable and how long do you want the model to run before retrieving the results
    MIP_Gap = input_data['micrositing_opt'].loc['MIP Gap'].item()
    time_limit = input_data['micrositing_opt'].loc['Time Limit'].item()
    grb_mdl.setParam('MIPGap', MIP_Gap)

    if time_limit == '':
        pass
    else:
        grb_mdl.setParam('TimeLimit', time_limit)
    

    # Objective function of the model to maximize the Power output of the wind farm for the maximum number of turbines that can be deployed
    obj_expr = sum(sum(Pe[:,t]*n for t in range(time_steps)))
    grb_mdl.setObjective(obj_expr, gp.GRB.MAXIMIZE)

    # Option to extract Unoptimized Gurobi Model
    extract_model_unopt = input_data['micrositing_opt'].loc['Extract Gurobi Model (Unoptimized)'].item()
    if extract_model_unopt:
        grb_mdl.write(f"gurobi_model_unopt_{timestamp}.lp")

    grb_mdl.setParam('LogFile', f'{output_folder}/LogFile_{timestamp}')
    

    return grb_mdl, output_folder, timestamp

def solve_gurobi_model(grb_mdl, solution_import):
    """
    Solves the Gurobi Model for the unoptimized model passed to the function.

    Parameters:
    grb_mdl(Object): An unsolved Gurobi model 
    solution_import(str): Takes a string input to determine if the passed gurobi model needs to be optimized or an existing gurobi model solution can be imported to save solving time
                                'gurobi_model_opt' - Imports a pre-saved .sol file with the results from a previous optimization
                                '' - Optimizes the above passed gurobi model (Default)

    Returns:
    Gurobi Model(object): Gurobi model generated based on the input specs.
    elapsed_time_str(str): Calculates the time taken to optimize the mdoel and stores it in a string format of HH:MM:SS
    """

    # Solve the Gurobi model for optimization if no existing solution file is passed
    if solution_import == '':
        # Record start time
        start_time = time.time()
        # Optimize model
        grb_mdl.optimize()
        # Record end time
        end_time = time.time()

        # Calculate elapsed time in seconds
        elapsed_time_seconds = int(end_time - start_time)
        # Convert elapsed time to HH:MM:SS format
        elapsed_time_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_time_seconds))

        # Print optimization time in HH:MM:SS format
        print(f"Optimization time: {elapsed_time_str}")
    else:
        # Load an existing solution file and store in the existing model
        print('Importing Existing Gurobi Solution')
        sol_file = f'examples/Gurobi Solutions/{solution_import}.sol'
        # Setting optimization time to 0 as existing solution was loaded
        elapsed_time_str = time.strftime('%H:%M:%S', time.gmtime(0))

        # Read the existing solution file into the model. Solution variables act as the starting point 
        grb_mdl.read(sol_file)
        # Model needs to run the optimization again to store the solution variables. 
        grb_mdl.optimize()
       

    return grb_mdl, elapsed_time_str

def extract_opt_model_results(gurobi_model_results, output_folder, timestamp, elapsed_time, extract_model_solution=True):
    """
    Extracts and exports results from the Gurobi model. Also generates and store plots if required.

    Parameters:
    grb_mdl(Object): A solved Gurobi model
    Output_folder (str): Location of results folder where the extracted results need to be stored
    Time stamp (str): Timestamp when script was initiated
    Elapsed Time (str): Time taken to solve the optimization model
    extract_model_solution (bool): 
    
    Returns:
    opt_array(arr): Binary array determined by optimization model
    Extracts all the results to a folder
    """

    print('Extracting Optimization Results')

    # Storing the gurobi model locally
    grb_mdl = gurobi_model_results

    # Sorting each variable of the model with respect to their sizes
    # Checking if the model was solved optimally
    if grb_mdl.status in [GRB.TIME_LIMIT, GRB.OPTIMAL, GRB.SOLUTION_LIMIT, GRB.INTERRUPTED]:
        
       # Initialize dictionaries to store values based on their type
        individual_vars = {} # This is to store any single values
        power_speed_array = {} # This is to store the 2D arrays for each turbine and timestep
        coord_array = {} # This is to save the 1D array of turbine palcement decision and their coordinates

        for var in grb_mdl.getVars():
            skip_strings = ["mlpregressor", "std_scaler"]  # List of substrings to skip. Don't want any intermediary ML variables to be stored
            skip_var = any(skip in var.VarName.lower() for skip in skip_strings) # Checks if the variable name belongs to the above ignore list
    
            if not skip_var:
                i = var.VarName[-2] # Storing the timestep index number
                stripped_var_name = var.VarName.split('[')[0] # Storing the variable name without the index number
                
                # Determine if the variable should be saved as individual value or as a list
                if stripped_var_name in ['ws', 'wd']:  # Store wind speeds and directions that were input by the user
                    individual_vars[f'{stripped_var_name}_{i}'] = var.X
                
                # Store variables that are not time step dependent in another dictionary
                elif stripped_var_name in ['locbin','x_coord','y_coord']:
                    # Create a list if it doesn't exist already
                    if stripped_var_name not in coord_array:
                        coord_array[stripped_var_name] = []
                    coord_array[stripped_var_name].append(var.X)
                
                # Else store list that are time step dependent in another dictionary and use the index i here
                else:
                    if i not in power_speed_array:
                        # Create an empty dictionary for each time step
                        power_speed_array[i] = {}
                    
                    # Create a list if it doesn't exist already in that timestep
                    if stripped_var_name not in power_speed_array[i]:
                        power_speed_array[i][stripped_var_name] = []
                    power_speed_array[i][stripped_var_name].append(var.X)
    

    # Storing the Objective value calculated by the model
    individual_vars['Wind Farm Power Output'] = grb_mdl.objVal
    # Storing the optimization time string imported earlier
    individual_vars['Optimization Time'] = elapsed_time
    # Summing up the total number of turbines deployed by the model
    individual_vars['Total Turbines Deployed'] = sum(coord_array['locbin'])
    
    # Create a DataFrame from the dictionaries
    individual_vars_df = pd.DataFrame(list(individual_vars.items()), columns=['Variable', 'Value'])
    coord_array_df = pd.DataFrame(coord_array)


    # Export DataFrame to Excel in the output folder
    excel_file = os.path.join(output_folder, f'results_{timestamp}.xlsx')
    # Write data to Excel with two sheets
    with pd.ExcelWriter(excel_file) as writer:
        individual_vars_df.to_excel(writer, sheet_name='Summary', index=False)
        coord_array_df.to_excel(writer, sheet_name='Micrositing Results', index=False)

        # Create separate sheets for each time step to store each time step respective output
        for key, value in power_speed_array.items():
            power_speed_array_df = pd.DataFrame(value)
            power_speed_array_df.to_excel(writer, sheet_name=f'Timestep_{key}', index=False)

    # Option to extract Optimized Gurobi model solution as a .sol file
    if extract_model_solution:
        if grb_mdl.status in [GRB.TIME_LIMIT, GRB.OPTIMAL, GRB.SOLUTION_LIMIT]:
            grb_mdl.write(f"{output_folder}/gurobi_model_{timestamp}.sol")

    print(f"Results exported to folder {output_folder}")

    opt_arr = np.array(coord_array['locbin'])

    return opt_arr

def extract_plots(opt_arr, input_data, output_folder, timestamp):
    """
    Generates and saves plots for the optimized solution generated by the model

    Parameters:
    opt_arr(array): A 1D array of the binary turbine placement decision variable provided by the optimized Gurobi model 
    input_data(dict): Input data dictionary which is ideally read and imported using read_input_data function
    output_folder(str): Location of the output folder where the results are stored (timestamp dependent)
    timestamp (str): Timestamp when script was initiated
    
    Returns:
    Stores wind farm simulation plot in the output folder
    """
    
    # Storing variables required to generate coordinates
    grid_size = input_data['training_data_params'].loc['Grid Size'].item()  # Grid Size of the model
    D =  input_data['turbine_data']['Diameter [m]'] # Diameter of the Turbine
    xD = input_data['micrositing_opt'].loc['xD'].item() # Inter-Turbine Distance factor
    # Generating coordinates 
    coordinates = generate_coordinates(D=D, xD=xD, grid_size=grid_size)

    # Creating an empty wind farm model based on the same specs the model was trained for
    wfm = create_wind_farm_model(input_data=input_data['training_data_params'], turbine_data=input_data['turbine_data'])

    # Multiplying the binary turbine decision to the coordinates
    coords = opt_arr[:, np.newaxis] * coordinates
    # Discarding all the coordinates are (0,0) as according to our model those turbines don't exist
    nonzero_indices = np.where(np.all(coords != 0, axis=1))
    coords = coords[nonzero_indices]

    # Storing all the x and y cooridnates of the placed turbines
    x_coords, y_coords = zip(*coords)

    # Using the user input wind speeds and directions to simulate
    windspeeds = input_data['micrositing_opt'].loc['Wind Speeds'].item()
    winddirections = input_data['micrositing_opt'].loc['Wind Directions'].item() 
    
    # Handling of singular or multiple values of wind speeds and directions
    if isinstance(windspeeds, str) and  isinstance(winddirections, str):
        windspeeds      = [float(value.strip()) for value in input_data['micrositing_opt'].loc['Wind Speeds'].iloc[0].split(',')]
        winddirections  = [float(value.strip()) for value in input_data['micrositing_opt'].loc['Wind Directions'].iloc[0].split(',')]

        # In PyWake duplicate wind speeds and wind directions cannot be plotted. This function adds a very small and negligible buffer
        # to duplicate values so that each wind speed and direction is unique for plotting purposes
        windspeeds = dupli_fixer(windspeeds)
        winddirections = dupli_fixer(winddirections)

    # Performing the simulation in PyWake for our optimized model solution
    sim_res = wfm(x=x_coords, y=y_coords, ws=windspeeds, wd=winddirections)

    # Calculating the power as per PyWake
    P_model = round(sim_res.power_ilk.sum()/1e6,3)
    
    # Generating and saving the windfarm plot using PyWake's built in Contour plot
    sim_res.flow_map().plot_wake_map()
    plt.title(f'Gurobi Optimized Layout Total Power output: {P_model} MW')
    fig = plt.gcf()
    fig.savefig(f'{output_folder}/plot_{timestamp}.png')

def dupli_fixer(list):
    """
    Adds a small buffer in any duplicate value in the list so that each value in unique

    Parameters:
    list (list): Takes a list of float or int values
    
    Returns:
    list (list): Modified list without any duplicates
    """

    value_count = {}

    for i, speed in enumerate(list):
        if speed in value_count:
            value_count[speed] += 1
        else:
            value_count[speed] = 1
        list[i] += 1e-5 * (value_count[speed] - 1)

    return list

def main():
    """
    Executes when main script is called

    Parameters:
    None
    
    Returns:
    None
    """
    # Process input file
    input_data, input_file_path = read_input_data()
    
    # Create a Gurobi model and check if a file name for a pre-trained ML model is specified
    trained_ML_model_import = input_data['micrositing_opt'].loc['Import Trained ML model'].item()
    gurobi_model, output_folder, timestamp = create_micrositing_gurobi_model(input_data=input_data, trained_ML_model_import=trained_ML_model_import, input_file_path=input_file_path)

    # Solve the Gurobi model created above and check if a file name for a pre-solved Gurobi model is specified
    solution_import = input_data['micrositing_opt'].loc['Import Gurobi Solution'].item()
    gurobi_model_results, elapsed_time_str = solve_gurobi_model(grb_mdl=gurobi_model, solution_import=solution_import)
    
    # Extract the reults from the Gurobi model and check if user wants to store the solution file for later use
    extract_model_solution = input_data['micrositing_opt'].loc['Extract Model Solution (Optimized)'].item()
    opt_arr = extract_opt_model_results(gurobi_model_results=gurobi_model_results, output_folder=output_folder, timestamp=timestamp, elapsed_time=elapsed_time_str, extract_model_solution=extract_model_solution)
    extract_plots(opt_arr=opt_arr, input_data=input_data, output_folder=output_folder, timestamp=timestamp)

if __name__ == "__main__":
    main()