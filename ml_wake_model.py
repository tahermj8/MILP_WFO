import numpy as np
import pandas as pd
import joblib
import random
import math

from itertools import combinations

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

from py_wake.site import UniformSite # type: ignore
from py_wake.deficit_models import IEA37SimpleBastankhahGaussianDeficit # type: ignore
from py_wake.wind_farm_models import PropagateDownwind # type: ignore
from py_wake.superposition_models import MaxSum # type: ignore

from model_analysis import model_analysis # type: ignore
from input_mngmt import read_input_data # type: ignore

def create_trained_ml_model(input_data):
    """
    Generates a trained ML Model based on the inputs from the input excel file.

    Parameters:
    input_data: Input data dictionary which is ideally read and imported using read_input_data function

    Returns:
    Trained Pipeline(object): Trained ML model using the training data generated from inputs provided in inputs file.
    Training Dataset(DataFrame): Dataset generated and used for training the ML model
    Empty Pipeline(object): Untrained ML model pipeline
    """

    # Generate a set of binary arrays to create possible layouts
    binary_arrays = binary_arrays_generator(input_data=input_data['training_data_params'])
    # Storing the grid size from the training_params sheet
    grid_size = input_data['training_data_params'].loc['Grid Size'].item()
    # Creating a wind farm model object in PyWake
    wind_farm_model = create_wind_farm_model(input_data=input_data['training_data_params'], turbine_data=input_data['turbine_data'])
    # Creating a dataset of simulation results from PyWake
    training_data = training_data_collection(input_data=input_data['training_data_params'], turbine_data=input_data['turbine_data'], binary_arrays=binary_arrays, wind_farm_model=wind_farm_model, grid_size=grid_size)

    # Creating a copy of the training dataset to extract for model analysis
    training_data_2 = training_data.copy()

    # Creating an empty ML pipeline with preprocessing and prediction model
    model_pipeline = create_ml_model_pipeline(input_data=input_data['ml_model_params'])

    save_model = input_data['training_data_params'].loc['Save Trained Model as external file'].item()
    # Passing the training data with the empty ML pipeline to create a trained ML model
    trained_model, trained_model_filename = ml_model_training(model_pipeline=model_pipeline, training_data=training_data, grid_size=grid_size, len_bin_arrays=len(binary_arrays), save_model=save_model)
    
    # Perform analysis of the trained model 
    perform_model_analysis = input_data['training_data_params'].loc['Perform model analysis'].item()
    if perform_model_analysis:
        model_analysis(model=trained_model, model_name=trained_model_filename, data=training_data_2)

    return trained_model, trained_model_filename, training_data_2, model_pipeline

def binary_arrays_generator(input_data):
    """
    Reads input file data and generates binary arrays as per the given parameters 

    Parameters:
    input_data(dict): Data read from the input file (training_data_params sheet)

    Returns:
    binary_arrays: Returns an array of binary arrays, each of the specified grid size
    """

    # Assigning input data parameters
    grid_size = input_data.loc['Grid Size'].item()
    min_turbs = input_data.loc['Min Turbines'].item()
    max_turbs = input_data.loc['Max Turbines'].item()
    num_layouts = input_data.loc['Number of binary layouts'].item()


    # Calculating the number of possible combinations for the given grid size with min and max number of turbines
    total_combinations = 0
    for i in range(min_turbs, max_turbs+1):
        total_combinations += math.comb(grid_size, i)
    print(f'There are {total_combinations:,} unique grid layouts, ranging from {min_turbs} to {max_turbs} turbines for a grid size of {grid_size}.')

    # Setting a seed for random selection
    random.seed(42)
    print('Running binary layouts generator...\n')

    # Create an empty dictionary to store all the layouts for different numbers of turbines
    all_combinations = {i: [] for i in range(min_turbs, max_turbs + 1)}

    # Generate all possible binary arrays with varying numbers of ones (representing turbines)
    for num_ones in range(min_turbs, max_turbs + 1):
        # Generate all combinations of positions for the turbines (ones) in the grid
        combos = combinations(range(grid_size), num_ones)
        
        # For each combination, create the corresponding binary array
        for combo in combos:
            # Initialize a binary array of length grid_size with all zeros
            binary_array = [0] * grid_size
            
            # Set the positions indicated by the current combination to ones (placing turbines)
            for idx in combo:
                binary_array[idx] = 1
            
            # Add the generated binary array to the dictionary under the current number of ones (turbines)
            all_combinations[num_ones].append(binary_array)


    # If the number of requested arrays is greater than the possible combinations, return all combinations
    if num_layouts == 0 or num_layouts >= total_combinations:
        binary_arrays = [item for sublist in all_combinations.values() for item in sublist]
        # Store the dictionary values as numpy arrays
        binary_arrays = np.array(binary_arrays)
        print(f'All {len(binary_arrays)} binary layouts generated\n')
        
        return binary_arrays

    # If a random subset of all the layouts is required then randomly select the arrays for each sum
    binary_arrays = []

    # This ensures the normal distribution of layout sum is observed in the sample selection as well
    for num_ones, arrays in all_combinations.items():
        proportion = len(arrays) / total_combinations
        num_samples = max(1, int(round(proportion * num_layouts)))
        binary_arrays.extend(random.sample(arrays, min(num_samples, len(arrays))))

    # If we have fewer samples due to rounding, randomly add more to meet the exact number
    while len(binary_arrays) < num_layouts:
        additional_sample = random.choice([item for sublist in all_combinations.values() for item in sublist])

        # Check if the randomly selected array does not already exist in the selected arrays
        if additional_sample not in binary_arrays:
            binary_arrays.append(additional_sample)
    # Store the dictionary values as numpy arrays
    binary_arrays = np.array(binary_arrays)

    print('Due to random sampling to maintain even distribution of all possible layouts,\nit is possible the total number of layouts might be more than requested.\n')
    print(f'Set of {len(binary_arrays)} random binary layouts generated\n')

    return binary_arrays

def create_wind_farm_model(input_data, turbine_data):
    """
    Creates wind farm model in PyWake for the given attributes.

    Parameters:
    input_data(dict): Data read from the input file (training_data_params sheet)
    turbine_data(dict): Turbine data dictionary

    Returns:
    wind_farm_model: Returns an empty wind farm model object in PyWake with the specified attributes.
    """
    # Defining Dictionaries
    site_dict = {'Uniform': UniformSite()}
    wake_models_dict = {'IEA37': IEA37SimpleBastankhahGaussianDeficit()}
    superposition_models_dict = {'MaxSum': MaxSum()}
    turbulence_models_dict = {'': None}

    # Retrieving WindTurbine Object defined in PyWake
    windturbine = turbine_data['PyWake Object']

    # Assigning PyWake objects depending on input parameters
    site = site_dict[input_data.loc['Site Type'].item()]
    wkm = wake_models_dict[input_data.loc['Wake Deficit Model'].item()]
    spm = superposition_models_dict[input_data.loc['Superposition Model'].item()]
    tbm = turbulence_models_dict[input_data.loc['Turbulence Model'].item()]

    # Creating a wind farm model with the above specified attributes
    wind_farm_model = PropagateDownwind(site,
                            windturbine,
                            wake_deficitModel=wkm,
                            superpositionModel=spm,
                            turbulenceModel = tbm)
    
    return wind_farm_model

def training_data_collection(input_data, turbine_data, binary_arrays, wind_farm_model, grid_size):
    """
    Creates the main training dataset for the machine learning Model. All simulations using PyWake will be done here based on the provided input data, set of binary array layouts and the wind farm model with wind farm attributes

    Parameters:
    input_data(dict): Takes in all the input data passed by the user, specially the number of wind directions and speeds to train the model (training_data_params)
    turbine_data(dict): Takes in turbine data dict
    binary_arrays(arr): Set of binary arrays where each array denotes a grid layout
    wind_farm_model(Object): PyWake's wind farm model which will carry out the simulation for each combination of wind speed and direction
    save_data(bool): An external download of the training dataset

    Returns:
    Training dataset(DataFrame): Simulates for all possible combinations of winds, directions and xD and compiles the data in one dataframe
    """

    # Creating a set of wind directions to simulate the model
    wdint = input_data.loc['Wind Direction Interval'].item() # Interval for wind directions specified by user
    # Wind direction in degrees
    wdstart = input_data.loc['Wind Direction Start'].item() # First wind direction (Ideally 0)
    wdend = input_data.loc['Wind Direction End'].item() # Last wind direction (Ideally 360=)
    wdlist = np.arange(wdstart, wdend, wdint).tolist() # Endpoint is not included because the results will be same as 0 degrees if end direction is 360

    # Creating a set of wind speeds to simulate the model
    wsint = input_data.loc['Wind Speed Interval'].item() # Interval for wind speeds specified by user
    wsmax = input_data.loc['Wind Speed End'].item()   # First wind speed
    wsmin = input_data.loc['Wind Speed Start'].item() # Last wind speed
    wslist = np.arange(wsmin, wsmax+1, wsint).tolist() # Creating intervals for wind speeds to simulate against

    # Creating a set of inter-turbine ratios to simulate the model
    minxD = input_data.loc['min xD'].item()  # min xD specified by user
    maxxD = input_data.loc['max xD'].item()  # max xD specified by user
    xDint = input_data.loc['xD Interval'].item() # Interval for each xD
    xDlist = np.arange(minxD, maxxD+1, xDint).tolist()

    # Turbine Diameter is extracted from the Turbine model specified in the Pywake wind farm model
    D = turbine_data['Diameter [m]']

    # Printout of total number of samples that will be generated depending on the number of layouts, wind speeds and directions
    print(f'Total training data samples that will be generated: {len(binary_arrays)*len(wdlist)*len(wslist)*len(xDlist)}')

    # Creating an empty dataframe to store simulation data
    data_df = pd.DataFrame(columns=['coords','nTurb','ws','wd','xD','WS_eff','P_eff'])
    
    i=0 # index counter

    ## Beginning of looping over each sample layout and storing their simulation results ##

    #Loop 1 - Iterates over the xD inter-turbine set. It is necessary for this to be the first loop in order to generate grid coordinates according to the given xD value
    for xD in xDlist:      
            # Calling the generate_coordinates function to generate a set of coordinates                          
            coordinates = np.array(generate_coordinates(D=D, xD=xD, grid_size=grid_size))
            
            # # Transform binary arrays to coordinates  
            coordlist = [] 
            for binary_array in binary_arrays: 
                    # Multiply each element of the binary array with the corresponding coordinate
                    result = coordinates * binary_array[:, np.newaxis]
                    coordlist.append(result)
            
            
            #Loop 2 - This loop loops through all the sets of coordinates which is equivalent to the number of binary arrays that were generated
            for coords in coordlist:                
                    
                    # From each coordinate set, all x coordinates and all y coordinates are separated to two different sets
                    # If a binary array is superimposed on coordinates, it will generate multiple (0,0) corodinates. These need to be removed before sending the coordinates for Pywake simulation in order to eliminate them from consideration
                    x_coords, y_coords = zip(*coords[~np.all(coords == [0, 0], axis=1)])

                    #Loop 3 & 4 - # Iterate over all wind directions and wind speeds
                    for ws in wslist:               
                            
                            for wd in wdlist:
                                    
                                    # Simulate wind park
                                    sim_res = wind_farm_model(x=x_coords, y=y_coords, ws=ws, wd=wd)

                                    # This part is storing the effective wind speed at each turbine as a list for each simulation.
                                    # Layouts with one turbine need be stored differently in order to avoid breaking PyWake's model
                                    if len(x_coords) > 1:
                                            WS_eff = np.squeeze(sim_res.WS_eff_ilk).tolist()
                                            P_eff = np.squeeze(sim_res.power_ilk).tolist()
                                    else:
                                            WS_eff = [np.squeeze(sim_res.WS_eff_ilk.item())]
                                            P_eff = [np.squeeze(sim_res.power_ilk.item())]
                                    
                                    # Speed and power values for the turbines which were removed by our binary array are being manually re-added to our dataset after simulation as 0 values.
                                    # Find the indices of coordinates with values (0,0)...
                                    zero_indices = [index for index, coord in enumerate(coords.tolist()) if coord == [0, 0]]
                                    # ...and Insert zeros into the readings list at those indices
                                    for index in zero_indices:
                                            WS_eff.insert(index, 0)
                                            P_eff.insert(index, 0)
                                    
                                    #Storing all collected data to the dataframe
                                    data_df.at[i,'coords'] = coords.tolist() # Set of all coordinates for the placed turbine
                                    data_df.at[i,'nTurb'] = len(x_coords) # Number of turbines for the specific simulation
                                    data_df.at[i,'ws'] = sim_res.ws.item()  # Free stream Wind speed for siulation
                                    data_df.at[i,'wd'] = sim_res.wd.item() # Wind direction for simulation
                                    data_df.at[i,'xD'] = xD # xD factor for the simulation
                                    data_df.at[i,'WS_eff'] = WS_eff # Effective wind speed calculated by PyWake for each turbine stored as a set
                                    data_df.at[i,'P_eff'] = P_eff   # Effective wturbine power calculated by PyWake for each turbine stored as a set
                                    i += 1
                    
                    # Printing the simulations completed after one loop        
                    print(f'Data collected: {len(data_df)}')

    # Here the coordinates are re-arranged from [(x,y)] format to [x] and [y] using another function
    data_df = data_df.apply(extract_coordinates, axis=1)
    data_df = data_df.drop(columns=['coords'])

    print('Training Dataset created')

    # Save the trained dataset as an external pkl file if required
    save_data = input_data.loc['Save Training Dataset as external file'].item()
    if save_data == True:
        joblib.dump(data_df, f'data/Trained Datasets/training_data_{grid_size}T_{len(binary_arrays)}L_{len(wslist)}S_{len(wdlist)}D_{len(xDlist)}xD.pkl')
        print(f'Training data extracted as "training_data_{grid_size}T_{len(binary_arrays)}L_{len(wslist)}S_{len(wdlist)}D_{len(xDlist)}xD.pkl" in data/Trained Datasets')

    return data_df

def generate_coordinates(D, xD, grid_size, buffer=500):
    """
    Generates coordinates based on the diameter of turbine, inter-turbine distance ratio and the grid size

    Parameters:
    D(int or float): Turbine diameter
    xD(int or float): Inter-turbine distance scale dependent on the diameter of turbine D
    grid_size(int): Coordinates need to be generated for how many number of turbines (in a square grid)
    buffer(int): This is to determine the first turbine coordinates offset from (0,0) (default=500) 

    Returns:
    Returns an array of coordinates for turbine placements [(x1,y1),(x2,y2)...] for the given grid size
    """

    # Multiple turbine diameter with inter turbine distance factor to get the minimum distance between any two turbines
    nxD = D * xD
    # To calculate how many tubines will be placed for a grid of size N in a square n*n shape
    Turbside = math.isqrt(grid_size)

    # Generating coordinate set depending on nxD and specified initial buffer
    coordinates = [] 
    for i in range(Turbside):
        for j in range(Turbside):
            x = buffer + (i * nxD)  # Starting x-coordinate at buffer
            y = buffer + (j * nxD)  # Starting y-coordinate at buffer
            coordinates.append((x, y))

    #Converting list to array
    coordinates = np.array(coordinates)
    return coordinates

def extract_coordinates(row):
    """
    Splits the [(x,y)] coordinates to separate x and y values

    Parameters:
    row: takes in a row of dataframe which has coordinates in column 'coords'

    Returns:
    Returns multiple columns with each 
    """
    coordinates = row['coords']
    x_coords = []
    y_coords = []
    # Store each coordinate in a separate list and create [x] and [y] lists
    for i, point in enumerate(coordinates):
        x_coords.append(point[0])
        y_coords.append(point[1])
    # Store each x coordinate in a separate column
    for i, x in enumerate(x_coords):
        row[f'x{i+1}'] = x
    # Store each y coordinate in a separate column
    for i, y in enumerate(y_coords):
        row[f'y{i+1}'] = y
    
    # Return the modified row
    return row

def create_ml_model_pipeline(input_data):
        """
        Creates a pipeline with pre-processing models and neural network model

        Parameters:
        input_data(dict): Takes input data with specifications of the ANN model (ml_model_params)

        Returns:
        pipeline(Object): Returns a ML Pipeline with Standard Scaler for preprocessing of data and MLP Regressor (ANN) with 'relu' activation
        """

        neurons = input_data.loc['neurons'].item()
        hidden_layers = input_data.loc['hidden layers'].item()
        hidden_layer_sizes = (neurons,) * hidden_layers
        max_iter = input_data.loc['max iter'].item()
        random_state = 42

        pipeline = Pipeline([
            ('scaler', StandardScaler()),  # StandardScaler for feature scaling
            ('mlp', MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation='relu', solver='adam', max_iter=max_iter, random_state=random_state))  # MLPRegressor as the neural network model
        ])
        return pipeline

def ml_model_training(model_pipeline, training_data, grid_size, len_bin_arrays, save_model=False):
        """
        Creates a trained machine learning model based on training dataset and the ML model configuration

        Parameters:
        model_pipeline(Object): Untrained model pipeline
        training_data(Dataframe): Training dataset which needs to be fit to the model
        grid_size(int): Size of the grid model is being trained for (Only for file naming purposes)
        len_bin_arrays(int): Total binary layouts sampled (Only for file naming purposes)
        save_model(bool): Save trained ML model offline for use with optimization

        Returns:
        model_pipeline(Object): Returns a trained ML Pipeline with Standard Scaler for preprocessing of data and MLP Regressor (ANN) with 'relu' activation
        """

        # Setting the effective wind speed from each turbine as our target variable that we want our model to predict
        y = np.vstack(np.array(training_data.WS_eff)) # Target

        # Dropping an extra columns that we don't want as target or features of our ML model
        training_data.drop(columns=['nTurb', 'xD', 'WS_eff', 'P_eff'], inplace=True)

        # Setting all remaining columns as our features for the ML
        X = training_data.values.astype(float) # Features

        # Fit and train the pipeline on the training data
        model_pipeline.fit(X, y)

        print('Trained ML Model generated')

        # Save the trained ML model as an external pkl file if required
        if save_model == True:
            hidden_layers = len(model_pipeline.named_steps['mlp'].hidden_layer_sizes)
            neurons = model_pipeline.named_steps['mlp'].hidden_layer_sizes[0]
            joblib.dump(model_pipeline, f'data/Trained ML models/trained_model_pipeline_{grid_size}T_{len_bin_arrays}L_{neurons}x{hidden_layers}.pkl')
            print(f'Trained model extracted as "trained_model_pipeline_{grid_size}T_{len_bin_arrays}L_{neurons}x{hidden_layers}.pkl" in data/Trained ML models')
            model_filename = f'trained_model_pipeline_{grid_size}T_{len_bin_arrays}L_{neurons}x{hidden_layers}'
        else:
            model_filename = None
            
        return model_pipeline, model_filename

def main():
    """
    Executes when main script is called

    Parameters:
    None
    
    Returns:
    None
    """
    # Process input file
    input_data, _ = read_input_data()

    # Send the processed data to create a trained ML model based on provided user inputs. Also extract the training dataset and the untrained ML model for model evaluation
    trained_model, model_filename, training_data, model_pipeline = create_trained_ml_model(input_data)

if __name__ == "__main__":
    main()

