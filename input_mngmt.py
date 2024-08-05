import pandas as pd
import math
import argparse
import xml.etree.ElementTree as ET
from py_wake.wind_turbines import WindTurbine

def read_input_data():
    """
    Reads data from the xls input file and parses the data

    Parameters:
    input_file: Excel input file path (default=test_model_wind).

    Returns:
    input_data: Parsed data from the input file.
    """
        # Create the parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument(
        '--file_path', 
        nargs='?',  # This makes the argument optional
        default='examples/Models/test_model_wind.xlsx',  # Specify your default file path here
        type=str, 
        help='Path to the Excel file (default: test_model_wind.xlsx)'
    )

    input_file_path = parser.parse_args().file_path
    # Read input file and store
    print(f'Reading input file {input_file_path}')
    input_file = pd.ExcelFile(input_file_path)

    # Read the 'micrositing' sheet with all the input parameters
    # basic_params = input_file.parse('training_data_params').fillna('').map(lambda x: x.strip() if type(x) == str else x).set_index('Parameter')
    # ml_model_params = input_file.parse('ml_model_params').fillna('').map(lambda x: x.strip() if type(x) == str else x).set_index('Parameter')
    # micrositing_opt = input_file.parse('micrositing_opt').fillna('').map(lambda x: x.strip() if type(x) == str else x).set_index('Parameter')
    
    # Assuming input_file is an ExcelFile object from pandas

    # Read and clean 'training_data_params' sheet
    basic_params = input_file.parse('training_data_params').fillna('')
    basic_params = basic_params.applymap(lambda x: x.strip() if isinstance(x, str) else x).set_index('Parameter')

    # Read and clean 'ml_model_params' sheet
    ml_model_params = input_file.parse('ml_model_params').fillna('')
    ml_model_params = ml_model_params.applymap(lambda x: x.strip() if isinstance(x, str) else x).set_index('Parameter')

    # Read and clean 'micrositing_opt' sheet
    micrositing_opt = input_file.parse('micrositing_opt').fillna('')
    micrositing_opt = micrositing_opt.applymap(lambda x: x.strip() if isinstance(x, str) else x).set_index('Parameter')

    
    # Storing all input data in a dictionary
    input_data = {'training_data_params': basic_params,
                  'ml_model_params': ml_model_params,
                  'micrositing_opt': micrositing_opt}
    
    # Taking the Turbine model and storing Turbine specs from its corresponding .wtg file
    turbine_model = input_data['training_data_params'].loc['Wind Turbine Model'].item()
    turbine_data = wtg_to_dict(turbine_model=turbine_model)
    input_data['turbine_data'] = turbine_data
    
    #Performing checks on the input data
    check_input_data(input_data)

    return input_data, input_file_path

def check_input_data(input_data):
    """
    Check validity of input data

    Parameters:
    input_data: Parsed data from the input file.

    Returns:
    """

    # Initializing variables
    grid_size = input_data['training_data_params'].loc['Grid Size'].item()
    min_turbs = input_data['training_data_params'].loc['Min Turbines'].item()
    max_turbs = input_data['training_data_params'].loc['Max Turbines'].item()
    num_layouts = input_data['training_data_params'].loc['Number of binary layouts'].item()
    checks_passed = True
    error_msg = ""

    # Check for validity of grid parameters
    if not isinstance(grid_size, int):
        checks_passed = False
        error_msg += f"- Grid Size {grid_size} is not an integer.\n"
    if not isinstance(min_turbs, int):
        checks_passed = False
        error_msg += f"- Minimum Turbines {min_turbs} is not an integer.\n"
    if not isinstance(max_turbs, int):
        checks_passed = False
        error_msg += f"- Maximum Turbines {max_turbs} is not an integer.\n"
    if not isinstance(num_layouts, int):
        checks_passed = False
        error_msg += f"- Maximum Turbines {num_layouts} is not an integer.\n"
    if not min_turbs <= max_turbs:
        checks_passed = False
        error_msg += f"- Minimum Turbines {min_turbs} is not lower than or equal to Maximum Turbines {max_turbs}.\n"
    if not max_turbs <= grid_size:
        checks_passed = False
        error_msg += f"- Maximum Turbines {max_turbs} is not lower than or equal to Grid Size {grid_size}.\n"
    
    # Check if the square of the calculated square root equals the original value
    grid_sqrt = math.isqrt(grid_size) 
    if not grid_sqrt * grid_sqrt == grid_size:
        checks_passed = False
        error_msg += f"- Grid Size {grid_size} cannot be arranged in a square shape.\n"
    
    if not checks_passed:
        raise Exception("Error in inputs: \n" + error_msg)
    
    return None

def wtg_to_dict(turbine_model):
    """
    Creates a dictionary with turbine specifications based on turbine model. Retrieved from the turbine's wtg file

    Parameters:
    turbine model(str): Model number of the turbine

    Returns:
    turbine_dict(dict): Parsed data from the input file.
    """

    # Creating a dictionary where model name corrsponds to the file name of the respective turbine model
    turbine_dict = {}
    model_dict = {'V80': 'Vestas V80 2 MW'}

    # Load the WTG(XML) file of the turbine
    file_path = f'examples\Turbine Specs\{model_dict[turbine_model]}.wtg'
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Extract generic info of turbine
    turbine_dict['Description']  = root.get('Description')
    turbine_dict['Manufacturer Name']  = root.get('ManufacturerName')
    turbine_dict['Diameter [m]']  = float(root.get('RotorDiameter'))
    turbine_dict['Height [m]'] = float(root.find('SuggestedHeights').find('Height').text)
    turbine_dict['Cut In Speed [m/s]'] = float(root.find('PerformanceTable').find('StartStopStrategy').get('LowSpeedCutIn'))
    turbine_dict['Cut Out Speed [m/s]'] = float(root.find('PerformanceTable').find('StartStopStrategy').get('HighSpeedCutOut'))
    
    # Extracting values from the first table of wind speeds and turbine power at default air density
    table = root.findall('PerformanceTable')[0]
    turbine_dict['Air Density [kg/m3]'] = float(table.get('AirDensity'))

    # Storing all wind speeds and power values in a list
    wind_speeds_0 = []
    power_outputs_0 = []
    thrust_coefficients = []
    for data_point in table.findall('./DataTable/DataPoint'):
        wind_speeds_0.append(float(data_point.get('WindSpeed')))
        power_outputs_0.append(float(data_point.get('PowerOutput')))
        thrust_coefficients.append(float(data_point.get('ThrustCoEfficient')))

    # Extending list to include speed values until -1 and a few speeds greater than cut out speed
    # Adding 0 power to all the adiditonal speed values added before cut in and after cut out
    wind_speeds_a = []
    wind_speeds_b = []
    power_outputs_a = []
    for i in range(-1, int(min(wind_speeds_0))):
        wind_speeds_a.append(float(i))
        wind_speeds_b.append(max(wind_speeds_0)+i+2)
        power_outputs_a.append(0)

    # Appending all wind speed and power lists and storing rated power
    wind_speeds = wind_speeds_a + wind_speeds_0 + wind_speeds_b
    power_outputs = power_outputs_a + power_outputs_0 + power_outputs_a
    power_outputs = [power/1e6 for power in power_outputs] # Converting to MW
    rated_power = max(power_outputs)

    turbine_dict['Rated Power [MW]'] = rated_power
    turbine_dict['Wind Speed [m/s]'] = wind_speeds
    turbine_dict['Power Output [MW]'] = power_outputs
    turbine_dict['Thrust Coefficient [-]'] = thrust_coefficients

    # Creating a PyWake Windturbine object from the wtg file. This is required for Training dataset and simulation
    turbine_pywake = WindTurbine.from_WAsP_wtg(file_path)
    turbine_dict['PyWake Object'] = turbine_pywake

    return turbine_dict

if __name__ == "__main__":
    read_input_data()