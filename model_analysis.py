from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt # type: ignore
import numpy as np


def model_analysis(model, data):
    """
    This performs a fit and validation analysis for the provided model and dataset 

    Parameters:
    model: This is the ML model or pipeline that needs to be validated
    data: Dataset against which the model needs to be tested against

    Returns:
    Outputs RMSE, MAE, MAPE, MSE, Test score, Box plot of RMSEs and Scatter plot of prediction vs actual data
    """

    # Setting the effective wind speed from each turbine as our target variable that we want our model to predict
    y = np.vstack(np.array(data.WS_eff)) # Target

    # Dropping an extra columns that we don't want as target or features of our ML model
    data.drop(columns=['xD','P_eff','nTurb','WS_eff'], inplace=True)

    # Setting all remaining columns as our features for the ML
    X = data.values.astype(float) # Features

    # Here we split the dataset into training and validation(testing) datasets. Current default is 80-20 but can be changed
    test_size=0.2
    random_state=42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)

    # Train the model using the scaled training data
    model.fit(X_train, y_train)

    # Make predictions on the test set on the trained model
    y_pred = model.predict(X_test)


    # Calculate RMSE on test and prediction values based on an external rmse function
    rmse = rmsecalc(y_test, y_pred)
    # Evaluate the model on the scaled testing data
    test_score = model.score(X_test, y_test)
    # Calculate mean squared error on the test set
    test_mse = mean_squared_error(y_test, y_pred)
    # Calculate mean absolute error on the test set
    test_mae = mean_absolute_error(y_test, y_pred)
    # Calculate mean absolute percentage error on the test set
    test_mape = mape(y_test, y_pred)

    print("Mean Squared Error:", test_mse)
    print("Mean Absolute Error:", test_mae)
    print("Mean Absolute Percentage Error:", round(test_mape,2),'%')
    print("Model Test Score:", test_score)
    print("Mean of all RMSEs:", np.mean(rmse))

    # Create boxplot of RMSE values to see the range of errors between test and prediction values
    boxplotmaker(rmse)

    # Create a scatter plot to compare test and prediction values
    scatterplotmaker(y_test, y_pred)

def rmsecalc(y_test, y_pred):
    """
    This performs an rmse calculation for the test and prediction values passed to it

    Parameters:
    y_test: A set of original target values
    y_pred: A set of predicted target values

    Returns:
    Calculates RMSE between two datasets and returns a set of RMSE values
    """

    rmse_values = []
    for i in range(y_test.shape[0]):  # Iterate over rows
        mse = mean_squared_error(y_test[i], y_pred[i])  # Calculate mean squared error for the current row
        rmse = np.sqrt(mse)  # Calculate RMSE from MSE
        rmse_values.append(rmse)
    
    return rmse_values

def mape(y_test, y_pred):
    """
    This performs Mean Absolute Percentage Error (MAPE) calculation for the test and prediction values passed to it

    Parameters:
    y_test: A set of original target values
    y_pred: A set of predicted target values

    Returns:
    Calculates the mean MAPE between two datasets and returns a single value for MAPE which is a mean of all MAPEs
    """
     
    # Calculate absolute error
    absolute_error = np.abs(y_test - y_pred)
    
    # Calculate percentage error, handling division by zero
    y_test_nonzero = np.where(y_test == 0, 1e-8, y_test)  # Replace zeros with a small non-zero value
    percentage_error = absolute_error / np.abs(y_test_nonzero)
    
    # Calculate MAPE and handle cases where y_test is zero
    mape = np.where(y_test != 0, percentage_error * 100, 0)
    
    # Calculate the mean of MAPE values
    mean_mape = np.mean(mape)
    
    return mean_mape

def boxplotmaker(rmse):
    """
    Creates a box plot for a set of RMSE values passed to it. This gives us a visual understanding of the error variation between test and prediction values

    Parameters:
    rmse: A set of calculated RMSE values

    Returns:
    Generates a boxplot with whiskers to give a range of the prediction error
    """

    # Box plot
    plt.figure(figsize=(8, 6))
    box = plt.boxplot(rmse, vert=True, showfliers=False)
    plt.xlabel('RMSE')
    plt.title('Box Plot of RMSE Values')
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Get the whiskers' positions
    lower_whisker_pos = box['whiskers'][0].get_ydata()
    upper_whisker_pos = box['whiskers'][1].get_ydata()

    # Count the number of extreme outliers
    extreme_outliers = [value for value in rmse if value < lower_whisker_pos.min() or value > upper_whisker_pos.max()]
    num_extreme_outliers = len(extreme_outliers)

    print("Total RMSE values (Validation set):", len(rmse))
    print("Number of extreme outliers:", num_extreme_outliers, ',' ,round(num_extreme_outliers/len(rmse)*100,1),'%')
    plt.show()

def scatterplotmaker(y_test, y_pred):
    """
    Creates a scatter plot for the testing and predicted values of the target dataset. This gives us a visual understanding of the variation between the two datasets.

    Parameters:
    y_test: A set of original target values
    y_pred: A set of predicted target values

    Returns:
    Generates a scatter plot of True vs Predicted values to get an overview of the error rate of the model.
    """

    # Plotting the results
    plt.figure(figsize=(10, 6))

    # Plotting the relationship between true and predicted values for each target variable
    for i in range(y_test.shape[1]):
        plt.scatter(y_test[:, i], y_pred[:, i], label=f"Target {i+1}", alpha=0.005, color='k')

    # A red straight line is plotted across the centre to show how close to the correct value the predicted value is 
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal Line')

    plt.title('True vs Predicted Values for Multiple Target Variables')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    # plt.legend()
    plt.grid(True)
    plt.show()