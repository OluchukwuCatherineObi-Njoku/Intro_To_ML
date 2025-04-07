import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Step 1: Loading Data

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

# Fill in the line below to read the file into a variable home_data
home_data = pd.read_csv(iowa_file_path)


# Step 2: Review The Data

# Print summary statistics in next line
home_data.describe()

# I inputted given values myself to test based on the results of describe
# What is the average lot size (rounded to nearest integer)?
avg_lot_size = 10517

# As of today, how old is the newest home (current year - the date in which it was built)
newest_home_age = 15

# Step 3: Specify Prediction Target. This is the value you want your model to predict, and to test it, it is based on
# avail values in your dataset
# So this model is to predict house prices, my training data has prices, so we need to specify the column as the
# target value to emulate based on given features



# 1 ------------- Model Training without splitting data into test and training data --------1

# To find out what target columns are available, you print it out'

home_data.columns

#then you select a target column that is the type of values that should be returned
y = home_data.SalePrice

# Step 4: Create X
# Now you will create a DataFrame called X holding the predictive features you want to determine the prices or target value.

# Create the list of features below
feature_names = ["LotArea","YearBuilt","1stFlrSF","2ndFlrSF","FullBath","BedroomAbvGr","TotRmsAbvGrd"]

# Select data corresponding to features in feature_names
X = home_data[feature_names]

print(X.head())

# Building your model requires sklearn
#  DecisionTreeRegressor is the name of our model

#Step 5: Specify and Fit/Train Model

#For model reproducibility, set a numeric value for random_state when specifying the model
iowa_model = DecisionTreeRegressor(random_state=1) #the val chosen mostly does not matter, but one must be there

# Fit/Train the model with data. Giving it the features, and their corresponding answwers
iowa_model.fit(X,y)

#Step 6: Make Predictions based on the data

predictions = iowa_model.predict(X)
print(predictions)

#The above makes predictions of what y is supposed to be, given X. If X has 5 houses, it predicts the prices for all 5 houses

print(y.head())
print(predictions)

# End of 1 ------------- Model Training without splitting data into test and training data --------1

# Model Validation

# Start 2 ---- splitting data into training and validation data then perf model validation

# Import the train_test_split function and uncomment
from sklearn.model_selection import train_test_split

# fill in and uncomment
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)

# Specify and fit the model

# Specify the model
iowa_model = DecisionTreeRegressor(random_state=1)

# Fit iowa_model with the training data.
iowa_model.fit(train_X, train_y)

# make predictions with validation data

# Predict with all validation observations
val_predictions = iowa_model.predict(val_X)

#calculate mae

val_mae = mean_absolute_error(val_y, val_predictions);

# uncomment following line to see the validation_mae
print(val_mae)

# function to retrieve MAE

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# make comparision of MAE at different tree depth, the least MAE is the best

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
for max_l_node in candidate_max_leaf_nodes:
    mae = get_mae(max_l_node,train_X,val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_l_node, mae))

# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
best_tree_size = 100

#Fit Model Using All Data now that we know the ideal tree height

# Fill in argument to make optimal size and uncomment
final_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)

# fit the final model and uncomment the next two lines
final_model.fit(X,y)

#-----------------------------------------------------------------------------------#
# Code you have previously used to load data
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)
# Create target object and call it y
y = home_data.SalePrice
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify Model
iowa_model = DecisionTreeRegressor(random_state=1)
# Fit Model
iowa_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

# Using best value for max_leaf_nodes
iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
iowa_model.fit(train_X, train_y)
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))
#-----------------------------------------------------------------------------------#

# Use a Random Forest

from sklearn.ensemble import RandomForestRegressor

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)

# fit your model
rf_model.fit(train_X, train_y)

# Calculate the mean absolute error of your Random Forest model on the validation data
val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(val_y,val_predictions)

print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))


