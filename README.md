# josephite

**Importing Libraries:**

numpy and pandas are libraries for numerical and data analysis, respectively.
train_test_split is a function from sklearn.model_selection used for splitting data into training and testing sets.
LinearRegression is a class from sklearn.linear_model for performing linear regression.
mean_squared_error and r2_score are functions from sklearn.metrics used for evaluating the performance of regression models.

**Loading Data:**

data = pd.read_csv(csv_file_path) reads the data from a CSV file into a pandas DataFrame.
print("Column names:", data.columns) prints out the column names of the DataFrame.

**Preparing Data:**

X contains the input features for the model, which are 'AT' (Temperature), 'AP' (Ambient Pressure), 'RH' (Relative Humidity), and 'V' (Exhaust Vacuum).
y contains the target variable, which is 'PE' (Energy Output).
train_test_split function splits the data into training and testing sets. Here, 80% of the data is used for training (X_train and y_train) and 20% for testing (X_test and y_test).

**Training the Model:**

A linear regression model is initialized using LinearRegression() and then trained on the training data using model.fit(X_train, y_train).

**Making Predictions:**

Predictions are made on the testing data using model.predict(X_test).

**Evaluating Model Performance:**

Mean Squared Error (MSE) and Coefficient of Determination (R²) are calculated to evaluate how well the model fits the data.
MSE measures the average squared difference between the actual and predicted values.
R² represents the proportion of the variance in the dependent variable that is predictable from the independent variables.
The calculated MSE and R² are then printed out.

**Feature Importance:**

The coefficients of the linear regression model represent the importance of each feature in predicting the target variable.
These coefficients are printed out along with the corresponding feature names to show how each feature contributes to the prediction.
