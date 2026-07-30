# Linear Regression Demo (`mlmy.py`)

The original contents of this repository: a small scikit-learn example that
fits a linear regression model to a power-plant dataset.

**Importing Libraries**

`numpy` and `pandas` are libraries for numerical and data analysis,
respectively. `train_test_split` from `sklearn.model_selection` splits data
into training and testing sets. `LinearRegression` from `sklearn.linear_model`
performs linear regression. `mean_squared_error` and `r2_score` from
`sklearn.metrics` evaluate model performance.

**Loading Data**

`data = pd.read_csv(csv_file_path)` reads the data from a CSV file into a
pandas DataFrame. `print("Column names:", data.columns)` prints the column
names.

**Preparing Data**

`X` contains the input features — `AT` (Temperature), `AP` (Ambient Pressure),
`RH` (Relative Humidity), and `V` (Exhaust Vacuum). `y` is the target variable,
`PE` (Energy Output). `train_test_split` uses 80% of the data for training and
20% for testing.

**Training the Model**

A linear regression model is initialised with `LinearRegression()` and trained
with `model.fit(X_train, y_train)`.

**Making Predictions**

Predictions are made on the test data with `model.predict(X_test)`.

**Evaluating Model Performance**

Mean Squared Error (MSE) and the Coefficient of Determination (R²) are
calculated. MSE measures the average squared difference between actual and
predicted values; R² is the proportion of variance in the dependent variable
predictable from the independent variables.

**Feature Importance**

The model's coefficients represent each feature's contribution to the
prediction and are printed alongside their feature names.
