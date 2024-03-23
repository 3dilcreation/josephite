
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


csv_file_path = r"C:\Users\harib\Downloads\train.csv"
data = pd.read_csv(csv_file_path)
print("Column names:", data.columns)


X = data[['AT', 'AP', 'RH', 'V']] 
y = data['PE']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Coefficient of Determination (R²): {r2}')


feature_importance = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print('\nFeature Importance:')
print(feature_importance)
