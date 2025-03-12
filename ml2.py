


import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error , r2_score , mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_ridge import KernelRidge





def compute_cost(X, y, theta):
    m = len(y)
    cost = (1 / (2 * m)) * np.sum(np.square(X.dot(theta) - y))
    return cost

def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        gradients = X.T.dot(X.dot(theta) - y) / m
        theta = theta - learning_rate * gradients
        cost_history[i] = compute_cost(X, y, theta)

    return theta, cost_history



# Load the dataset
df = pd.read_csv('C:\\Users\\laptop\\Desktop\\ml2\\cars.csv')

# Data Cleaning
df['horse_power'] = df['horse_power'].replace(r'^\D*$', np.nan, regex=True)
df.dropna(subset=['horse_power'], inplace=True)

mask = df['top_speed'].str.contains(r'^\D*$', regex=True)
df.drop(df[mask].index, inplace=True)
df.reset_index(drop=True, inplace=True)

string_columns = df.select_dtypes(include=['object']).columns
df[string_columns] = df[string_columns].apply(lambda x: x.str.strip())

pattern = r'([A-Z]+)\s([\d,]+)'
extracted = df['price'].str.extract(pattern, expand=True)
extracted.columns = ['currency', 'amount']

df = pd.concat([df, extracted], axis=1)
df.drop(columns='price', inplace=True)
df.rename(columns={'amount': 'price'}, inplace=True)

df['price'] = df['price'].str.replace(',', '').astype(float)

country_to_currency = {
    'ksa': 'SAR',
    'egypt': 'EGP',
    'bahrain': 'BHD',
    'qatar': 'QAR',
    'oman': 'OMR',
    'kuwait': 'KWD',
    'uae': 'AED'
}

currency_to_usd = {
    'SAR': 0.27,
    'EGP': 0.032,
    'BHD': 2.65,
    'QAR': 0.27,
    'OMR': 2.60,
    'KWD': 3.30,
    'AED': 0.27
}

df['currency'] = df['country'].map(country_to_currency)
df['price'] = df.apply(lambda row: row['price'] * currency_to_usd[row['currency']], axis=1)

mean_prices = df.groupby('car name')['price'].transform('mean')
df['price'] = df['price'].fillna(mean_prices)

median_na = df.groupby('brand')['price'].transform('median')
df['price'] = df['price'].fillna(median_na)
df.dropna(subset=['price'], inplace=True)

df.drop(columns='currency', inplace=True)
df.reset_index(drop=True, inplace=True)

indices = df[df['top_speed'].str.contains('Seater')].index
temp = df.loc[indices, 'seats']
df.loc[indices, 'seats'] = df['top_speed']
df.loc[indices, 'top_speed'] = temp

columns = ['seats', 'engine_capacity', 'brand', 'top_speed', 'cylinder', 'horse_power']

def get_mode(series):
    mode = series.mode()
    return mode[0] if not mode.empty else None

df[columns] = df.groupby('car name')[columns].transform(get_mode)

df['seats'] = df['seats'].replace(r'^\D*$', np.nan, regex=True)

def extract_seats(value):
    if pd.isna(value):
        return value
    if type(value) == float:
        return int(value)
    if 'Seater' in value or 'Seats' in value:
        return int(value.split()[0])

df['seats'] = df['seats'].apply(extract_seats)

median = df.groupby('brand')['seats'].transform('median')
df['seats'] = df['seats'].fillna(median)
df.dropna(subset=['seats'], inplace=True)

df['seats'] = df['seats'].astype(int)
df['cylinder'] = df['cylinder'].replace('N/A, Electric', 0)

df['top_speed'] = df['top_speed'].astype(float)
df['cylinder'] = df['cylinder'].astype(float)
df['horse_power'] = df['horse_power'].astype(int)
df['engine_capacity'] = df['engine_capacity'].astype(float)

def find_cylinder_count(engine_capacity):
    if engine_capacity < 1.5:
        return 3
    elif engine_capacity < 2:
        return 4
    elif engine_capacity < 2.5:
        return 5
    elif engine_capacity < 4:
        return 6
    elif engine_capacity < 6:
        return 8
    elif engine_capacity < 8:
        return 12

df['cylinder'] = df['cylinder'].fillna(df['engine_capacity'].apply(find_cylinder_count))

mask = df[df['engine_capacity'] >= 900]
df.loc[mask.index, 'engine_capacity'] = df.loc[mask.index, 'engine_capacity'] / 1000

mask = df[(df['engine_capacity'] >= 100) & (df['engine_capacity'] < 500)]
df.loc[mask.index, 'engine_capacity'] = df.loc[mask.index, 'engine_capacity'] / 100

df['cylinder'] = df['cylinder'].fillna(df['engine_capacity'].apply(find_cylinder_count))
df['cylinder'] = df['cylinder'].astype(int)

def extract_valid_year(year):
    if len(year) == 1:
        return year[0]
    elif len(year) == 2:
        if int(year[0]) > 1900 and int(year[0]) < 2025:
            return year[0]
        elif int(year[1]) > 1900 and int(year[1]) < 2025:
            return year[1]

years = df['car name'].str.findall(r'\b\d{4}\b')
df['year'] = years.apply(extract_valid_year)
df['year'] = df['year'].astype(int)

df.drop(columns='car name', inplace=True)
df.reset_index(drop=True, inplace=True)

df.loc[df['horse_power'] < 50, 'horse_power'] = df.groupby('brand')['horse_power'].transform('median')
df.loc[(df['top_speed'] < 50) | (df['top_speed'] > 500), 'top_speed'] = df.groupby('brand')['top_speed'].transform('median')

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Assuming `df` is the original dataset
# Frequency Encoding for brand and country
brand_freq = df['brand'].value_counts().to_dict()
country_freq = df['country'].value_counts().to_dict()

df['brand_freq'] = df['brand'].map(brand_freq)
df['country_freq'] = df['country'].map(country_freq)

# Drop the original categorical columns
df.drop(columns=['brand', 'country'], inplace=True)

# Normalization (excluding 'price')
numerical_features = ['engine_capacity', 'cylinder', 'horse_power', 'top_speed', 'seats', 'year', 'brand_freq', 'country_freq']
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(df[numerical_features])

# Normalization function
def normalize(df, numerical_features, scaler):
    scaled_nums = scaler.transform(df[numerical_features])
    scaled_nums_df = pd.DataFrame(scaled_nums, columns=numerical_features)
    df.drop(columns=numerical_features, inplace=True)
    df_normalized = pd.concat([df.reset_index(drop=True), scaled_nums_df.reset_index(drop=True)], axis=1)
    return df_normalized

# Normalize the entire dataset
df = normalize(df, numerical_features, scaler)

# Save the cleaned dataset
df.to_csv('C:\\Users\\laptop\\Desktop\\ml2\\clean_cars.csv', index=False)

# Splitting the cleaned dataset
train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Save the splits
train_df.to_csv('C:\\Users\\laptop\\Desktop\\ml2\\train_cars.csv', index=False)
val_df.to_csv('C:\\Users\\laptop\\Desktop\\ml2\\val_cars.csv', index=False)
test_df.to_csv('C:\\Users\\laptop\\Desktop\\ml2\\test_cars.csv', index=False)

# Print the sizes of each set to confirm the split
print(f'Training set size: {train_df.shape[0]}')
print(f'Validation set size: {val_df.shape[0]}')
print(f'Test set size: {test_df.shape[0]}')

# Assuming train_df has been prepared with features and target
X = train_df.drop(columns=['price']).values
y = train_df['price'].values

X_val = val_df.drop(columns=['price']).values
y_val = val_df['price'].values

# Add a bias term (intercept)
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# Closed-form solution
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)


# Initialize parameters
theta = np.random.randn(X_b.shape[1])
iterations = 1000
learning_rate = 0.01

theta, cost_history = gradient_descent(X_b, y, theta, learning_rate, iterations)

# Predictions
y_pred_closed_form = X_b.dot(theta_best)
y_pred_gradient_descent = X_b.dot(theta)

# Calculate RMSE
mse_closed_form = mean_squared_error(y, y_pred_closed_form)
mae_closed_form_val = mean_absolute_error(y, y_pred_closed_form)
r2_closed_form_val = r2_score(y, y_pred_closed_form)

mse_gradient_descent = mean_squared_error(y, y_pred_gradient_descent)
mae_gradient_descent = mean_absolute_error(y, y_pred_gradient_descent)
r2_gradient_descent = r2_score(y, y_pred_gradient_descent)


print(f"\nClosed-form: MSE = {mse_closed_form:.4f}, MAE = {mae_closed_form_val:.4f}, R_sequared = {r2_closed_form_val:.4f}\n")
print(f"Gradient Descent: MSE = {mse_gradient_descent:.4f}, MAE = {mae_gradient_descent:.4f}, R_sequared = {r2_gradient_descent:.4f}")



# --- LASSO Regression (Manual Implementation) ---
class LassoRegression:
    def __init__(self, alpha=0.1, n_iterations=1000, learning_rate=0.01):
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.theta = np.random.randn(X_b.shape[1])
        m = X_b.shape[0]

        for _ in range(self.n_iterations):
            gradients = 2 / m * X_b.T.dot(X_b.dot(self.theta) - y) + self.alpha * np.sign(self.theta)
            self.theta -= self.learning_rate * gradients

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta)


# Train LASSO Regression
lasso = LassoRegression(alpha=0.1, learning_rate=0.1, n_iterations=1000)
lasso.fit(X, y)
y_pred_lasso = lasso.predict(X)

# Evaluate LASSO Regression
mse_lasso = mean_squared_error(y, y_pred_lasso)
mae_lasso_val = mean_absolute_error(y, y_pred_lasso)
r2_lasso = r2_score(y, y_pred_lasso)


print(f"\nLasso Regression: MSE = {mse_lasso:.4f}, MAE = {mae_lasso_val:.4f}, R_sequared = {r2_lasso:.4f}")



# --- Ridge Regression (Manual Implementation) ---
class RidgeRegression:
    def __init__(self, alpha=0.1, n_iterations=1000, learning_rate=0.01):
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.theta = np.random.randn(X_b.shape[1])
        m = X_b.shape[0]

        for _ in range(self.n_iterations):
            gradients = 2 / m * X_b.T.dot(X_b.dot(self.theta) - y) + self.alpha * self.theta
            self.theta -= self.learning_rate * gradients

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta)


# Train Ridge Regression
ridge = RidgeRegression(alpha=0.1, learning_rate=0.1, n_iterations=1000)
ridge.fit(X, y)
y_pred_ridge = ridge.predict(X)

mse_ridge_val = mean_squared_error(y, y_pred_ridge)
mae_ridge_val = mean_absolute_error(y, y_pred_ridge)
r2_ridge_val = r2_score(y, y_pred_ridge)


print(f"\nRidge Regression: MSE = {mse_ridge_val:.4f}, MAE = {mae_ridge_val:.4f}, R_sequared = {r2_ridge_val:.4f}\n")


# Polynomial Regression (varying polynomial degree from 2 to 10)
polynomial_degrees = [2, 3, 4, 5, 6]
poly_rmse = []

for degree in polynomial_degrees:
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    
    # Closed-form solution for Polynomial Regression
    theta_poly = np.linalg.inv(X_poly.T.dot(X_poly)).dot(X_poly.T).dot(y)
    y_pred_poly = X_poly.dot(theta_poly)

    mse_poly_val = mean_squared_error(y, y_pred_poly)
    mae_poly_val = mean_absolute_error(y, y_pred_poly)
    r2_poly_val = r2_score(y, y_pred_poly)

    poly_rmse.append((degree, mse_poly_val, mae_poly_val, r2_poly_val))
    print(f"Polynomial Regression (degree {degree}): MSE = {mse_poly_val:.4f}, MAE = {mae_poly_val:.4f}, R_sequared = {r2_poly_val:.4f}")

  
# Kernel Ridge Regression with RBF kernel
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
rbf_ridge = KernelRidge(kernel='rbf')
rbf_ridge_cv = GridSearchCV(rbf_ridge, param_grid, cv=5)
rbf_ridge_cv.fit(X, y)
rbf_ridge_best = rbf_ridge_cv.best_estimator_

# Get predictions for Kernel Ridge Regression
y_pred_krr = rbf_ridge_best.predict(X)

# Calculate performance metrics for KRR
mse_rbf_ridge_val = mean_squared_error(y, y_pred_krr)
mae_rbf_ridge_val = mean_absolute_error(y, y_pred_krr)
r2_rbf_ridge_val = r2_score(y, y_pred_krr)

print(f"\nRBF Kernel Ridge Regression: MSE = {mse_rbf_ridge_val:.4f}, MAE = {mae_rbf_ridge_val:.4f}, R_squared = {r2_rbf_ridge_val:.4f}")

# Forward Selection Implementation
def forward_selection(X_train, X_val, y_train, y_val, max_features=5):
    selected_features = []
    best_r2 = -np.inf  # Start with the worst possible R² value
    remaining_features = list(range(X_train.shape[1]))  # All feature indices
    best_model = None
    
    for _ in range(min(max_features, X_train.shape[1])):
        best_feature = None
        for feature in remaining_features:
            # Select features for the current iteration
            current_features = selected_features + [feature]
            X_train_current = X_train[:, current_features]
            X_val_current = X_val[:, current_features]
            
            # Add a bias term (intercept)
            X_train_current_b = np.c_[np.ones((X_train_current.shape[0], 1)), X_train_current]
            X_val_current_b = np.c_[np.ones((X_val_current.shape[0], 1)), X_val_current]

            # Train model using closed-form solution (or any other method you prefer)
            theta_best = np.linalg.inv(X_train_current_b.T.dot(X_train_current_b)).dot(X_train_current_b.T).dot(y_train)
            y_pred_val = X_val_current_b.dot(theta_best)

            # Evaluate the model performance
            mse = mean_squared_error(y_val, y_pred_val)
            r2 = r2_score(y_val, y_pred_val)

            # Update the best feature if the model performance improves
            if r2 > best_r2:
                best_r2 = r2
                best_feature = feature
                best_model = theta_best

        # Add the best feature to the selected features
        if best_feature is not None:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
        else:
            break  # Stop if no feature improves performance

    # Return the final selected features and the model
    return selected_features, best_model

# Perform forward selection
selected_features, best_model = forward_selection(X, X_val, y, y_val)

# Display the selected features
print(f"\nSelected features: {selected_features}")

# Predict with the selected features
X_train_selected = X[:, selected_features]
X_val_selected = X_val[:, selected_features]
X_train_selected_b = np.c_[np.ones((X_train_selected.shape[0], 1)), X_train_selected]
X_val_selected_b = np.c_[np.ones((X_val_selected.shape[0], 1)), X_val_selected]

# Final prediction with the selected features
y_pred_final = X_val_selected_b.dot(best_model)

# Evaluate the final model
mse_final = mean_squared_error(y_val, y_pred_final)
mae_final = mean_absolute_error(y_val, y_pred_final)
r2_final = r2_score(y_val, y_pred_final)

print(f"Forward Selection Final Model: MSE = {mse_final:.4f}, MAE = {mae_final:.4f}, R_squared = {r2_final:.4f}")

print("\n")













# Find the degree with the minimum MSE in poly_results
min_poly_mse_degree, min_poly_mse, _, _ = min(poly_rmse, key=lambda x: x[1])

# Now store only the minimum MSE from Polynomial Regression in the models dictionary
models = {
    'Closed-form': {'MSE': mse_closed_form, 'R-squared': r2_closed_form_val},
    'Gradient Descent': {'MSE': mse_gradient_descent, 'R-squared': r2_gradient_descent},
    'Lasso Regression': {'MSE': mse_lasso, 'R-squared': r2_lasso},
    'Ridge Regression': {'MSE': mse_ridge_val, 'R-squared': r2_ridge_val},
    'Polynomial Regression': min_poly_mse,  # Store only the minimum MSE from Polynomial Regression
    'Kernel Ridge': {'MSE': mse_rbf_ridge_val, 'R-squared': r2_rbf_ridge_val},
}

# Find the best model by MSE
best_model_by_mse = min(models.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else x[1]['MSE'])

# If the best model is Polynomial Regression, print the degree and MSE
if best_model_by_mse[0] == 'Polynomial Regression':
    print(f"Best model by MSE: {best_model_by_mse[0]} with MSE = {best_model_by_mse[1]} (Degree: {min_poly_mse_degree})")
else:
    print(f"Best model by MSE: {best_model_by_mse[0]} with MSE = {best_model_by_mse[1]}")


print("\n")

# Polynomial Regression for degree 2
degree_2 = 2
poly_features_2 = PolynomialFeatures(degree=degree_2)
X_poly_2 = poly_features_2.fit_transform(X)  # Training data

# Apply the same transformation to the test set
X_poly_2_test = poly_features_2.transform(X_val)  # Test data

# Closed-form solution for Polynomial Regression (degree 2)
theta_poly_2 = np.linalg.inv(X_poly_2.T.dot(X_poly_2)).dot(X_poly_2.T).dot(y)
y_pred_poly_2 = X_poly_2.dot(theta_poly_2)
y_pred_poly_2_test = X_poly_2_test.dot(theta_poly_2)

# Calculate MSE, MAE, and R-squared for the degree 2 model on the test set
mse_poly_2_test = mean_squared_error(y_val, y_pred_poly_2_test)
mae_poly_2_test = mean_absolute_error(y_val, y_pred_poly_2_test)
r2_poly_2_test = r2_score(y_val, y_pred_poly_2_test)

print(f"\nPolynomial Regression (degree 2) - Test Set: MSE = {mse_poly_2_test:.4f}, MAE = {mae_poly_2_test:.4f}, R-squared = {r2_poly_2_test:.4f}")


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import PolynomialFeatures

# Assuming train_df and val_df have been prepared with features and target
X_train = train_df.drop(columns=['price']).values  # Features
y_train = train_df['price'].values  # Target variable
X_val = val_df.drop(columns=['price']).values  # Validation features
y_val = val_df['price'].values  # Validation target

# Fit models (Lasso, Ridge, Polynomial, Kernel Ridge)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_val)

ridge = Ridge(alpha=0.1)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_val)

# Polynomial Regression (using degree=2 for example)
poly_degree = 2
poly_features = PolynomialFeatures(degree=poly_degree)
X_train_poly = poly_features.fit_transform(X_train)
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
X_val_poly = poly_features.transform(X_val)
y_pred_poly = poly_model.predict(X_val_poly)

# Kernel Ridge Regression (using RBF kernel for example)
krr = KernelRidge(alpha=1.0, kernel='rbf')
krr.fit(X_train, y_train)
y_pred_krr = krr.predict(X_val)

# --- 1. Error Distribution (Residuals) Plot ---
# Lasso
residuals_lasso = y_val - y_pred_lasso
sns.histplot(residuals_lasso, kde=True, label="Lasso", color='blue')
plt.title('Error Distribution (Residuals) - Lasso Regression')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Ridge
residuals_ridge = y_val - y_pred_ridge
sns.histplot(residuals_ridge, kde=True, label="Ridge", color='green')
plt.title('Error Distribution (Residuals) - Ridge Regression')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Polynomial
residuals_poly = y_val - y_pred_poly
sns.histplot(residuals_poly, kde=True, label="Polynomial", color='orange')
plt.title('Error Distribution (Residuals) - Polynomial Regression')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Kernel Ridge
residuals_krr = y_val - y_pred_krr
sns.histplot(residuals_krr, kde=True, label="Kernel Ridge", color='purple')
plt.title('Error Distribution (Residuals) - Kernel Ridge Regression')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# --- 2. Feature Importance Plot (For Linear Models like Lasso and Ridge) ---
# For Lasso and Ridge: feature importance is the absolute value of the coefficients
lasso_feature_importance = np.abs(lasso.coef_)  # Lasso coefficients
ridge_feature_importance = np.abs(ridge.coef_)  # Ridge coefficients

# Polynomial Feature Importance (including interaction terms)
poly_feature_importance = np.abs(poly_model.coef_)

# Assuming the feature names are the same as in the original dataset
features = train_df.drop(columns=['price']).columns

# Lasso Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(features, lasso_feature_importance)
plt.title('Feature Importance - Lasso Regression')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Ridge Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(features, ridge_feature_importance)
plt.title('Feature Importance - Ridge Regression')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Polynomial Feature Importance (if applicable)
plt.figure(figsize=(10, 6))
plt.barh(poly_features.get_feature_names_out(input_features=features), poly_feature_importance)
plt.title('Feature Importance - Polynomial Regression')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# --- 3. Predictions vs. Actual Values Plot ---
# Plot for Lasso model
plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_pred_lasso, color='blue', label='Lasso Predictions')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], color='red', linestyle='--', label='Perfect Prediction')
plt.title('Predictions vs. Actual - Lasso Regression')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()

# Plot for Ridge model
plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_pred_ridge, color='green', label='Ridge Predictions')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], color='red', linestyle='--', label='Perfect Prediction')
plt.title('Predictions vs. Actual - Ridge Regression')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()

# Plot for Polynomial model
plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_pred_poly, color='orange', label='Polynomial Predictions')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], color='red', linestyle='--', label='Perfect Prediction')
plt.title('Predictions vs. Actual - Polynomial Regression')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()

# Plot for Kernel Ridge model
plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_pred_krr, color='purple', label='Kernel Ridge Predictions')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], color='red', linestyle='--', label='Perfect Prediction')
plt.title('Predictions vs. Actual - Kernel Ridge Regression')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()
