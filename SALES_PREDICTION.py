import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

DATA_FILE = os.path.join(os.path.dirname(__file__), 'advertising.csv')
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f'Dataset not found: {DATA_FILE}')

df = pd.read_csv(DATA_FILE)

print('First 5 rows:')
print(df.head(), end='\n\n')

print('Dataset info:')
df.info()
print()

print('Descriptive statistics:')
print(df.describe(), end='\n\n')

sns.set_style('whitegrid')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.scatterplot(x='TV', y='Sales', data=df, ax=axes[0])
axes[0].set_title('TV vs Sales')
axes[0].set_xlabel('TV Advertising (in thousands)')
axes[0].set_ylabel('Sales (in thousands)')

sns.scatterplot(x='Radio', y='Sales', data=df, ax=axes[1])
axes[1].set_title('Radio vs Sales')
axes[1].set_xlabel('Radio Advertising (in thousands)')
axes[1].set_ylabel('Sales (in thousands)')

sns.scatterplot(x='Newspaper', y='Sales', data=df, ax=axes[2])
axes[2].set_title('Newspaper vs Sales')
axes[2].set_xlabel('Newspaper Advertising (in thousands)')
axes[2].set_ylabel('Sales (in thousands)')

plt.tight_layout()
plt.show()

X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}\n')

model = LinearRegression()
model.fit(X_train, y_train)
print('Linear Regression model trained successfully.\n')

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'R-squared (R2): {r2:.2f}\n')

results_df = pd.DataFrame({'Actual Sales': y_test, 'Predicted Sales': y_pred})

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Actual Sales', y='Predicted Sales', data=results_df)
plt.title('Actual vs. Predicted Sales')
plt.xlabel('Actual Sales (in thousands)')
plt.ylabel('Predicted Sales (in thousands)')

max_val = max(y_test.max(), y_pred.max())
min_val = min(y_test.min(), y_pred.min())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')

plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(results_df['Actual Sales'], color='blue', label='Actual Sales', kde=True, stat='density', linewidth=0.5)
sns.histplot(results_df['Predicted Sales'], color='lightblue', label='Predicted Sales', kde=True, stat='density', linewidth=0.5, alpha=0.7)
plt.title('Distribution of Actual vs. Predicted Sales')
plt.xlabel('Sales (in thousands)')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

