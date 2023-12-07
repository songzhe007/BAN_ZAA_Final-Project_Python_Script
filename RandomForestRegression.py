import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


file_path = '/kaggle/input/spotify-songs/spotify_songs.csv'
df = pd.read_csv(file_path)

le = LabelEncoder()
df['track_artist'] = le.fit_transform(df['track_artist'])


features = ['danceability', 'energy', 'loudness', 'duration_ms', 'speechiness', 'acousticness', 
            'track_artist', 'instrumentalness', 'tempo' ]
target = 'track_popularity'

X = df[features]
y = df[target]

# Set 70% of dataset as train_set and the remaining portion serves as test_set
train_size = 0.7
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, random_state=42)

# Perform Random Forest Regression on model training & serving
model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)




# Calculate the mean squared error (MSE) of our model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on Test Data: {mse}')

# Calculate the R-squared Value
r2 = r2_score(y_test, y_pred)
print(f'R-squared: {r2}')

feature_importances = model.feature_importances_
for feature, importance in zip(features, feature_importances):
    print(f'Feature {feature}: Importance = {importance}')


sample_indices = np.random.choice(len(y_test), 100, replace=False)

# Calculate the Residual value & Define the threshold (1.05 * std of residuals)
residuals = y_test.iloc[sample_indices] - y_pred[sample_indices]
threshold = 1.05 * np.std(residuals)

outlier_indices = np.abs(residuals) > threshold

# Plot the scater point graph
plt.scatter(y_test.iloc[sample_indices][~outlier_indices], y_pred[sample_indices][~outlier_indices], label='Predictions')
plt.xlabel('True Values')
plt.ylabel('Predictions')

# Plot the trend line
z = np.polyfit(y_test.iloc[sample_indices], y_pred[sample_indices], 1)
p = np.poly1d(z)
plt.plot([min(y_test.iloc[sample_indices]), max(y_test.iloc[sample_indices])], [p(min(y_test.iloc[sample_indices])), p(max(y_test.iloc[sample_indices]))], color='red', linewidth=2, linestyle='dashed', label='Trendline')

plt.title('True Values vs Predictions with Trendline')
plt.legend()
plt.show()
