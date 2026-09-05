import numpy as np
import pandas as pd

data = pd.read_csv(
    '/Users/blitz/Documents/aiMinorChallenge/REHAB/Rehab_exercise/d03_feature_data/rehab_exercise_features.csv'
)

feature_columns = data.columns[8:]
data.head()

train_data = data[data['split'] == 'train']
test_data = data[data['split'] == 'test']

X_train = train_data[feature_columns].to_numpy()
y_train = train_data['movement_id'].to_numpy()

X_test = test_data[feature_columns].to_numpy()
y_test = test_data['movement_id'].to_numpy()

print('Training examples:', len(X_train))
print('Test examples:', len(X_test))

mean = X_train.mean(axis=0)
standard_deviation = X_train.std(axis=0)

X_train = (X_train - mean) / (standard_deviation + 0.000001)
X_test = (X_test - mean) / (standard_deviation + 0.000001)

centroids = []

for movement_id in range(16):
    movement_examples = X_train[y_train == movement_id]
    centroid = movement_examples.mean(axis=0)
    centroids.append(centroid)

centroids = np.array(centroids)
print('Centroids created:', len(centroids))

predictions = []

for example in X_test:
    distances = []

    for centroid in centroids:
        distance = np.sqrt(np.sum((example - centroid) ** 2))
        distances.append(distance)

    predicted_movement = np.argmin(distances)
    predictions.append(predicted_movement)

predictions = np.array(predictions)

correct_predictions = predictions == y_test
accuracy = correct_predictions.mean()

print('Accuracy:', round(accuracy * 100, 2), '%')