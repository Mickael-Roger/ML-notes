# Introduction
## First ML Model

### Prepare dataset

```python
import pandas as pd

melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 

# Display all columns
melbourne_data.columns   

# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)

# By convention, the prediction target is called y
y = melbourne_data.Price

# By convention, X represents the features used
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]

# Give a quick review of the data used (Sum, number of data, avg, 25/50/75 percentiles, min, max, std)
X.describe()

# Display firsts rows
X.head()

```

### Build the model

```python
from sklearn.tree import DecisionTreeRegressor

# First step: Define model.
melbourne_model = DecisionTreeRegressor(random_state=1)

# Second step : Fit (Capture patterns from provided data. This is the heart of modeling)
melbourne_model.fit(X, y)

# Thrid step : Predict (We normally should use a different dataset)
print(melbourne_model.predict(X.head()))    # Predict for the first 5 house in the dataset

# Forth setp : Evaluate

```

### Validate the model

Mean Absolute Error (MAE) is one of the metric used to evaluate a model quality.

error = actual - predicted -> Average error per prediction

```python
from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)

```

### Split data into train and test dataset 

```python
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

```

## Definitions
### Overfitting
A model matches the training data almost perfectly, but does poorly in validation and other new data

### Underfitting
When a model fails to capture important distinctions and patterns in the data, so it performs poorly even in training data, that is called underfitting.

### Model evaluation
We look for the best value between underfitting and overfitting.
![Overfitting vs Underfitting](http://i.imgur.com/2q85n9s.png "")

### DecisionTreeRegressor
This is a decision tree. The more level (leaves) we use in the decision tree, the more we move from underfitting area to overfitting area. For instance, if we have only one level, it could be a simple tree like this:

Is the house has more than 2 bed? Yes, price is x. No price is y

### Overfitting vs Underfitting

- Overfitting: capturing spurious patterns that won't recur in the future, leading to less accurate predictions, or
- Underfitting: failing to capture relevant patterns, again leading to less accurate predictions.

















