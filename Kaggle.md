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
