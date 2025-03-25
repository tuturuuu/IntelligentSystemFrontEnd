import os
import numpy as np
import pytz
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
from datetime import datetime
import warnings
from sklearn.metrics import (r2_score, max_error, mean_absolute_percentage_error,
                            explained_variance_score, mean_squared_error,
                            mean_squared_log_error, mean_poisson_deviance,
                            mean_absolute_error, root_mean_squared_error)
import tensorflow as tf
import keras
import keras.backend as K
from keras.layers import Input, GRU, Dense, Dropout
from keras.models import Model
from tensorflow_probability.python.layers import DistributionLambda
from tensorflow_probability.python.distributions import Normal

# Print package versions for debugging
import pkg_resources
for package in ['tensorflow', 'numpy', 'pandas', 'matplotlib',
                'scikit-learn', 'tensorflow-probability', 'keras']:
    try:
        version = pkg_resources.get_distribution(package).version
        print(f"{package}=={version}")
    except pkg_resources.DistributionNotFound:
        print(f"{package} is not installed")

# Utility functions
iran_timezone = pytz.timezone(zone='Iran')
dateformat = '%Y-%m-%d %H:%M:%S'  # Include seconds in the format

def normalize(x_, minmax):
    """Normalizes input data within the specified min-max range."""
    return (x_ - minmax[0]) / (minmax[1] - minmax[0])

def denormalize(x_, minmax):
    """Denormalizes input data based on the specified min-max range."""
    return minmax[0] + ((minmax[1] - minmax[0]) * x_)

def create_probabilistic_gru_model(input_shape=(6, 1)):
    """
    Creates a Probabilistic GRU model similar to the one you were trying to load.
    
    Args:
        input_shape: Shape of input sequences (timesteps, features)
        
    Returns:
        Compiled Keras model
    """
    # Input layer
    inputs = Input(shape=input_shape)
    
    # GRU layers
    x = GRU(64, return_sequences=True)(inputs)
    x = GRU(32)(x)
    
    # Dense layers
    x = Dense(16)(x)
    x = Dropout(0.2)(x)
    
    # Probabilistic output - predicts mean and standard deviation
    mu = Dense(1)(x)
    sigma = Dense(1, activation='softplus')(x)
    
    # Create probabilistic distribution output
    dist = DistributionLambda(
        lambda params: Normal(loc=params[0], scale=params[1]),
        output_shape=(1,)
    )([mu, sigma])
    
    # Create and compile model
    model = Model(inputs=inputs, outputs=dist)
    model.compile(
        loss=lambda y_true, y_pred: -y_pred.log_prob(y_true),
        optimizer='adam'
    )
    
    return model

def plot_prediction(y_train, y_test, y_pred, model_name=None):
    """Generate a plot to visualize the results of time series prediction."""
    # Set up plot parameters
    plt.rcParams.update({'figure.figsize': [18, 6]})

    # Handle dates for plotting
    try:
        # For datetime index
        dates = [mdate.date2num(datetime.fromtimestamp(datetime.strptime(str(d), dateformat).timestamp())) 
                for d in data.index[-len(y_test):]]
    except:
        # Fallback for numeric index
        dates = list(range(len(y_test)))

    fig, ax = plt.subplots(1)

    # If using datetime, format the axis
    if isinstance(data.index, pd.DatetimeIndex):
        ax.xaxis.set_major_formatter(mdate.DateFormatter(dateformat))
    
    # Plot test data
    ax.plot(dates, denormalize(y_test, close_minmax), label='Test', color='k', linewidth=1)
    
    # Plot predictions
    if isinstance(y_pred, np.ndarray):
        # For simple models with direct predictions
        ax.plot(dates, denormalize(y_pred, close_minmax), label='Prediction', color='r', linewidth=1)
    else:
        # For probabilistic models with mean and variance
        ax.plot(dates, denormalize(y_pred.mean(axis=1), close_minmax), label='Prediction', color='r', linewidth=1)
        if y_pred.shape[1] > 1:
            ax.fill_between(dates, 
                        denormalize(y_pred.mean(axis=1) + .5 * y_pred.std(axis=1), close_minmax),
                        denormalize(y_pred.mean(axis=1) - .5 * y_pred.std(axis=1), close_minmax),
                        alpha=.25, label='Prediction Bounds', color='c')
    
    ax.set_xlim([dates[0], dates[-1]])
    ax.set_title(f'Model: {model_name}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.tick_params(axis='x', rotation=15)
    ax.legend()
    plt.tight_layout()
    plt.show()

def split_data(close_price=None, training_len=None, testing_len=None, analyzing_duration=None):
    """Split time series data into training and testing sets for sequence prediction."""
    # Initialize lists to store training and testing data
    train_X, train_y = [], []
    test_X, test_y = [], []

    # Generate training sequences and corresponding output values
    for idx in range(training_len - analyzing_duration):
        train_X.append(close_price[idx:analyzing_duration + idx].reshape(analyzing_duration))
        train_y.append(close_price[idx + analyzing_duration])

    # Reshape training data into the required format
    train_X = np.asarray(train_X).reshape((len(train_X), analyzing_duration, 1))
    train_y = np.asarray(train_y)

    # Generate testing sequences and corresponding output values
    for idx in range(training_len - analyzing_duration, training_len + testing_len - analyzing_duration):
        test_X.append(close_price[idx:analyzing_duration + idx].reshape(analyzing_duration))
        test_y.append(close_price[idx + analyzing_duration])

    # Reshape testing data into the required format
    test_X = np.asarray(test_X).reshape((len(test_X), analyzing_duration, 1))
    test_y = np.asarray(test_y)
    
    return train_X, train_y, test_X, test_y

# Load data - Update path if needed
try:
    data_path = './Apple_20120101_to_20241212.csv'  # Update this path to match your local file location
    data = pd.read_csv(data_path)
    print(f"Successfully loaded data from {data_path}")
except FileNotFoundError:
    print(f"Error: Could not find the file at {data_path}")
    print("Please update the path to match your local file location")
    raise

data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

# Convert the 'Date' column to datetime objects
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Set parameters
test_split_percentage = .8
validation_split_percentage = .2
epochs = 15  # Reduced for faster results
batch_size = 32
verbose = 1  # Changed to 1 to see training progress
analysis_duration = 6
price_pairs = 'BTCUSD'

# Normalize the data
alpha, beta = .8, 1.2
close_price = pd.Series.to_numpy(data['Close'])
data_len = close_price.shape[0]
train_len = int((test_split_percentage * data_len) // analysis_duration * analysis_duration)
test_len = int(data_len - train_len)
close_minmax = np.array([alpha * close_price.min(), beta * close_price.max()], dtype='float32')

close_price = close_price.reshape((data_len, 1)).astype('float32')
close_price = normalize(close_price, close_minmax)

# Split the data
X_train, y_train, X_test, y_test = split_data(
    close_price=close_price, 
    training_len=train_len, 
    testing_len=test_len, 
    analyzing_duration=analysis_duration
)
X_val = X_train[int(train_len - (validation_split_percentage * train_len)):]
y_val = y_train[int(train_len - (validation_split_percentage * train_len)):]
X_train = X_train[:int(train_len - (validation_split_percentage * train_len))]
y_train = y_train[:int(train_len - (validation_split_percentage * train_len))]

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Testing data shape: {X_test.shape}")

# Create a new model instead of loading the incompatible one
print("Creating a new model...")
model_name = 'local_gru_prob_model'
K.clear_session()
model = create_probabilistic_gru_model(input_shape=(analysis_duration, 1))
model.summary()

# Train the model
print("Training the model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    verbose=verbose
)

# Save the model
model.save(f'{model_name}.keras')
print(f"Model saved as {model_name}.keras")

# Generate predictions
test_p = model.predict(X_test)

# Calculate metrics
try:
    # For probabilistic model
    y_true = denormalize(y_test, close_minmax)
    y_pred = denormalize(test_p.mean(axis=1), close_minmax)
    y_pred_reshaped = y_pred[:, np.newaxis] if len(y_pred.shape) == 1 else y_pred
except:
    # For simple model
    y_true = denormalize(y_test, close_minmax)
    y_pred = denormalize(test_p, close_minmax)
    y_pred_reshaped = y_pred

# Calculate evaluation metrics
model_test_results = {}
model_test_results['r2_score'] = float(r2_score(y_true, y_pred_reshaped))
model_test_results['mean_absolute_percentage_error'] = float(mean_absolute_percentage_error(y_true, y_pred_reshaped))
model_test_results['explained_variance_score'] = float(explained_variance_score(y_true, y_pred_reshaped))
try:
    model_test_results['mean_squared_log_error'] = float(mean_squared_log_error(y_true, y_pred_reshaped))
except:
    model_test_results['mean_squared_log_error'] = None  # In case of negative predictions
model_test_results['mean_absolute_error'] = float(mean_absolute_error(y_true, y_pred_reshaped))
model_test_results['root_mean_squared_error'] = float(root_mean_squared_error(y_true, y_pred_reshaped))
model_test_results['max_error'] = float(max_error(y_true, y_pred_reshaped))

# Plot the results
plot_prediction(y_train, y_test, test_p, model_name=model_name)

# Print metrics
print("\nModel Evaluation:")
for metric, value in model_test_results.items():
    if value is not None:
        print(f"{metric}: {value:.4f}")

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plt.show()