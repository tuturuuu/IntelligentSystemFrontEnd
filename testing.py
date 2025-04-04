    # -*- coding: utf-8 -*-
    """testing.ipynb

    Automatically generated by Colab.

    Original file is located at
        https://colab.research.google.com/drive/1b1iyl5474YvBUVbSm4EdequrF2UKsMfs
    """



    import tensorflow as tf

    import os
    import numpy as np
    import pytz
    import pandas as pd
    import glob
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdate
    from datetime import datetime, timedelta
    import time
    import warnings
    from sklearn.metrics import r2_score
    from sklearn.metrics import max_error
    from sklearn.metrics import mean_absolute_percentage_error as mape
    from sklearn.metrics import explained_variance_score as evs
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.metrics import mean_squared_log_error as msle
    from sklearn.metrics import mean_poisson_deviance as mpd
    from sklearn.metrics import mean_absolute_error as mae
    from sklearn.metrics import root_mean_squared_error as rmse
    from sklearn.metrics import PredictionErrorDisplay
    import tensorflow as tf
    import keras
    import keras.backend as K
    from keras.layers import Input, LSTM, Dense, BatchNormalization, Dropout
    from keras.layers import Activation, TimeDistributed, Flatten, GRU, Bidirectional
    from keras.models import Model, model_from_json, load_model
    from keras.callbacks import Callback
    from tensorflow_probability.python.layers import DistributionLambda
    from tensorflow_probability.python.distributions import Normal

    model_path = "./amazon_review/amazon_review/1743696135/"  # or use the full path if needed
    reconstructed_model = tf.saved_model.load(model_path)

    print('Signature here: ' )
    print(reconstructed_model.signatures)

    # import pkg_resources
    # for package in ['tensorflow', 'numpy', 'ml_dtypes', 'pandas', 'matplotlib',
    #                 'scikit-learn', 'tensorflow-probability', 'keras']:
    #     try:
    #         version = pkg_resources.get_distribution(package).version
    #         print(f"{package}=={version}")
    #     except pkg_resources.DistributionNotFound:
    #         print(f"{package} is not installed")

    iran_timezone = pytz.timezone(zone = 'Iran')
    dateformat = '%Y-%m-%d %H:%M:%S'  # Include seconds in the format

    def normalize(x_, minmax):
        return (x_ - minmax[0]) / (minmax[1] - minmax[0])

    def denormalize(x_, minmax):
        return minmax[0] + ((minmax[1] - minmax[0]) * x_)

    # def plot_prediction(y_train, y_test, y_pred, model_name=None):
    #     # Set up plot parameters
    #     plt.rcParams.update({'figure.figsize': [18, 6]})

    #     # Convert Timestamp objects to strings before using strptime
    #     dates = [mdate.date2num(datetime.fromtimestamp(datetime.strptime(str(d), dateformat).timestamp())) for d in data.index[-len(y_test):]]

    #     fig, ax = plt.subplots(1)

    #     ax.xaxis.set_major_formatter(mdate.DateFormatter(dateformat))
    #     ax.plot(dates, denormalize(y_test, close_minmax), label='Test', color='k', linewidth=1)
    #     ax.plot(dates, denormalize(y_pred.mean(axis=1), close_minmax), label='Prediction', color='r', linewidth=1)
    #     if y_pred.shape[1] > 1:
    #         ax.fill_between(dates, denormalize(y_pred.mean(axis=1) + .5 * y_pred.std(axis=1), close_minmax),
    #                         denormalize(y_pred.mean(axis=1) - .5 * y_pred.std(axis=1), close_minmax),
    #                         alpha=.25, label='Prediction Bounds', color='c')
    #     ax.set_xlim([dates[0], dates[-1]])
    #     ax.set_title(f'Best Model: {model_name}')
    #     ax.set_xlabel('Time')
    #     ax.set_ylabel('Price')
    #     ax.tick_params(axis='x', rotation=15)
    #     ax.legend()
    #     plt.show()

    def split_data(close_price = None, training_len = None, testing_len = None, analyzing_duration = None):
        train_X, train_y = [], []
        test_X, test_y = [], []

        # Generate training sequences and corresponding output values
        for idx in range(training_len - analyzing_duration):
            train_X.append(close_price[idx:analyzing_duration + idx].reshape(analyzing_duration))
            train_y.append(close_price[idx + analyzing_duration])

        # Reshape training data into the required format
        train_X, train_y = np.asarray(train_X).reshape((len(train_X), analyzing_duration, 1)), np.asarray(train_y)

        # Generate testing sequences and corresponding output values
        for idx in range(training_len - analyzing_duration, training_len + testing_len - analyzing_duration):
            test_X.append(close_price[idx:analyzing_duration + idx].reshape(analyzing_duration))
            test_y.append(close_price[idx + analyzing_duration])

        # Reshape testing data into the required format
        test_X, test_y = np.asarray(test_X).reshape((len(test_X), analyzing_duration, 1)), np.asarray(test_y)
        return train_X, train_y, test_X, test_y

    import pandas as pd
    import numpy as np

    data = pd.read_csv('./data/Microsoft_20120101_to_20241212.csv')

    data.columns = ['Date', 'Open','High','Low','Close','Volume']

    # show number of rows where the 'Date' falls between January 1st, 2024, and December 12th, 2024

    # Convert the 'Date' column to datetime objects if it's not already
    data['Date'] = pd.to_datetime(data['Date'])

    # Filter the DataFrame
    start_date = pd.to_datetime('2024-01-01')
    end_date = pd.to_datetime('2024-12-12')
    filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

    # Display the filtered data
    filtered_data

    data.set_index('Date', inplace = True)

    test_split_percentage = .8
    validation_split_percentage = .2
    epochs = 20
    batch_size = 32
    verbose = 0
    analysis_duration = 6
    price_pairs = 'BTCUSD'

    """
    Parameters for normalizing cryptocurrency closing prices:
    - alpha (float): Scaling factor for the minimum value.
    - beta (float): Scaling factor for the maximum value.
    - close_price (numpy.ndarray): NumPy array containing closing prices.
    - data_len (int): Total length of the price data.
    - train_len (int): Length of the training data.
    - test_len (int): Length of the testing data.
    - close_minmax (numpy.ndarray): Min-max normalization range for closing prices.
    """
    alpha, beta = .8, 1.2
    close_price = pd.Series.to_numpy(data['Close'])
    data_len = close_price.shape[0]
    train_len = int((test_split_percentage * data_len) // analysis_duration * analysis_duration)
    test_len = int(data_len - train_len)
    close_minmax = np.array([alpha * close_price.min(), beta * close_price.max()], dtype = 'float32')

    """
    Description:
    - Normalize the closing prices of cryptocurrency data and save the min-max normalization range.

    Parameters:
    - close_price (numpy.ndarray): Reshaped and normalized closing prices.
    - data_len (int): Total length of the price data.
    - close_minmax (numpy.ndarray): Min-max normalization range for closing prices.
    - price_pairs (str): Symbol pairs for the cryptocurrency prices dataset.
    """
    close_price = close_price.reshape((data_len, 1)).astype('float32')
    close_price = normalize(close_price, close_minmax)
    np.save(f'close_minmax_{price_pairs.lower()}.npy', close_minmax)

    X_train, y_train, X_test, y_test = split_data(close_price = close_price, training_len = train_len, testing_len = test_len, analyzing_duration = analysis_duration)
    X_val, y_val = X_train[int(train_len - (validation_split_percentage * train_len)):], y_train[int(train_len - (validation_split_percentage * train_len)):]
    X_train, y_train = X_train[:int(train_len - (validation_split_percentage * train_len))], y_train[:int(train_len - (validation_split_percentage * train_len))]

    from sklearn.metrics import r2_score
    from sklearn.metrics import max_error
    from sklearn.metrics import mean_absolute_percentage_error as mape
    from sklearn.metrics import explained_variance_score as evs
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.metrics import mean_squared_log_error as msle
    from sklearn.metrics import mean_poisson_deviance as mpd
    from sklearn.metrics import mean_absolute_error as mae
    from sklearn.metrics import root_mean_squared_error as rmse
    # Add this line before loading the model
    from keras.layers import GaussianNoise # Import GaussianNoise layer


    """
    Description:
    - Performs transfer learning by loading a pre-trained Probabilistic GRU model.
    - Transfer learning is a machine learning technique where knowledge gained from one model (source model) is applied to another related model (target model).
    - In this code, transfer learning is implemented for time series forecasting using a pre-trained Probabilistic GRU model.
    - The source model, 'best_BTCPred_gru_prob,' is trained on Bitcoin price data.
    - The target model, 'best_TRXPred_gru_prob,' is trained on Tron price data using the weights transferred from the source model.
    - This approach leverages the learned patterns and features from the Bitcoin data to enhance the training of the Tron model.
    - The transfer_model() function loads the pre-trained model, freezes the layers up to the output layer, and compiles it for training.
    - The model is then trained on Tron data, and the performance is evaluated and recorded in transfer_model_test_result.
    - Various metrics such as R2 score, mean absolute percentage error, explained variance score, etc., are used to assess the model's performance.


    Parameters:
    - K.clear_session(): Clears the Keras session to release resources.
    - m_name_ (str): Model name for identification.
    - transfer_model_test_result (dict): Dictionary to store evaluation results.
    - model (tf.keras.Model): Transfer learning model loaded from a pre-trained model.
    - out_path_best (str): Path to store the best model during training.
    - X_train, y_train, X_test, y_test (numpy.ndarray): Training and test data.
    - epochs, batch_size, verbose (int): Training parameters.
    - close_minmax (numpy.ndarray): Min-max values for denormalization.

    Returns:
    - None: The transfer learning model is trained, evaluated, and results are recorded in transfer_model_test_result.
    """
    y_pred_best, models_test_result = {}, {}
    K.clear_session()
    m_name_ = 'fulltrained_BTCPred_lstm_time_dist'
    transfer_model_test_result = {}
    transfer_model_test_result.update({f'{m_name_}': {}})

    best_model = tf.saved_model.load(model_path)

    # Get the serving function
    infer = best_model.signatures["serving_default"]

    print(best_model.signatures)

    # Ensure X_test is a Tensor with the expected shape (None, 6, 1)

    X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)

    # Run inference
    test_p_best = infer(close_price=X_test_tensor)

    # Extract predictions as a NumPy array
    test_p_best = test_p_best["distribution_lambda"].numpy()  # Fix applied

    # Now this works because test_p_best is an array
    y_pred_best[f'{m_name_}'] = test_p_best

    y_pred_best.update({f'{m_name_}': test_p_best})

    transfer_model_test_result[f'{m_name_}'].update({'r2_score': float(r2_score(denormalize(y_test, close_minmax),
                                                                                denormalize(test_p_best.mean(axis = 1), close_minmax)))})
    transfer_model_test_result[f'{m_name_}'].update({'mean_absolute_percentage_error': float(mape(denormalize(y_test, close_minmax),
                                                                                            denormalize(test_p_best.mean(axis = 1), close_minmax)))})
    transfer_model_test_result[f'{m_name_}'].update({'explained_variance_score': float(evs(denormalize(y_test, close_minmax),
                                                                                        denormalize(test_p_best.mean(axis = 1), close_minmax)))})
    transfer_model_test_result[f'{m_name_}'].update({'mean_squared_log_error': float(msle(denormalize(y_test, close_minmax),
                                                                                        denormalize(test_p_best.mean(axis = 1), close_minmax)))})
    transfer_model_test_result[f'{m_name_}'].update({'mean_poisson_deviance': float(mpd(denormalize(y_test, close_minmax),
                                                                                        denormalize(test_p_best.mean(axis = 1), close_minmax)))})
    transfer_model_test_result[f'{m_name_}'].update({'max_error': float(max_error(denormalize(y_test, close_minmax),
                                                                                denormalize(test_p_best.mean(axis = 1), close_minmax)))})
    transfer_model_test_result[f'{m_name_}'].update({'mean_absolute_error': float(mae(denormalize(y_test, close_minmax),
                                                                                    denormalize(test_p_best.mean(axis = 1), close_minmax)))})
    transfer_model_test_result[f'{m_name_}'].update({'root_mean_squared_error': float(rmse(denormalize(y_test, close_minmax),
                                                                                        denormalize(test_p_best.mean(axis = 1), close_minmax)))})

    plot_prediction(y_train, y_test, y_pred_best[m_name_], model_name = m_name_.replace('_', ' '))

    from sklearn.metrics import mean_squared_error, mean_absolute_error

    y_true = denormalize(y_test, close_minmax)
    y_pred = denormalize(y_pred_best[m_name_].mean(axis = 1), close_minmax)[:, np.newaxis]

    # Calculate Metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(y_pred)

    # Print Metrics
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")