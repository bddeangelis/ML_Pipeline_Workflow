import numpy as np
import dill as pickle  # Use dill as it is a more full-featured serializer than pickle
import h5py
import os
import time
import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

# PipelineModel: A class defining a model for predicting locomotion variables from limb phases.
# NOTE: This model class simply makes it easier to load and save data in the desired formats. The heavy lifting is done
# with the Pipeline class of sklearn. By using Pipeline, we don't need to to define our own fit() and predict() methods
# but can simply call the methods defined by the Pipeline class using: <obj>.pipe.fit() and <obj>.pipe.predict()
class PipelineModel(object):

    # Constructor
    def __init__(self, config=None):

        # Define the model name
        model_timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        self.name = 'PipelineModel_' + model_timestamp

        # Set default configuration dictionary
        self.config = {
            'destination_path': None,
            'data_file': None,
            'test_size': 0.1,
            'hidden_layer_sizes': (200,10,200),
            'activation': 'tanh',
            'solver': 'sgd',
            'alpha': 0.001,
            'learning_rate': 'adaptive',
            'max_iter': 10000,
            'tol': 1e-6,
            'verbose': True}

        # Update the configuration if any options are supplied as an argument
        if config is not None and isinstance(config, dict):
            self.config.update(config)

        # Make a folder in the destination path for storing the files associated with model
        if not os.path.isdir(os.path.join(self.config['destination_path'], model_timestamp)):
            os.mkdir(os.path.join(self.config['destination_path'], model_timestamp))
        self.config['destination_path'] = os.path.join(self.config['destination_path'], model_timestamp)

        # Set the logging handlers
        handlers = [
                logging.FileHandler(os.path.join(self.config['destination_path'], ('Logfile_%s.txt' % self.name))),
                logging.StreamHandler()]

        # Configure the logging settings
        logging.basicConfig(level=logging.INFO, format='%(asctime)-15s : %(message)s', handlers=handlers)

        # Print a header
        if __name__ == "__main__":
            logging.info('Training model %s.', self.name)
            logging.info('Data source path: %s.', self.config['data_file'])
            logging.info('MLP hidden layer sizes: %s.', str(self.config['hidden_layer_sizes']))

        # Define the steps that are contained within the analysis pipeline
        normalize = StandardScaler()
        net = MLPRegressor(hidden_layer_sizes=self.config['hidden_layer_sizes'],
                           activation=self.config['activation'],
                           solver=self.config['solver'],
                           alpha=self.config['alpha'],
                           learning_rate=self.config['learning_rate'],
                           max_iter=self.config['max_iter'],
                           tol=self.config['tol'],
                           verbose=self.config['verbose'])
        estimators = [('scale', normalize), ('mlp', net)]

        # Initialize the pipeline for the analysis
        self.pipe = Pipeline(memory=None, steps=estimators)

    # This method splits the dataset into training and test data
    def split_data(self, data, X, y):

        # NOTE: This just brings the method in sklearn.model_selection into the class. We can perform additional
        # modifications here if needed in the future.
        start_time = time.time()

        # Single Split
        data['X_train'], data['X_test'], data['y_train'], data['y_test'] \
            = train_test_split(X, y, test_size=self.config['test_size'])

        logging.info('Successfully split the data set into train and test: %f seconds.', (time.time() - start_time))

        return data

    # Method to evaluate the performance of the fit
    def eval_model(self):
        # Implement if we need a different metric than the r^2 score from MLPRegressor()
        pass

    # This method loads in the data from matlab .mat files
    def load_data_from_mat(self, file_path):

        logging.info('Importing data from %s', file_path)
        start_time = time.time()

        # Create an h5py object to read from the HDF5 .mat file
        mat_file = h5py.File(file_path, mode='r')

        # Import predictor and target variables from the .mat file
        X = np.array(mat_file['X'].value.swapaxes(0,1))
        y = np.array(mat_file['y'].value.swapaxes(0,1))

        # Log the time for completion
        logging.info('Imported data from .mat file in %f seconds.' % (time.time() - start_time))

        return X, y

    # This method is used to pickle the model for later re-use
    def pickle_model(self):
        start_time = time.time()
        model_file = os.path.join(self.config['destination_path'], ('%s.pk' % self.name))
        with open(model_file, 'wb') as output:
            pickle.dump(self, output, -1)
        logging.info('Pickled the model: %f seconds', time.time()-start_time)
        return None

    # This method is used to save dictionaries in the hdf5 format
    def save_as_hdf5(self, data, dict_name ):

        start_time = time.time()

        # Form the save path
        path = os.path.join(self.config['destination_path'], (dict_name + '_%s.hdf5' % self.name))

        # Create an HDF5 file
        with h5py.File(path, "w") as hf:
            # Save data
            for k, v in data.items():
                hf.create_dataset(k, data=v)
            hf.close()
        logging.info('Saved %s as HDF5 file %s in %f seconds.' % (dict_name, path, (time.time() - start_time)))
        return None

# Test script with example model workflow
def main(data_file, destination_path):

    # Initialize a dictionary for storing the data
    data = {'source_file': data_file,
            'test_size': None,
            'X_train': None,
            'y_train': None,
            'X_test': None,
            'y_test': None,
            }

    # Initialize a dictionary for storing the predictions
    predictions ={'source_file': data_file,
                  'associated_model': None,
                  'y_train_predict': None,
                  'y_test_predict': None,
                  'train_score': None,
                  'test_score': None
                  }

    # Initialize some configuration fields
    config = {'destination_path': destination_path,
              'data_file': data_file
              }

    # Initialize a model
    model = PipelineModel(config=config)

    # Store some information from the model in the data and prediction dictionaries
    data['test_size'] = model.config['test_size']
    predictions['associated_model'] = model.name

    # Load the data
    X, y = model.load_data_from_mat(data_file)

    # Split the data into training and test sets
    data = model.split_data(data, X, y)

    # Train a model
    start_time = time.time()
    model.pipe.fit(data['X_train'], data['y_train'])
    logging.info('Successfully fit a model: %f seconds', time.time()-start_time)

    # Generate predictions on the training set
    start_time = time.time()
    predictions['y_train_predict'] = model.pipe.predict(data['X_train'])
    logging.info('Successfully ran predictions on training data: %f seconds', time.time()-start_time)

    # Run the model on the test set
    start_time = time.time()
    predictions['y_test_predict'] = model.pipe.predict(data['X_test'])
    logging.info('Successfully ran predictions on test data: %f seconds', time.time()-start_time)

    # Store the ground truth y values in predictions for convenience of later analysis
    predictions['y_train'] = data['y_train']
    predictions['y_test'] = data['y_test']

    # Evalutate the performance of the model
    predictions['train_score'] = model.pipe.score(data['X_train'], data['y_train'])
    logging.info('Training r^2 Score: %f' % predictions['train_score'])
    predictions['test_score'] = model.pipe.score(data['X_test'], data['y_test'])
    logging.info('Testing r^2 Score: %f' % predictions['test_score'])

    # Pickle the model for later use
    model.pickle_model()

    # Write the predictions to file for later analysis
    model.save_as_hdf5(predictions, 'predictions')

    # # Write the data to file for later use
    # model.save_as_hdf5(data, 'data')

    return None

if __name__ == "__main__":
    data_file = '/Users/bdeangelis/Desktop/Datasets/LocomotionModelDatasets/ModelTraining_IsoD1_Glass_winlen_100_ex_50k.mat'
    destination_path = '/Users/bdeangelis/Desktop/Models'
    main(data_file, destination_path)
