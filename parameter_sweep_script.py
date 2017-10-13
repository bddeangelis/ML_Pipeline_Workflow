import time
import logging
import dill as pickle  # Use dill as it is a more full-featured serializer than pickle
import os
from pipeline_model import PipelineModel
from sklearn.model_selection import GridSearchCV

# Define the path to the data file
data_file = '/Users/bdeangelis/Desktop/Datasets/LocomotionModelDatasets/ModelTraining_IsoD1_Glass_winlen_100_ex_50k.mat'
destination_path = '/Users/bdeangelis/Desktop/Models'

# Initialize some configuration fields
config = {'destination_path': destination_path,
          'data_file': data_file
          }

# Initialize a model
model = PipelineModel(config=config)

# Initialize an empty dictionary for storing the data splits
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

# Load the data
X, y = model.load_data_from_mat(data_file)

# Split the data into train and test
data = model.split_data(data, X, y)

# Define a set of parameters for the GridSearchCV
# parameters = dict(mlp__hidden_layer_sizes=[(200,), (400,), (1000,), (200, 200), (400, 400), (1000,1000)],
#                   mlp__activation=[('tanh'),('logistic')],
#                   mlp__alpha= [(0.0001), (0.001), (0.01)],
#                   mlp__learning_rate_init= [(0.0001), (0.001), (0.01)],
#                   mlp__learning_rate = [('adaptive')],
#                   mlp__solver=[('sgd')])

# Define a set of parameters for initially determining the range for the optimal learning rate
parameters = dict(mlp__hidden_layer_sizes=[(200,), (400,), (1000,)],
                  mlp__activation=[('tanh')],
                  mlp__learning_rate_init= [(0.0001), (0.001), (0.01)],
                  mlp__learning_rate = [('adaptive')],
                  mlp__solver=[('sgd')])

# Initialize a cross-validated grid search
cv = GridSearchCV(model.pipe, param_grid=parameters, refit=True)
cv_timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
cv.name = 'CV_GridSearch_PipelineModel_' + cv_timestamp

# Store some information from the model in the data and prediction dictionaries
data['test_size'] = model.config['test_size']
predictions['associated_model'] = cv.name

# Use GridSearchCV to sweep the model parameters
start_time = time.time()
cv.fit(data['X_train'], data['y_train'])
logging.info('Successfully swept the set of model parameters: %f seconds', time.time()-start_time)

# Print the best estimator to the log
logging.info('The best estimator from GridSearchCV:')
logging.info(cv.best_estimator_)
logging.info('With parameters:')
logging.info(cv.best_params_)

# Store the predictions from the best performing model
start_time = time.time()
predictions['y_train_predict'] = cv.predict(data['X_train'])
predictions['y_test_predict'] = cv.predict(data['X_test'])
logging.info('Successfully ran predictions on training and test data: %f seconds', time.time()-start_time)

# Report the score from the best fit model to the log
predictions['train_score'] = cv.score(data['X_train'], data['y_train'])
logging.info('Training r^2 Score: %f' % predictions['train_score'])
predictions['test_score'] = cv.score(data['X_test'], data['y_test'])
logging.info('Testing r^2 Score: %f' % predictions['test_score'])

# Save the cross-validated grid search model
start_time = time.time()
cv_file = os.path.join(model.config['destination_path'], ('%s.pk' % cv.name))
with open(cv_file, 'wb') as output:
    pickle.dump(cv, output, -1)
logging.info('Pickled the cross-validated grid search model: %f seconds', time.time()-start_time)

# Save the predictions from the best fit model from grid search for later analysis
model.save_as_hdf5(predictions, 'predictions')
