# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Precision: 0.7947732513451191
Recall: 0.64504054897068
F-beta: 0.7121212121212122
learning_rate: 0.05, 
'loss': 'log_loss',
 'max_depth': 5,
  'max_features': None,
   'max_leaf_nodes': None,
    'min_impurity_decrease': 0.0,
     'min_samples_leaf': 1,
      'min_samples_split': 5, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 600, 'n_iter_no_change': None, 'random_state': None, 'subsample': 1.0, 'tol': 0.0001, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False
## Intended Use
A model to predit the person's income range for use in related fields such as loan approval systems.
## Training Data
The training data is the census data at the UCI library.
## Evaluation Data
With 0.2 fraction of the census data, the model was tested
## Metrics
Precision: 0.7845631891433418
Recall: 0.5921895006402048
F-beta: 0.67493615468807
## Ethical Considerations
This dataset shall not be considered representative of salary distribution.

## Caveats and Recommendations
An outdated census database, make it not adequate representative of the population or the salary distribution.