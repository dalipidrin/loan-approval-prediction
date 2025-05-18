import joblib
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# the path from where to read the preprocessed data
PREPROCESSED_DATA_PATH = '../data/preprocessed'


def load_training_model(path):
    """
    Load the trained model and preprocessing objects from the specified path.

    :param path: The file path to the saved training model.
    :return: A tuple containing the trained model, the LabelEncoder, and the StandardScaler.
    """
    trained_model = joblib.load(path)
    return trained_model['model'], trained_model['le'], trained_model['scaler']


def evaluate_model(model, test_features, test_outcomes, model_name):
    """
    Evaluate a trained classification model on test data and print a classification report.

    :param model: The trained machine learning model to evaluate.
    :param test_features: The input features for the test dataset.
    :param test_outcomes: The test outcomes, such as whether a loan was approved (1) or not (0).
    :param model_name: The name of the model to display in the evaluation header.
    :return: None.
    """
    print(f"--- {model_name} Evaluation ---")
    y_pred = model.predict(test_features)
    print(classification_report(test_outcomes, y_pred))


def get_feature_importance(model):
    """
    Retrieve the feature importance values from a trained model.

    :param model: The trained model.
    :return: None.
    """

    feature_names = ['income', 'credit_score', 'loan_amount', 'loan_term', 'employment_status_encoded']
    importances = model.feature_importances_
    plt.figure(figsize=(8, 4))
    plt.barh(feature_names, importances)
    plt.title('Decision Tree Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('../reports/figures/tree_feature_importance.png')
    plt.show()


# load the preprocessed data
x_test = pd.read_csv(f'{PREPROCESSED_DATA_PATH}/x_test.csv').values
y_test = pd.read_csv(f'{PREPROCESSED_DATA_PATH}/y_test.csv').values

# load the training models
knn, _, _ = load_training_model('../models/knn_model.joblib')
tree, _, _ = load_training_model('../models/tree_model.joblib')

# evaluate and test KNN
evaluate_model(knn, x_test, y_test, model_name="KNN")
# evaluate and test Decision Tree
evaluate_model(tree, x_test, y_test, model_name="Decision Tree")

# get the decision tree feature importance
get_feature_importance(tree)
