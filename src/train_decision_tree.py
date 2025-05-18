import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# the path from where to read the preprocessed data
PREPROCESSED_DATA_PATH = '../data/preprocessed'
# the path where to store the trained model
MODEL_PATH = '../models/tree_model.joblib'


def main():
    # load the preprocessed data
    x_train = pd.read_csv(f'{PREPROCESSED_DATA_PATH}/x_train.csv').values
    y_train = pd.read_csv(f'{PREPROCESSED_DATA_PATH}/y_train.csv').values.ravel()
    # load encoder and scaler
    le = joblib.load(f'{PREPROCESSED_DATA_PATH}/label_encoder.pkl')
    scaler = joblib.load(f'{PREPROCESSED_DATA_PATH}/scaler.pkl')

    # train Decision Tree model
    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(x_train, y_train)

    # save the model along with the LabelEncoder and StandardScaler to ensure consistent preprocessing (encoding and scaling) when
    # predicting new data
    to_save = {'model': tree, 'le': le, 'scaler': scaler}
    joblib.dump(to_save, MODEL_PATH)
    print(f"Decision Tree model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
