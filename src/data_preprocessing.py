import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# the path from where to read the raw data
RAW_DATA_PATH = '../data/raw/loan_data.csv'
# the path where to store the preprocessed dataset
PREPROCESSED_DATA_PATH = '../data/preprocessed'


def load_data(path: str) -> pd.DataFrame:
    """
    Load raw CSV into a DataFrame.
    """
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Encode categorical variables, scale features, and split into train/test sets.

    Preprocesses the dataset for machine learning by performing several key steps. First, it encodes categorical variables, specifically the
    'employment_status' column into a numerical format using label encoding. Next, it scales the numerical features so that each has a mean
    of 0 and a standard deviation of 1. This standardization ensures that features are on a similar scale, which helps improve the
    performance and convergence of many models by preventing features with larger numeric ranges from dominating others.

    After scaling, the method splits the dataset into training and testing sets. The training set is used to teach the model, while the
    testing set is used to evaluate how well the model generalizes to unseen data.

    :param df: The input DataFrame containing the dataset to be preprocessed.
    :param test_size: Proportion of the dataset to include in the test split.
    :param random_state: Random number generator to ensure reproducibility of the split.
    :return: X_train, X_test, y_train, y_test, along with fitted encoder and scaler.
    """

    # encode employment_status to a numerical value making it usable by machine learning models that require numeric input
    le = LabelEncoder()
    df['employment_status_encoded'] = le.fit_transform(df['employment_status'])

    # prepare feature matrix x and target vector y where x represents the input features used to make predictions, and y represents the
    # target variable (whether the loan is approved or not) that the model will learn to predict
    feature_cols = ['income', 'credit_score', 'loan_amount', 'loan_term', 'employment_status_encoded']
    x = df[feature_cols]
    y = df['loan_approved']

    # scaling the features: standardize the feature matrix so that each feature has a mean of 0 and a standard deviation of 1. This is
    # important because machine learning algorithms such as KNN and Decision Tree perform better when features are on a similar scale. This
    # avoids having one feature have more impact than another simply due to its larger numeric range.
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Split the dataset into training and testing subsets and store the results in the following variables:
    # - x_train: the input features used to train the model
    # - x_test: the input features used to evaluate the model's performance on unseen data
    # - y_train: the corresponding target values for x_train (used to teach the model)
    # - y_test: the corresponding target values for x_test (used to assess accuracy or other metrics)
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=test_size, random_state=random_state
    )

    return x_train, x_test, y_train, y_test, le, scaler


def main():
    # load and preprocess
    df = load_data(RAW_DATA_PATH)
    x_train, x_test, y_train, y_test, le, scaler = preprocess(df)

    # save processed data and models
    pd.DataFrame(x_train).to_csv(os.path.join(PREPROCESSED_DATA_PATH, 'x_train.csv'), index=False)
    pd.DataFrame(x_test).to_csv(os.path.join(PREPROCESSED_DATA_PATH, 'x_test.csv'), index=False)
    pd.DataFrame(y_train).to_csv(os.path.join(PREPROCESSED_DATA_PATH, 'y_train.csv'), index=False, header=["loan_approved"])
    pd.DataFrame(y_test).to_csv(os.path.join(PREPROCESSED_DATA_PATH, 'y_test.csv'), index=False, header=["loan_approved"])

    # save encoder and scaler
    joblib.dump(le, os.path.join(PREPROCESSED_DATA_PATH, 'label_encoder.pkl'))
    joblib.dump(scaler, os.path.join(PREPROCESSED_DATA_PATH, 'scaler.pkl'))

    print(f"Preprocessed data and models saved to: {PREPROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()

