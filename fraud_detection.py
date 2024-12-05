
import pandas as pnds
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 

def load_data(file_path):
    # Loading the dataset from the specified file path.
    try:
        data = pnds.read_csv(file_path)
        return data
    except FileNotFoundError:
        print("Error: The specified CSV file was not found.")
        exit()

def check_data(data):
    # Checking if the dataset contains the 'Class' column.
    if 'Class' not in data.columns:
        print("Error: The dataset must contain a 'Class' column.")
        exit()
    else:
        print("The dataset contains the target 'Class' column.")

def train_model(X, y):
   # Training the model.
    model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    model.fit(X, y)
    return model


def getting_user_input(features):
    # Getting user input for transaction details.
    user_data = {}
    print("Enter The Transaction Details")
    
    for feature in features:
        while True:
            try:
                value = float(input(f"Enter {feature} value: "))
                user_data[feature] = [value]
                break  # Exiting the loop if input is valid
            except ValueError:
                print("Invalid Input. Please enter a numeric value.")
    return pnds.DataFrame(user_data)

def main():
    # Loading the dataset
    data = load_data('C:/Users/thush/OneDrive/Desktop/Minor Project/transactions.csv')
    
    # Checking if 'Class' column exists
    check_data(data)

    # Separating features and target
    X = data.drop('Class', axis=1)
    y = data['Class']

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initializing and training the model
    model = train_model(X_train, y_train)
    print("Model training complete.")

    # Adding a confusion matrix, precision,recall,F1 score to display the performance of the model
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", conf_matrix)

    true_labels = [0, 1, 0, 1, 0, 0, 1] 
    predicted_labels = [0, 1, 0, 0, 0, 1, 1] 

    precision = precision_score(true_labels, predicted_labels)
    print(f'Precision: {precision:.2f}')

    recall = recall_score(true_labels, predicted_labels)
    print(f'Recall: {recall:.2f}')

    f1 = f1_score(true_labels, predicted_labels)
    print(f'F1 Score: {f1:.2f}')

    # Evaluating the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f} %")

    # Predicting the result
    user_input = getting_user_input(X.columns)

    # Ensuring user input has the same columns as the training data
    if not all(col in user_input.columns for col in X.columns):
        print("Error: The input data does not match the expected feature set.")
        exit()

    prediction = model.predict(user_input)

    # Displaying the result
    if prediction[0] == 1:
        print("This transaction is fraudulent.")
    else:
        print("This transaction is valid.")

if __name__ == "__main__":
    main()
