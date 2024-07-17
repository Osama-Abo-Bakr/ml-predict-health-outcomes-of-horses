# ML - Predict Health Outcomes of Horses

## Project Overview

This project aims to predict health outcomes for horses using machine learning techniques. It involves data preprocessing, feature engineering, model building, hyperparameter tuning, and evaluation to achieve high accuracy in predictions.

## Table of Contents

1. [Libraries and Tools](#libraries-and-tools)
2. [Data Reading](#data-reading)
3. [Data Preprocessing](#data-preprocessing)
4. [Data Visualization](#data-visualization)
5. [Feature Engineering](#feature-engineering)
6. [Model Building](#model-building)
7. [Hyperparameter Tuning](#hyperparameter-tuning)
8. [Neural Network](#neural-network)
9. [Installation](#installation)
10. [Usage](#usage)
11. [Conclusion](#conclusion)
12. [Contact](#contact)

## Libraries and Tools

- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical operations
- **Matplotlib, Seaborn**: Data visualization
- **Scikit-learn**: Preprocessing, feature extraction, and model building
- **Keras, TensorFlow**: Neural network building
- **Imbalanced-learn (SMOTE)**: Handling imbalanced data
- **XGBoost**: Advanced gradient boosting
- **FLAML, AutoGluon**: AutoML tools for model optimization

## Data Reading

### Description
The dataset consists of health records of horses. Each record has several features, and the target variable is the health outcome.

### Code
```python
train = pd.read_csv(r"D:\Courses language programming\6_Deep Learning\Deep Learning Project\Competitions\Predict Health Outcomes of Horses\train.csv")
test = pd.read_csv(r"D:\Courses language programming\6_Deep Learning\Deep Learning Project\Competitions\Predict Health Outcomes of Horses\test.csv")

test_id = test["id"]
```

## Data Preprocessing

### Description
Preprocessing involves filling missing values, dropping unnecessary columns, and encoding categorical variables.

### Code
```python
def filling_Data(data):
    for col in data.columns:
        if data[col].dtype == "O":
            data[col].fillna(data[col].value_counts().index[0], inplace=True)
        else:
            data[col].fillna(data[col].median(), inplace=True)
    return data
            
train = filling_Data(train)
test = filling_Data(test)

def drop_col(data):
    data.drop(columns=["id", "hospital_number"], axis=1, inplace=True)
    return data

train = drop_col(train)
test = drop_col(test)

def Lab_Enc(data):
    la = LabelEncoder()
    data_obj = data.select_dtypes(include=["object"])
    for col in data_obj.columns:
        data[col] = la.fit_transform(data[col])
    return data

train = Lab_Enc(train)
test = Lab_Enc(test)
```

## Data Visualization

### Description
Histograms are used to visualize the distribution of features.

### Code
```python
train.hist(figsize=(20, 20))
plt.show()
```

## Feature Engineering

### Description
Log transformation is applied to certain features to handle skewness. SMOTE is used to handle imbalanced data.

### Code
```python
def log_transform(data):
    for col in ["respiratory_rate", "total_protein"]:
        median = data[col].median()
        data[col] = np.log(data[col])
        data[col] = data[col].replace([-np.inf, np.inf], median)
    return data

train = log_transform(train)
test = log_transform(test)

X_input = train.drop(columns="outcome", axis=1)
Y_output = train["outcome"]

new_x, new_y = SMOTE().fit_resample(X_input, Y_output)

train = pd.concat([new_x, new_y], axis=1)
```

## Model Building

### Description
Various models are built and evaluated, including SGDClassifier, RandomForestClassifier, AdaBoostClassifier, XGBClassifier, AutoML, and AutoGluon.

### Code
```python
# SGDClassifier
model = SGDClassifier(penalty="l1")
model.fit(x_train, y_train)
print(f"The accuracy_score of training is ==> {model.score(x_train, y_train)}")
print(f"The accuracy_score of testing is ==> {model.score(x_test, y_test)}")

# RandomForestClassifier
model_RF = RandomForestClassifier(max_depth=150, n_estimators=100)
model_RF.fit(x_train, y_train)
print(f"The Accuracy SCore Train is {model_RF.score(x_train, y_train)}")
print(f"The Accuracy SCore Test is {model_RF.score(x_test, y_test)}")

# AdaBoostClassifier
model_AD = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1000, min_samples_split=5, min_samples_leaf=3), n_estimators=200, learning_rate=0.01)
model_AD.fit(x_train, y_train)
print(f"The predict Score Train is ==> {model_AD.score(x_train, y_train)}")
print(f"The predict Score Test is ==> {model_AD.score(x_test, y_test)}")

# XGBClassifier
model_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=150, max_leaves=5, learning_rate=0.1)
model_xgb.fit(X, Y)
print(f"The predict Score Train is ==> {model_xgb.score(x_train, y_train)}")
print(f"The predict Score Test is ==> {model_xgb.score(x_test, y_test)}")
```

## Hyperparameter Tuning

### Description
GridSearchCV is used to find the best hyperparameters for the models.

### Code
```python
param = {"n_estimators": np.arange(100, 500, 100), "max_depth": np.arange(10, 100, 10), "max_leaves": [1, 2, 3, 4, 5]}
new_model_xgb = GridSearchCV(estimator=xgb.XGBClassifier(), param_grid=param, verbose=6, cv=5, n_jobs=-1)
new_model_xgb.fit(x_train, y_train)
print(f"The Accuracy SCore Train is {new_model_xgb.score(x_train, y_train)}")
print(f"The Accuracy SCore Test is {new_model_xgb.score(x_test, y_test)}")
```

## Neural Network

### Description
A neural network is built using Keras and TensorFlow.

### Code
```python
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
import tensorflow.keras as k

label = to_categorical(Y, 3)
X1 = train.drop(["outcome"], axis=1)

x_train1, x_test1, y_train1, y_test1 = train_test_split(X1, label, train_size=0.8, random_state=42)

model_nn = k.models.Sequential([
    k.layers.Dense(512, activation="relu"),
    k.layers.Dense(128, activation="tanh"),
    k.layers.Dense(256, activation="tanh"),
    k.layers.Dense(128, activation="tanh"),
    k.layers.Dense(64, activation="tanh"),
    k.layers.Dense(32, activation="relu"),
    k.layers.Dense(16, activation="tanh"),
    k.layers.Dense(1, activation="softmax")
])

model_nn.compile(optimizer="adam", loss=k.losses.CategoricalFocalCrossentropy(), metrics=["accuracy"])
history = model_nn.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), validation_split=0.2)
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Osama-Abo-Bakr/ml-predict-health-outcomes-of-horses.git
   ```

2. Navigate to the project directory:
   ```bash
   cd ml-predict-health-outcomes-of-horses
   ```

## Usage

1. **Prepare Data**:
   - Ensure the dataset is available at the specified path.

2. **Run the Scripts**:
   - Execute the provided scripts to preprocess data, build models, and evaluate their performance.

3. **Predict Outcomes**:
   - Use the trained models to predict health outcomes for new horse data.

## Conclusion

This project demonstrates the application of various machine learning techniques to predict health outcomes for horses. It highlights the importance of data preprocessing, feature engineering, model building, and hyperparameter tuning to achieve high accuracy.

## Contact

For questions or collaborations, please reach out via:

- **Email**: [Gmail](mailto:osamaoabobakr12@gmail.com)
- **LinkedIn**: [LinkedIn](https://linkedin.com/in/osama-abo-bakr-293614259/)
