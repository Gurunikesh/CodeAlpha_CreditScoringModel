import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve

from sklearn.ensemble import RandomForestClassifier


import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")
pd.set_option("Display.max_rows",None)
pd.set_option("Display.max_columns",None)
df.head()

df.isnull().sum()

df.info()

df.describe()

num_cols = df.select_dtypes(include=['int64','float64'])
outliers_before = {}
outliers_after = {}
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outlier_mask = (df[col] < lower) | (df[col] > upper)
    outliers_before[col] = outlier_mask.sum()
    
    df.loc[outlier_mask, col] = int(df[col].mean())

    outlier_mask1 = (df[col] < lower) | (df[col] > upper)
    outliers_after[col] = outlier_mask1.sum()

print("Outliers before handling:", outliers_before)
print("\n Outliers after handling:", outliers_after)

#df['Gender']=df['Gender'].map({'Male':0,'Female':1})
df['Education']=df['Education'].map({'High School':0,'Bachelor':1,'Master':2,'PhD':3})
df['Payment_History']=df['Payment_History'].map({'Bad':0,'Average':1,'Good':2})
df['Employment_Status']=df['Employment_Status'].map({'Unemployed':0,'Self-Employed':1,'Employed':2})
df['Residence_Type']=df['Residence_Type'].map({'Rented':0,'Owned':1,'Mortgaged':2})
df['Marital_Status']=df['Marital_Status'].map({'Single':0,'Married':1,'Divorced':2})

df['Gender']=df['Gender'].map({'Male':0,'Female':1})

X = df.drop("Creditworthiness", axis=1)
y = df["Creditworthiness"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

def evaluate(y_test, y_pred, name):
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    

evaluate(y_test, y_pred_rf, "Random Forest")

y_prob = rf.predict_proba(X_test)[:,1]
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

importance = rf.feature_importances_
features = X.columns

plt.figure()
plt.barh(features, importance)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.show()

y_prob = rf.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()

import joblib
joblib.dump(rf, "credit_model.pkl")
joblib.dump(scaler, "scaler.pkl")
