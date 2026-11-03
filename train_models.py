import json, joblib, pandas as pd
from data_preparation import engineer_features
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

df=engineer_features(pd.read_csv("deneme_cleaned_english.csv"))
features=[c for c in df.columns if c!="Diagnosis"]
X,y=df[features],df["Diagnosis"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
num=["Age","Back_Pain_Severity","Quality_of_Life","Family_Risk_Count","Symptom_Burden"]
cat=[c for c in features if c not in num]
pre=ColumnTransformer([("num",StandardScaler(),num),("cat",OneHotEncoder(handle_unknown="ignore",sparse_output=False),cat)])
models={"Logistic Regression":LogisticRegression(max_iter=2000,random_state=42),"Decision Tree":DecisionTreeClassifier(max_depth=5,min_samples_leaf=4,random_state=42),"KNN":KNeighborsClassifier(n_neighbors=7),"Naive Bayes":GaussianNB(),"Support Vector Machine":SVC(kernel="rbf",probability=True,random_state=42)}
rows=[]; files={}
for name,est in models.items():
 pipe=Pipeline([("preprocessor",pre),("classifier",est)]); pipe.fit(X_train,y_train)
 pred=pipe.predict(X_test); prob=pipe.predict_proba(X_test)[:,list(pipe.classes_).index("Positive")]
 rows.append({"ML Model Name":name,"Accuracy":accuracy_score(y_test,pred),"AUC":roc_auc_score((y_test=="Positive").astype(int),prob),"Precision":precision_score(y_test,pred,pos_label="Positive",zero_division=0),"Recall":recall_score(y_test,pred,pos_label="Positive",zero_division=0),"F1":f1_score(y_test,pred,pos_label="Positive",zero_division=0),"MCC":matthews_corrcoef(y_test,pred)})
 fn=name.lower().replace(" ","_")+".joblib"; files[name]=fn; joblib.dump(pipe,"model/"+fn)
pd.DataFrame(rows).to_csv("model_metrics.csv",index=False)
X_test.assign(Diagnosis=y_test).to_csv("test_data.csv",index=False)
joblib.dump(features,"model/feature_names.joblib")
json.dump(files,open("model/model_files.json","w"),indent=2)
print(pd.DataFrame(rows).round(4))
