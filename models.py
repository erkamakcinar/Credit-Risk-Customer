# Model Building
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

#Model Evaluation
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,f1_score

# Logistic Regreesion Model
def model_lr(x_train,x_test,y_train,y_test):
    global acc_lr,f1_lr

    lr=LogisticRegression()
    lr.fit(x_train,y_train)
    y_pred=lr.predict(x_test)

    conf_lr=confusion_matrix(y_test,y_pred)
    acc_lr=accuracy_score(y_test,y_pred)
    f1_lr=f1_score(y_test,y_pred)
    clf_lr=classification_report(y_test,y_pred)

    print('*********** Logistic Regression***********')
    print('\n')
    print('Accuracy : ',acc_lr)
    print('F1 Score : ',f1_lr)
    print(10*'=====')
    print('Confusion Matrix :\n',conf_lr)
    print(10*'=====')
    print('Classification Report :\n',clf_lr)
    print(30*'========')
    return acc_lr, f1_lr
    
# Decision Tree Model
def model_dt(x_train,x_test,y_train,y_test):
    global acc_dt,f1_dt

    lr=DecisionTreeClassifier()
    lr.fit(x_train,y_train)
    y_pred=lr.predict(x_test)

    conf_dt=confusion_matrix(y_test,y_pred)
    acc_dt=accuracy_score(y_test,y_pred)
    f1_dt=f1_score(y_test,y_pred)
    clf_dt=classification_report(y_test,y_pred)

    print('***********Decision Tree***********')
    print('\n')
    print('Accuracy : ',acc_dt)
    print('F1 Score : ',f1_dt)
    print(10*'=====')
    print('Confusion Matrix :\n',conf_dt)
    print(10*'=====')
    print('Classification Report :\n',clf_dt)
    print(30*'========')
    return acc_dt, f1_dt

# K Nearest Neighbor Model
def model_knn(x_train,x_test,y_train,y_test):
    global acc_knn,f1_knn

    lr=KNeighborsClassifier()
    lr.fit(x_train,y_train)
    y_pred=lr.predict(x_test)

    conf_knn=confusion_matrix(y_test,y_pred)
    acc_knn=accuracy_score(y_test,y_pred)
    f1_knn=f1_score(y_test,y_pred)
    clf_knn=classification_report(y_test,y_pred)

    print('***********K Nearest Neighbor***********')
    print('\n')
    print('Accuracy : ',acc_knn)
    print('F1 Score : ',f1_knn)
    print(10*'=====')
    print('Confusion Matrix :\n',conf_knn)
    print(10*'=====')
    print('Classification Report :\n',clf_knn)
    print(30*'========')
    return acc_knn, f1_knn

# Random Forest Model
def model_rf(x_train,x_test,y_train,y_test):
    global acc_rf,f1_rf

    lr=RandomForestClassifier()
    lr.fit(x_train,y_train)
    y_pred=lr.predict(x_test)

    conf_rf=confusion_matrix(y_test,y_pred)
    acc_rf=accuracy_score(y_test,y_pred)
    f1_rf=f1_score(y_test,y_pred)
    clf_rf=classification_report(y_test,y_pred)

    print('***********Random Forest***********')
    print('\n')
    print('Accuracy : ',acc_rf)
    print('F1 Score : ',f1_rf)
    print(10*'=====')
    print('Confusion Matrix :\n',conf_rf)
    print(10*'=====')
    print('Classification Report :\n',clf_rf)
    print(30*'========')
    return acc_rf, f1_rf
