from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import streamlit as st
import pandas as pd
import PreProcessing

class Classifier:
    def __init__(self, clf_name, data_set):
        self.clf_name = clf_name
        self.data_set = data_set
        self.classifier = None
        self.pre_process = PreProcessing.PreProcessing(self.data_set)

    def classifier_param(self):
        param = dict()
        st.sidebar.subheader("Tune Parameters")
        if self.clf_name == "Logistic Regression":
            st.sidebar.markdown("""**Note:** Only following parameters can be tuned""")
            random_state = st.sidebar.slider("Random State", 0, 100, 0)
            param["random_state"] = random_state

        elif self.clf_name == "KNN":
            st.sidebar.markdown("""**Note:** Only following parameters can be tuned""")
            n_neighbors = st.sidebar.slider("Number of Neighbors", 1, 30, 5)
            optionals = st.sidebar.expander("Optional Parameters", False)
            p = optionals.slider("Power Parameter", 1, 5, 2)
            metric = optionals.selectbox("Distance Metric", ("minkowski", "euclidean", "manhattan", "chebyshev"))
            param["n_neighbors"] = n_neighbors
            param["p"] = p
            param["metric"] = metric

        elif self.clf_name == "SVM":
            st.sidebar.markdown("""**Note:** Only following parameters can be tuned""")
            C = st.sidebar.slider("Regularization Parameter", 0.01, 10.0, 1.0)
            kernel = st.sidebar.selectbox("Kernel Type", ("rbf", "linear", "poly", "sigmoid", "precomputed"))
            random_state = st.sidebar.slider("Random State", 0, 100, 0)
            param["C"] = C
            param["kernel"] = kernel
            param["random_state"] = random_state

        elif self.clf_name == "Naive Bayes":
            st.sidebar.markdown("""Using Gaussian Naive Bayes (GaussianNB) with __default__ parameters""")

        elif self.clf_name == "Decision Tree":
            st.sidebar.markdown("""**Note:** Only following parameters can be tuned""")
            criterion = st.sidebar.selectbox("Quality of a Split", ("entropy", "gini"))
            random_state = st.sidebar.slider("Random State", 0, 100, 0)
            param["criterion"] = criterion
            param["random_state"] = random_state

        elif self.clf_name == "Random Forest":
            st.sidebar.markdown("""**Note:** Only following parameters can be tuned""")
            n_estimators = st.sidebar.slider("Number of trees", 1, 500, 100)
            criterion = st.sidebar.selectbox("Quality of a Split", ("entropy", "gini"))
            random_state = st.sidebar.slider("Random State", 0, 100, 0)
            param["n_estimators"] = n_estimators
            param["criterion"] = criterion
            param["random_state"] = random_state

        return param

    def set_classifier(self):
        param = self.classifier_param()
        if self.clf_name == "Logistic Regression":
            self.classifier = LogisticRegression(random_state=param["random_state"])

        elif self.clf_name == "KNN":
            self.classifier = KNeighborsClassifier(n_neighbors=param["n_neighbors"],
                                                   metric=param["metric"],
                                                   p=param["p"])

        elif self.clf_name == "SVM":
            self.classifier = SVC(kernel=param["kernel"],
                                  random_state=param["random_state"])

        elif self.clf_name == "Naive Bayes":
            self.classifier = GaussianNB()

        elif self.clf_name == "Decision Tree":
            self.classifier = DecisionTreeClassifier(criterion=param["criterion"],
                                                     random_state=param["random_state"])

        else:
            self.classifier = RandomForestClassifier(n_estimators=param["n_estimators"],
                                                     criterion=param["criterion"],
                                                     random_state=param["random_state"])

        # Write model parameters
        para_dict = self.classifier.get_params()
        used_param = pd.DataFrame([para_dict], index=["Parameters"])
        st.markdown("""**Parameters:** Be sure to scroll to review all parameters.""")
        st.write(used_param)

        return self.classifier

    def build_predict_accuracy(self):
        unscaled_feature = self.pre_process.split_data()
        scaled_feature = self.pre_process.feature_scale()
        classifier = self.set_classifier()
        classifier.fit(scaled_feature[0], unscaled_feature[2])
        y_pred = classifier.predict(scaled_feature[1])

        con_matrix = pd.DataFrame(confusion_matrix(unscaled_feature[3], y_pred),
                                  columns=["Predicted: No", "Predicted: Yes"],
                                  index=["Actual: No", "Actual: Yes"])
        accuracy = accuracy_score(unscaled_feature[3], y_pred)

        return con_matrix, accuracy
