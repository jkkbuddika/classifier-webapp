####################
# Import libraries #
####################

import streamlit as st
import pandas as pd
import PreProcessor
import Classifier
import Plotter

#################################################
# Build and customize the ui: Main and side bar #
#################################################

# Set page config
apptitle = "Classifier App"
st.set_page_config(page_title=apptitle, page_icon=":eyeglasses:")

# Hide the menu bar
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

# Name and the header of the web app
st.write("""
# Classification Web Application
This web app allows users to train a classification model on input data. Please note that users can tune some \
frequently altered parameters to get the maximum accuracy.
""")

# Set the sidebar header
st.sidebar.title("User Input Parameters")

###############
# Handle Data #
###############

# Data input
st.sidebar.header("Input Data")
example_data = "https://raw.githubusercontent.com/jkkbuddika/classifier_webapp/main/data/breast_cancer.csv"
st.sidebar.markdown(f"""[Example CSV Input File]({example_data})""")

# Upload data
uploaded_data = st.sidebar.file_uploader("Upload your CSV Input File", type=["csv"])
st.subheader("Input Data")

# Handle data if not uploaded
if uploaded_data is not None:
    data_set = pd.read_csv(uploaded_data)

else:
    st.markdown("""Using example data as input: **Breast cancer** data set from UCI ML Repository""")
    with st.expander("Read details about the data set"):
        st.markdown("""
        The example data set was adapted from the UCL Breast Cancer Wisconsin (Original) Data Set. \
        To read details about the data set please click \
        [here](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29).
        
        **Attribute Information:**

        1. Sample code number: id number
        2. Clump Thickness: 1 - 10
        3. Uniformity of Cell Size: 1 - 10
        4. Uniformity of Cell Shape: 1 - 10
        5. Marginal Adhesion: 1 - 10
        6. Single Epithelial Cell Size: 1 - 10
        7. Bare Nuclei: 1 - 10
        8. Bland Chromatin: 1 - 10
        9. Normal Nucleoli: 1 - 10
        10. Mitoses: 1 - 10
        11. Class: 2 for benign, 4 for malignant

        **Note:** The sample code number has been removed when creating the example data set. Moreover, \
        the provided example data set has been pre-processed.
        """)
    data_set = pd.read_csv(example_data)

# Print input data summary
st.markdown(f"""
**Input data summary**

- Number of features: {len(data_set.columns)}
- Number of instances: {len(data_set)}
""")

# Handle classifier
def classifier_algorithm():
    st.sidebar.subheader("Select the Classifier")
    algorithm = st.sidebar.selectbox("Classifier", ("Logistic Regression",
                                                    "KNN",
                                                    "SVM",
                                                    "Naive Bayes",
                                                    "Decision Tree",
                                                    "Random Forest"))
    input_para = {"algorithm": algorithm}
    parameters = pd.DataFrame(input_para, index=["Parameters"])
    return parameters

# Classification algorithm
cla_algo = classifier_algorithm()
cla_algo = cla_algo.iloc[0, 0]
st.subheader("Classifier and Parameters")
st.markdown(f"""**Classifier:** {cla_algo}""")

# Train the model and predict using test data
classifier = Classifier.Classifier(cla_algo, data_set)
model = classifier.build_predict_accuracy()

# Display model accuracy
st.subheader("Model Accuracy")
st.markdown(f"""**Confusion Matrix:**""")
st.write(model[0])
st.markdown(f"""**Accuracy Score:** {round(model[1], 5)}""")

# Visualize data using dimension reduction
st.subheader("Visualization following Dimension Reduction")
pre_pro = PreProcessor.PreProcessor(data_set)
prin_comp = pre_pro.dimension_reduction()
plot = Plotter.Plotter(x_data=prin_comp[0], y_data=prin_comp[1],
                       title="Visualized Data: Classification",
                       x_label="PC1", y_label="PC2",
                       color_array=prin_comp[3], color_map="Dark2")
plot.plot_scatter()

# Add social media information
st.write("""[![Star](https://img.shields.io/github/stars/jkkbuddika/classifier_webapp.svg?logo=github&style=social)](https://gitHub.com/jkkbuddika/classifier_webapp) \
[![Follow](https://img.shields.io/twitter/follow/KasunBuddika7?style=social)](https://www.twitter.com/KasunBuddika7)""")
