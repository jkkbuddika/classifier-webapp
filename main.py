####################
# Import libraries #
####################

import streamlit as st
import pandas as pd
import PreProcessing
import Classifier
import Plotter

####################
# Customize the ui #
####################

# Set page config
apptitle = 'Classifier App'
st.set_page_config(page_title=apptitle, page_icon=":eyeglasses:")

# Hide the menu bar
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

######################
# Build the ui: Main #
######################

st.write("""
# Classification Web Application
This web app allows users to train a classification model on input data.
""")

##########################
# Build the ui: Side bar #
##########################

# Set the sidebar header
st.sidebar.title("User Input Parameters")

###############
# Handle Data #
###############

st.sidebar.header("Input Data")
example_data = "https://raw.githubusercontent.com/jkkbuddika/classifier_webapp/main/data/breast_cancer.csv"
st.sidebar.markdown(f"""[Example CSV Input File]({example_data})""")

# Upload data
uploaded_data = st.sidebar.file_uploader("Upload your CSV Input File", type=["csv"])

# Handle data if not uploaded
if uploaded_data is not None:
    data_set = pd.read_csv(uploaded_data)
else:
    st.write("Using example data as input: Breast cancer data set from UCI ML Repository")
    with st.expander("See notes"):
        st.markdown("""
        Details about the data set!
        """)
    data_set = pd.read_csv(example_data)

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

# Train and evaluate the classification model
cla_algo = classifier_algorithm()
cla_algo = cla_algo.iloc[0, 0]
st.subheader("Classifier and Parameters")
st.markdown(f"""**Classifier:** {cla_algo}""")
classifier = Classifier.Classifier(cla_algo, data_set)
model = classifier.build_predict_accuracy()

# Display model accuracy
st.subheader("Model Accuracy")
st.markdown(f"""**Confusion Matrix:**""")
st.write(model[0])
st.markdown(f"""**Accuracy Score:** {round(model[1], 5)}""")

# Visualize data using dimension reduction
st.subheader("Visualization following Dimension Reduction")
pre_pro = PreProcessing.PreProcessing(data_set)
prin_comp = pre_pro.dimension_reduction()
plot = Plotter.Plotter(x_data=prin_comp[0], y_data=prin_comp[1],
                       title="Visualized Data: Classification",
                       x_label="PC1", y_label="PC2",
                       color_array=prin_comp[3], color_map="Dark2")
plot.plot_scatter()

# Add social media information
st.write("""[![Star](https://img.shields.io/github/stars/jkkbuddika/classifier_webapp.svg?logo=github&style=social)](https://gitHub.com/jkkbuddika/classifier_webapp) \
[![Follow](https://img.shields.io/twitter/follow/KasunBuddika7?style=social)](https://www.twitter.com/KasunBuddika7)""")