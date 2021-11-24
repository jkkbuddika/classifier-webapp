from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class PreProcessor:
    '''
    Pre-process input data for ML algorithms.
    '''
    def __init__(self, data_set):
        self.data_set = data_set

    def split_attributes(self):
        X = self.data_set.iloc[:, :-1].values
        y = self.data_set.iloc[:, -1].values

        return X, y

    def split_data(self):
        variables = self.split_attributes()
        X_train, X_test, y_train, y_test = train_test_split(variables[0], variables[1], test_size=0.2, random_state=0)

        return X_train, X_test, y_train, y_test

    def feature_scale(self):
        sc = StandardScaler()
        split_data = self.split_data()
        X_train = sc.fit_transform(split_data[0])
        X_test = sc.transform(split_data[1])

        return X_train, X_test

    def dimension_reduction(self):
        pca = PCA(n_components=2)
        variables = self.split_attributes()
        X = variables[0]
        y = variables[1]
        features_projected = pca.fit_transform(X, y=None)
        features_PC1 = features_projected[:, 0]
        features_PC2 = features_projected[:, 1]

        return features_PC1, features_PC2, X, y
