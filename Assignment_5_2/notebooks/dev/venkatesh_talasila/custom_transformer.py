from sklearn.base import BaseEstimator

class CustomTransformer(BaseEstimator):
    def __init__(self):
        """Object creation for the class
        
        """
        pass
    
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        """Transformation to standardize the data

        Parameters
        ----------
            X : pd.DataFrame
                data to be transformed
                
        Returns
        -------
            pd.DataFrame
                dataframe with transformed values of the data            
        """
        X_tr = (X - X.mean()) / X.std()
        return X_tr
    
    def inverse_transform(self, X, X_tr):
        """Inverse transformation to return the original data

        Parameters
        ----------
            X : pd.DataFrame
            X_tr : pd.DataFrame
                transformed_data
            
        Returns
        -------
            pd.DataFrame
                dataframe with original values of the data            
        """
        X_tr = (X_tr * X.std()) + X.mean()
        return X_tr