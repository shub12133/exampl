import numpy as np
import pandas as pd
import scipy.stats as stats
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline 
from sklearn.externals import joblib


df = pd.read_csv('Insurance_Marketing-Customer-Value-Analysis.csv')
df = df.drop(['Customer','Vehicle Size', 'Total Claim Amount','Income','Months Since Last Claim', 'Months Since Policy Inception', 'Marital Status','Coverage', 'Education','Effective To Date','State', 'Gender', 'Location Code', 'Policy', 'Sales Channel'] , axis = 1)
df.head()

#Custom Transformer that extracts columns passed as argument to its constructor 
class FeatureSelector( BaseEstimator, TransformerMixin ):
    #Class Constructor 
    def __init__( self, feature_names ):
        self._feature_names = feature_names 
    
    #Return self nothing else to do here    s
    def fit( self, X, y =None):
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None):
        return X[ self._feature_names ] 

LE = LabelEncoder()
class CategoricalTransformer( BaseEstimator, TransformerMixin ):
    #Class constructor method that takes in a list of values as its argument
    def __init__(self, cat_cols = ['Response', 'EmploymentStatus', 'Number of Open Complaints',
       'Number of Policies', 'Policy Type', 'Renew Offer Type',
       'Vehicle Class']):
        self._cat_cols = cat_cols
        
    #Return self nothing else to do here
    def fit( self, X, y = None  ):
        return self
    
    #Transformer method we wrote for this transformer 
    def transform(self, X , y = None ):

       if self._cat_cols:
           for i in X[cat_cols]:
            X[i]= LE.fit_transform(X[i])
        
       return X.values 

class NumericalTransformer(BaseEstimator, TransformerMixin):
    #Class Constructor
    def __init__( self, MPA_log = True):
        self._MPA_log = MPA_log
        
    #Return self, nothing else to do here
    def fit( self, X, y = None):
        return self 
    
    #Custom transform method we wrote that creates aformentioned features and drops redundant ones 
    def transform(self, X, y = None):
        if self._MPA_log:
            X['Monthly Premium Auto'] = np.log(X['Monthly Premium Auto'])

        
        return X.values


#Categrical features to pass down the categorical pipeline 
categorical_features = ['Response', 'EmploymentStatus', 'Number of Open Complaints',
       'Number of Policies', 'Policy Type', 'Renew Offer Type',
       'Vehicle Class']


#Defining the steps in the categorical pipeline 
categorical_pipeline = Pipeline( steps = [ ( 'cat_selector', FeatureSelector(categorical_features) ),( 'cat_transformer', CategoricalTransformer() )])
                                  
         

#Numerical features to pass down the numerical pipeline 
numerical_features = ['Monthly Premium Auto']

    
#Defining the steps in the numerical pipeline     
numerical_pipeline = Pipeline( steps = [ ( 'num_selector', FeatureSelector(numerical_features) ),
                                  
                                  ( 'num_transformer', NumericalTransformer() ) ] )

full_pipeline = FeatureUnion(transformer_list = [ ( 'categorical_pipeline', categorical_pipeline ), ( 'numerical_pipeline', numerical_pipeline ) ] )

cat_cols = ['Response', 'EmploymentStatus', 'Number of Open Complaints',
       'Number of Policies', 'Policy Type', 'Renew Offer Type',
       'Vehicle Class']

df.columns
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

y = np.ravel(np.log(y))

full_pipeline_RF = Pipeline( steps = [('full_pipeline', full_pipeline),('model', RandomForestRegressor(max_depth=21, min_samples_leaf= 8, random_state=0))])
full_pipeline_RF.fit(X,y)

# Saving model to disk
pickle.dump(full_pipeline_RF, open('model.pkl','wb'))

#joblib.dump(full_pipeline_RF, 'scaled_tree_clf.pkl') 

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

z = {'rows': [0,1,78,0,1,1,1,0]}
type(z)
z.values()

z1 = pd.DataFrame.from_dict(z, orient = 'index', columns = ['Response', 'EmploymentStatus','Monthly Premium Auto', 'Number of Open Complaints', 'Number of Policies','Policy Type', 'Renew Offer Type', 'Vehicle Class'])
z1

np.exp(model.predict(z1))

