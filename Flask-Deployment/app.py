import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline 
import django_heroku
django_heroku.settings(locals())


class FeatureSelector( BaseEstimator, TransformerMixin ):
    #Class Constructor 
    def __init__( self, feature_names ):
        self._feature_names = feature_names 
    
    #Return self nothing else to do here    
    def fit( self, X, y =None):
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None):
        return X[ self._feature_names ] 
    pass

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
    pass

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
    pass

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

full_pipeline = FeatureUnion( transformer_list = [ ( 'categorical_pipeline', categorical_pipeline ), ( 'numerical_pipeline', numerical_pipeline ) ] )

cat_cols = ['Response', 'EmploymentStatus', 'Number of Open Complaints',
       'Number of Policies', 'Policy Type', 'Renew Offer Type',
       'Vehicle Class']

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    z = {'rows': int_features}
    final_features = pd.DataFrame.from_dict(z, orient = 'index', columns = ['Response', 'EmploymentStatus','Monthly Premium Auto', 'Number of Open Complaints', 'Number of Policies','Policy Type', 'Renew Offer Type', 'Vehicle Class'])
    
    prediction = model.predict(final_features)
    
    output = round(np.exp(prediction[0]),2)

    return render_template('index.html', prediction_text='Customer Lifetime Value $ {}'.format(output))


if __name__ == "__main__":  

	model = pickle.load(open('model.pkl','rb'))

	print("loaded OK")

	app.run(debug=True)

    
