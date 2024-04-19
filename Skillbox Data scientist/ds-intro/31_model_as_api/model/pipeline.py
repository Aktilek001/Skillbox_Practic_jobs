import pandas as pd
import numpy as np
import dill

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from datetime import datetime



def main():
    

    def filter_data(df):
        columns_to_drop = [
            'id',
            'url',
            'region',
            'region_url',
            'price',
            'manufacturer',
            'image_url',
            'description',
            'posting_date',
            'lat',
            'long'
        ]

        return df.drop(columns_to_drop, axis=1)


    def smooth_year(data):
        def calculate_outliers(data):
            Q1 = np.quantile(data, 0.25)
            Q3 = np.quantile(data, 0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return lower_bound, upper_bound
        
        lower_bound, upper_bound = calculate_outliers(data)
        data[data < lower_bound] = lower_bound
        data[data > upper_bound] = upper_bound
        return data.round()

    def short_model(x):
        if not pd.isna(x):
            return x.lower().split(' ')[0]
        else:
            return x
        

    def create_features(df):
        df['short_model'] = df['model'].apply(short_model)
        df['age_category'] = df['year'].apply(lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average'))
        df.drop('model', axis=1, inplace=True)
        return df
    
    
    df_original = pd.read_csv('data/homework.csv')
    df_original = pd.DataFrame(df_original)
    df = df_original.copy()
    
    preprocessor1 = Pipeline(steps=[
                                    ('filter', FunctionTransformer(filter_data)),
                                    ('create_features', FunctionTransformer(create_features))
    ])
    # df = preprocessor1.fit_transform(df)
    
    
    X = df.drop('price_category', axis=1)
    y = df['price_category']

    numerical_features = make_column_selector(dtype_include=['int64', 'float64'])
    categorical_features = make_column_selector(dtype_include=['object'])
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('smooth_year', FunctionTransformer(smooth_year)),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ])
    
    
    models = (
        LogisticRegression(random_state=42, max_iter=1000),
        RandomForestClassifier(random_state=42, n_estimators=1000),
        MLPClassifier(random_state=42, max_iter=1000)
    )
    
    
    best_score = .0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            ('filter', preprocessor1),
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy', error_score='raise')
        print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe
    best_pipe.fit(X, y)
    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')
    
    metadata = {
        'name': 'Car price prediction model',
        'author': 'Akty',
        'version': 1,
        'date': datetime.now(),
        'type': type(best_pipe.named_steps["classifier"]).__name__,
        'accuracy': best_score
    }

    with open('cars_pipe.pkl', 'wb') as file:
        dill.dump({'model': best_pipe, 'metadata': metadata}, file, recurse=True)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
