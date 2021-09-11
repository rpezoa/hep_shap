import pandas as pd
import shap


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adamax, Nadam
import talos
from talos.utils import lr_normalizer
from talos.metrics.keras_metrics import f1score, precision, recall 
from talos.utils import hidden_layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense

from talos.utils.best_model import best_model, activate_model

seed_=420
# Read data file
df = pd.read_csv("/Users/rpezoa/Documents/HEP_Data_Analysis/ATLASImbalanceLearning/data/higgs/phpZLgL9q.csv")
df.rename(columns = {'class': 'label'}, inplace = True)
# Removing last row containinng "?" values
df.drop(df.tail(1).index,inplace=True) # drop last n rows
df = df.apply(pd.to_numeric)
# Pandas dataframe for correlation matrix without label column
df_corr = df.drop('label', inplace=False, axis=1)

# Scaling data
y = df["label"]
X = df.iloc[:,1:]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)
df_scaled = pd.DataFrame(scaled_data, columns=X.columns)


# Training, validation, and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed_)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, shuffle = True, test_size=0.2, random_state=seed_)

p = {'lr': [0.001],
     'activation':['relu'],
     'optimizer': [SGD],
     'loss': ['binary_crossentropy'],
     'shapes': ['brick'],
     'first_neuron': [4],
     'hidden_layers':[0, 1, 2],
     'dropout': [0, .5],
     'batch_size': [32],
     'epochs': [20],
     'last_activation': ['sigmoid'],
    'weight_regulizer': [None]}


# define the input model
def higgs_model(x_train, y_train, x_val, y_val, params):
    model = Sequential()
    model.add(Dense(params['first_neuron'],
                    input_dim=x_train.shape[1],
                    activation=params['activation']))

    model.add(Dropout(params['dropout']))

    hidden_layers(model, params, 1)

    model.add(Dense(1, activation=params['last_activation']))

    model.compile(optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
                  loss=params['loss'],
                  metrics=[f1score,precision, recall])

    out = model.fit(x_train, y_train,
                    batch_size=params['batch_size'],
                    epochs=params['epochs'],
                    verbose=0,
                    validation_data=(x_val, y_val))
    return out, model


scan_object = talos.Scan(x = X_train.to_numpy(),
                         y = y_train.to_numpy(),
                         x_val=X_val.to_numpy(),
                         y_val=y_val.to_numpy(),
                         params=p,
                         model=higgs_model,
                         experiment_name='higgs_4')

talos.Deploy(scan_object, model_name='higgs_deploy_4',metric='f1score')

best_idx = scan_object.best_model(metric='val_f1score', asc=True)

model_id = scan_object.data['val_f1score'].astype('float').argmax()

print(model_id)

model = activate_model(scan_object, model_id)

model.save("higgs_model_4.h5")

