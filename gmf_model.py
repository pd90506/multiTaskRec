import numpy as np
import pandas as pd
import tensorflow.keras as keras
from tensorflow.keras import backend as K

from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Lambda, Activation
from tensorflow.keras.layers import Embedding, Input, Dense, Multiply, Reshape, Flatten
from tensorflow.keras.optimizers import Adagrad, Adam, SGD, RMSprop
from tensorflow.keras.regularizers import l2

from Dataset import Dataset

def get_model(num_users=943, num_items=1682, factor=20, regs=[0.0001,0.0001]):
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')

    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = factor, name = 'user_embedding',
                                  embeddings_initializer = initializers.RandomNormal(), embeddings_regularizer = l2(regs[0]), input_length=1)
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = factor, name = 'item_embedding',
                                  embeddings_initializer = initializers.RandomNormal(), embeddings_regularizer = l2(regs[1]), input_length=1)   
    
    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MF_Embedding_User(user_input))
    item_latent = Flatten()(MF_Embedding_Item(item_input))
    
    # Element-wise product of user and item embeddings 
    predict_vector = Multiply()([user_latent, item_latent])
    
    # Final prediction layer
    #prediction = Lambda(lambda x: K.sigmoid(K.sum(x)), output_shape=(1,))(predict_vector)
    prediction = Dense(1, activation='relu', kernel_initializer='lecun_uniform', name = 'prediction')(predict_vector)
    
    model = Model(inputs=[user_input, item_input], 
                outputs=[prediction])

    return model


def get_train_instances(df):
    user_input = df['userId'].values
    item_input = df['itemId'].values
    label = df['rating'].values
    return user_input, item_input, label

def fit():
    learning_rate = 0.001
    path = 'Data/'   
    batch_size = 256 
    verbose = 1
    
    model = get_model(num_users=943, num_items=1682, factor=20, regs=[0.0001,0.0001])
    model.compile(optimizer=Adam(lr=learning_rate), loss='mean_squared_error', metrics=['mae', 'mse'])

    dataset = Dataset(path, size='100k')
    train, test = dataset.train_ratings, dataset.test_ratings
    train_user, train_item, train_label = get_train_instances(train)
    test_user, test_item, test_label = get_train_instances(test)
    for i in range(0,40):
        print('Iteration:{}'.format(i))
        hist = model.fit([np.array(train_user), np.array(train_item)], np.array(train_label), batch_size=batch_size, epochs=1, verbose=verbose, shuffle=True)
        test_loss = model.evaluate([test_user, test_item], test_label, batch_size=256)
        print(test_loss)


if __name__ == "__main__":
    fit()