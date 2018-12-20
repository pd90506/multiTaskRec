import tensorflow as tf
from tensorflow.keras import backend as K

from tensorflow.keras import initializers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, Dense, Multiply, Flatten
from tensorflow.keras.regularizers import l2
from Dataset import Dataset, get_train_instances_genre


def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))


def get_model(num_users=943, num_items=1682, factor=20, regs=[0.0001, 0.0001]):
    # Input variables
    num_task = 19
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')
    task_input = Input(shape=(num_task,), dtype='float', name='genre_input')

    MF_Embedding_User = Embedding(
            input_dim=num_users,
            output_dim=factor,
            name='user_embedding',
            embeddings_initializer=initializers.RandomNormal(),
            embeddings_regularizer=l2(regs[0]), input_length=1)
    MF_Embedding_Item = Embedding(
            input_dim=num_items,
            output_dim=factor,
            name='item_embedding',
            embeddings_initializer=initializers.RandomNormal(),
            embeddings_regularizer=l2(regs[1]),
            input_length=1)
    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MF_Embedding_User(user_input))
    item_latent = Flatten()(MF_Embedding_Item(item_input))

    # Create user preference coefficients based on number of tasks
    user_pref = Dense(
            units=num_task,
            activation='relu',
            kernel_initializer='lecun_uniform',
            name='user_pref')(user_latent)
    
    # Element-wise product of user and item embeddings
    predict_vector = Multiply()([user_latent, item_latent])
    
    # Final prediction layer
    prediction = Dense(
            units=num_task,
            activation='relu',
            kernel_initializer='lecun_uniform',
            name='prediction')(predict_vector)

    prediction = Multiply()([prediction, task_input])
    
    model = Model(
            inputs=[user_input, item_input, task_input],
            outputs=[prediction])

    return model