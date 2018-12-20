import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, Dense, Multiply, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from Dataset import Dataset, get_train_instances


def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))


def get_model(num_users=943, num_items=1682, factor=20, regs=[0.0001, 0.0001]):
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    MF_Embedding_User = Embedding(
        input_dim=num_users,
        output_dim=factor,
        name='user_embedding',
        embeddings_initializer=initializers.RandomNormal(),
        embeddings_regularizer=l2(regs[0]),
        input_length=1)
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
    # Element-wise product of user and item embeddings
    predict_vector = Multiply()([user_latent, item_latent])
    # Final prediction layer
    prediction = Dense(
        units=1,
        activation='relu',
        kernel_initializer='lecun_uniform',
        name='prediction')(predict_vector)
    model = Model(
        inputs=[user_input, item_input],
        outputs=[prediction])
    return model


def fit(data='ml-1m'):
    learning_rate = 0.001
    path = 'Data/'
    batch_size = 256
    verbose = 1
    factor = 16
    regs = [0.00001, 0.00001]
    loss = root_mean_squared_error
    metrics = ['mae', root_mean_squared_error]
    epochs = 50

    if data == 'ml-1m':
        num_users = 6040
        num_items = 3706
    elif data == 'ml-100k':
        num_users = 943
        num_items = 1682
    else:
        raise ValueError('No a valid dataset name.')
    model = get_model(
        num_users=num_users,
        num_items=num_items,
        factor=factor,
        regs=regs)
    model.compile(optimizer=Adam(lr=learning_rate), loss=loss, metrics=metrics)

    dataset = Dataset(path, size=data)
    train, test = dataset.train_ratings, dataset.test_ratings
    train_user, train_item, train_label = get_train_instances(train)
    test_user, test_item, test_label = get_train_instances(test)

    # start training iterations
    # init the best loss and iters
    best_loss, best_mae, best_mse = model.evaluate(
        x=[test_user, test_item],
        y=test_label,
        batch_size=batch_size)
    best_iter = -1

    for i in range(0, epochs):
        print('Iteration:{}'.format(i))
        hist = model.fit(
            x=[np.array(train_user), np.array(train_item)],
            y=np.array(train_label),
            batch_size=batch_size,
            epochs=1,
            verbose=verbose,
            shuffle=True)
        test_loss = model.evaluate(
            x=[test_user, test_item],
            y=test_label,
            batch_size=batch_size)
        print('The test MAE = {}, RMSE = {}'.format(
            test_loss[1], test_loss[2]))
        # select the best iter based on mse
        if test_loss[2] < best_mse:
            best_iter = i
            best_loss, best_mae, best_mse = test_loss
    print('The best iteration is {}, with MAE = {}, RMSE = {}'.format(
        best_iter, best_mae, best_mse))


if __name__ == "__main__":
    fit(data='ml-100k')
