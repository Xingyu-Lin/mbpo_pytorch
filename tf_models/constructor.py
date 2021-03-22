import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from tf_models.fc import FC
from tf_models.bnn import BNN


def construct_model(obs_dim=11, act_dim=3, rew_dim=1, hidden_dim=200, num_networks=7, num_elites=5, session=None):
    print('[ BNN ] Observation dim {} | Action dim: {} | Hidden dim: {}'.format(obs_dim, act_dim, hidden_dim))
    params = {'name': 'BNN', 'num_networks': num_networks, 'num_elites': num_elites, 'sess': session}
    model = BNN(params)

    model.add(FC(hidden_dim, input_dim=obs_dim + act_dim, activation="swish", weight_decay=0.000025))
    model.add(FC(hidden_dim, activation="swish", weight_decay=0.00005))
    model.add(FC(hidden_dim, activation="swish", weight_decay=0.000075))
    model.add(FC(hidden_dim, activation="swish", weight_decay=0.000075))
    model.add(FC(obs_dim + rew_dim, weight_decay=0.0001))
    model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})
    return model


def format_samples_for_training(samples):
    obs = samples['observations']
    act = samples['actions']
    next_obs = samples['next_observations']
    rew = samples['rewards']
    delta_obs = next_obs - obs
    inputs = np.concatenate((obs, act), axis=-1)
    outputs = np.concatenate((rew, delta_obs), axis=-1)
    return inputs, outputs


def reset_model(model):
    model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model.name)
    model.sess.run(tf.initialize_vars(model_vars))


if __name__ == '__main__':
    num_networks = 7
    num_elites = 5
    state_size = 17
    action_size = 6
    reward_size = 1
    pred_hidden_size = 200

    model = construct_model(obs_dim=state_size, act_dim=action_size, hidden_dim=pred_hidden_size, num_networks=num_networks,
                            num_elites=num_elites)
    variables_names = [v.name for v in tf.trainable_variables()]
    values = model._sess.run(variables_names)
    # tf_weights = {}
    # for k, v in zip(variables_names, values):
    #     print("Variable: ", k)
    #     print("Shape: ", v.shape)
    #     tf_weights[k] = v
    #     # print(v)
    # with open('tf_weights.pkl', 'wb') as f:
    #     pickle.dump(tf_weights, f)
    # exit()
    import pickle
    with open('tf_weights.pkl', 'rb') as f:
        tf_weights = pickle.load(f)
    for v in tf.global_variables():
        if v.name in tf_weights.keys():
            model._sess.run(v.assign(tf_weights[v.name]))
            print(v, v.name)
    # exit()
    BATCH_SIZE = 5250
    import time
    st_time = time.time()
    with open('test.npy', 'rb') as f:
        train_inputs = np.load(f)
        train_labels = np.load(f)
    for i in range(0, 1000, BATCH_SIZE):
        # train_inputs = np.random.random([BATCH_SIZE, state_size + action_size]).astype(np.float32)
        # train_labels = np.random.random([BATCH_SIZE, state_size + 1]).astype(np.float32)
        # model.predict(train_inputs[:100])
        model.train(train_inputs, train_labels, batch_size=256, holdout_ratio=0.2)
        # exit()
        # np.set_printoptions(precision=7)
        # mean, var = model.predict(train_inputs[:100])
        # print(mean)
        # print(mean.shape)
        # print(np.mean(mean))
        # print(var.shape)
        # print(mean[0])
        # print(var[0])
        # exit()

    print(time.time()-st_time)
    # model.predict(Variable(torch.from_numpy(test_inputs[:1000])))
