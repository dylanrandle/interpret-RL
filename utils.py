import pandas as pd
import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

from corels import *

def load_model_pkl(path):
    """ loads a model stored as pickle file (e.g. from sklearn) """
    with open(path, 'rb') as f:
        return pickle.load(f)

def policy_from_q_model(model, actions):
    """ returns a policy function for a Q-function-based model """
    def policy(state):
        return np.argmax(
            [model.predict(np.hstack((state, a)).reshape(1, -1)) \
            for a in actions])
    return policy

def generate_trajectory(policy, simulator, T=200):
    """
    Generate 1 trajectory of length T from the given policy
    """
    states = []
    actions = []
    rewards = []
    next_states = []

    simulator.reset()
    state = simulator.observe()

    for _ in range(T):
        a = policy(state)
        r, next_state = simulator.perform(a)

        states.append(state)
        actions.append(a)
        rewards.append(r)
        next_states.append(next_state)

        state = next_state

    states = np.vstack(states)
    actions = np.vstack(actions)
    rewards = np.vstack(rewards)
    next_states = np.vstack(next_states)

    return states, actions, rewards, next_states

def evaluate_policy(policy, simulator, T=200):
    """
    Evaluate the given policy by calculating the cumulative reward of
    of a T-step trajectory generated from this policy
    """
    simulator.reset()
    state = simulator.observe()
    cumulative_reward = 0
    for i in range(T):
        action = policy(state)
        reward, state = simulator.perform(action)
        cumulative_reward += reward

    return cumulative_reward

def np_to_df(X):
    """ transform numpy matrix of states to
        dataframe with state labels """
    feature_names = ['state_{}'.format(i) for i in range(X.shape[1])]
    return pd.DataFrame(data=X, columns=feature_names)

def bin_states(X, nbins=5, use_edges=None):
    """
    inputs
        use_edges: dictionary (same as returned `bins`) to use instead of recomputing the bins
                  this should be used on the test data, as we want to use the same bins as fit
                  on the train data
    returns
        X: binned (from histogram) version of X (dataframe)
        save_edges: edges of bins for each bin (dictionary)


    """
    X = X.copy()
    edges_dict = dict()
    for s in range(X.shape[1]): # axis 1 assumed to contain states
        if not use_edges:
            freq, edges = np.histogram(X['state_%d' % s], bins=nbins)
            edges_dict['state_%d' % s] = edges
        else:
            edges = use_edges['state_%d' % s]

        for i in range(len(edges) - 1):
            X[f'State {s+1}: {edges[i]:.2} to {edges[i+1]:.2}'] = X['state_%d' % s].between(edges[i], edges[i+1])

        X.drop(columns=['state_%d' % s], inplace=True)

    return X, edges_dict if not use_edges else use_edges

def bin_states_pctile(X, pctiles=[5, 95], use_edges=None):
    """
    inputs
        use_edges: dictionary (same as returned `bins`) to use instead of recomputing the bins
                  this should be used on the test data, as we want to use the same bins as fit
                  on the train data
    returns
        X: binned (from histogram) version of X (dataframe)
        save_edges: edges of bins for each bin (dictionary)


    """
    if len(pctiles) > 2:
        raise RuntimeError('Expected only 2 pctiles at a time (to create low, medium, high reading)')

    X = X.copy()
    edges_dict = dict()

    for s in range(X.shape[1]): # axis 1 assumed to contain states
        if not use_edges:
            edges = np.percentile(X['state_%d' % s], pctiles)
            edges_dict['%d' % s] = edges
        else:
            edges = use_edges['%d' % s]

        X['state%d_low' % s] = X['state_%d' % s].between(X['state_%d' % s].min(), edges[0])
        X['state%d_normal' % s] = X['state_%d' % s].between(edges[0], edges[1])
        X['state%d_high' % s] = X['state_%d' % s].between(edges[1], X['state_%d' % s].max())

        X.drop(columns=['state_%d' % s], inplace=True)

    return X, edges_dict if not use_edges else use_edges

def extra_trees_classifier_policy_generator(X_train, y_train, sample_weight=None):
    """
    Generate policy using ExtraTreesClassifier

    @return (state -> action)
    """
    regressor = ExtraTreesClassifier()
    regressor.fit(X_train, y_train, sample_weight=sample_weight)
    policy = lambda s: regressor.predict(s.reshape(1,-1))[0]
    return policy, regressor

def logistic_classifier_policy_generator(X_train, y_train, sample_weight=None):
    """
    Generate policy using LogisticRegression

    @return (state -> action)
    """
    regressor = LogisticRegression()
    regressor.fit(X_train, y_train, sample_weight=sample_weight)
    policy = lambda s: regressor.predict(s.reshape(1,-1))[0]
    return policy, regressor

def decision_tree_policy_generator(X_train, y_train, max_depth=None, sample_weight=None):
    """
    Generate policy using DecisionTreeClassifier

    @return (state -> action)
    """
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X_train, y_train, sample_weight=sample_weight)
    policy = lambda s: clf.predict(s.reshape(1,-1))[0]

    return policy, clf

def corels_policy_generator(X_train, y_train, nbins=5, maxiter=10000,
    max_card=2, min_support=0.01, use_pctile=False, pctiles=[5, 95],
    c_reg=0.000001, sample_weight=None):
    """
    Generate policy using corels

    @return (state -> action)
    """
    if sample_weight is not None:
        if (sample_weight > 0).any():
            # resample to achieve sample weight effect
            sample_weight_norm = sample_weight / np.sum(sample_weight)
            samp_ids = np.random.choice(len(X_train), size=len(X_train), replace=True, p=sample_weight_norm)
            X_train = X_train[samp_ids, :]
            y_train = y_train[samp_ids, :]

    X_df = np_to_df(X_train)

    bin_fn = bin_states if not use_pctile else bin_states_pctile
    X_bin, edges_dict = bin_fn(X_df, nbins=nbins)

    y_train = y_train.flatten()
    most_common_action = np.argmax(np.bincount(y_train))
    feature_names_binned = list(X_bin.columns)

    models = []
    scores = []
    for a in range(0, 4):
        c = CorelsClassifier(n_iter=maxiter, max_card=max_card,
            min_support=min_support, c=c_reg,
            verbosity=[] #["loud", "samples"]
        )
        c.fit(X_bin, y_train == a, features=feature_names_binned)
        models.append(c)
        score = c.score(X_bin, y_train == a)
        scores.append(score)

    def policy(state):
        state_df = np_to_df(state.reshape(1,-1))
        state_bin, _ = bin_fn(state_df, use_edges=edges_dict)
        preds = []
        for i in range(4):
            preds.append(models[i].predict(state_bin))
        y_pred = np.vstack(preds).flatten()
        if np.max(y_pred) == 0:
            return most_common_action
        else:
            return np.argmax(y_pred * scores)

    return policy, models
