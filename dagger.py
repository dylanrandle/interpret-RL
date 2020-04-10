from tqdm import tqdm
import numpy as np
from utils import generate_trajectory, evaluate_policy

def dagger(simulator, expert_policy, imitation_policy_generator, N=5, gamma=0.9, seed=42):
    """
    Create a policy that imitate the expert policy

    Params:
    ---------------------------------
    expert_policy : (state -> action)
        The expert policy
    imitation_policy_generator : (X_train, y_train) -> (state -> action)
        A generator that generates policy from the given dataset. (A policy is a function from state -> action)
    gamma : Float
        decay rate of beta
    N : int
        number of iterations
    """
    np.random.seed(seed)

    S_aggregate = []
    A_aggregate = []

    # Initialize optimal_policy as the expert policy
    # However, we don't want to return the expert policy back so we initialize
    # best_evaluation to something very low, so that any policy can take over
    # and become the optimal_policy
    optimal_policy = expert_policy
    best_evaluation = -1e10
    best_model = None
    evaluations = []

    pi_hat = expert_policy
    beta = 1

    for i in tqdm(range(N)):
        pi_i = lambda state: round(beta * expert_policy(state) + (1 - beta) * pi_hat(state))

        # generate trajectories using the new policy
        S1, A, R, S2 = generate_trajectory(pi_i, simulator)
        # get the expert actions for all the states visited by the new policy
        expert_actions = np.apply_along_axis(expert_policy, 1, S1).reshape(-1,1)

        # Train a new policy from the aggregated dataset
        S_aggregate.append(S1)
        A_aggregate.append(expert_actions)

        DX_train = np.vstack(S_aggregate)
        Dy_train = np.vstack(A_aggregate)

        pi_hat, model = imitation_policy_generator(DX_train, Dy_train)

        evaluation = evaluate_policy(pi_hat, simulator) # random or static??
        evaluations.append(evaluation)
        print(f'Iter {i}: Evaluation = {evaluation}')
        if evaluation >= best_evaluation:
            optimal_policy = pi_hat
            best_evaluation = evaluation
            best_model = model

        beta = beta * gamma

    print(f"Best evaluation = {best_evaluation} ")
    return {"policy": optimal_policy, "model": best_model, "evals": evaluations}

def q_dagger(simulator, expert_policy, imitation_policy_generator, expert_max_q, expert_min_q, N=5, seed=42):
    """
    Create a policy that imitate the expert policy

    Params:
    ---------------------------------
    expert_policy : (state -> action)
        The expert policy
    imitation_policy_generator : (X_train, y_train) -> (state -> action)
        A generator that generates policy from the given dataset. (A policy is a function from state -> action)
    N : int
        number of iterations
    """
    np.random.seed(seed)

    S_aggregate = []
    A_aggregate = []
    W_aggregate = []

    # Initialize optimal_policy as the expert policy
    # However, we don't want to return the expert policy back so we initialize
    # best_evaluation to something very low, so that any policy can take over
    # and become the optimal_policy
    optimal_policy = expert_policy
    best_evaluation = -1e10
    best_model = None
    evaluations = []

    pi_hat = expert_policy

    for i in tqdm(range(N)):
        if i == 0:
            pi_i = lambda state: expert_policy(state)
        else:
            pi_i = lambda state: pi_hat(state)

        # generate trajectories using the new policy
        S1, A, R, S2 = generate_trajectory(pi_i, simulator)

        # get the expert actions for all the states visited by the new policy
        expert_actions = np.apply_along_axis(expert_policy, 1, S1).reshape(-1,1)
        expert_max_qvals = np.apply_along_axis(expert_max_q, 1, S1).reshape(-1,1)
        expert_min_qvals = np.apply_along_axis(expert_min_q, 1, S1).reshape(-1,1)

        # calculate Q-dagger loss function weights
        weights = (A != expert_actions) * (expert_max_qvals - expert_min_qvals)

        # Train a new policy from the aggregated dataset
        S_aggregate.append(S1)
        A_aggregate.append(expert_actions)
        W_aggregate.append(weights)

        DX_train = np.vstack(S_aggregate)
        Dy_train = np.vstack(A_aggregate)
        Dw_train = np.vstack(W_aggregate).flatten()

        pi_hat, model = imitation_policy_generator(DX_train, Dy_train, sample_weight=Dw_train)

        evaluation = evaluate_policy(pi_hat, simulator)
        evaluations.append(evaluation)
        print(f'Iter {i}: Evaluation = {evaluation}')
        if evaluation >= best_evaluation:
            optimal_policy = pi_hat
            best_evaluation = evaluation
            best_model = model

    print(f"Best evaluation = {best_evaluation} ")
    return {"policy": optimal_policy, "model": best_model, "evals": evaluations}

if __name__=="__main__":
    from utils import load_model_pkl, policy_from_q_model
    from utils import corels_policy_generator, decision_tree_policy_generator
    from simulators.hiv_simulator import HIVSimulator
    from simulators.waypoint import WaypointWorld

    # HIV Simulator
    # sim = HIVSimulator(perturb_bounds=(-1, 1))
    # exp = load_model_pkl("models/fqi-regressor-final.pkl")
    # actions = sim.binary_action_codes
    # exp_policy = policy_from_q_model(exp, actions)

    # Waypoint World
    sim = WaypointWorld()
    pi_b_waypoints = [(1, 2), (7, 4), (3, 6), (6, 8), (10, 10)]
    pi_blackbox_waypoints = [(1, 2), (2, 4), (3, 6), (6, 8), (10, 10)]

    def pi(s, waypoints):
        """ s = state = (x, y)"""
        s_x, s_y = s
        ind = int(s_y / 2)
        ind = ind - 1 if ind >= len(waypoints) else ind
        wp_x, wp_y = waypoints[ind]
        a_x = wp_x - s_x
        a_y = wp_y - s_y

        # NOTE: discrete actions for CORELS
        if np.abs(a_x) > np.abs(a_y):
            # left/right
            return 2 if a_x > 0 else 3
        else:
            # up/down
            return 1 if a_y > 0 else 0

    exp_policy = lambda s: pi(s, pi_blackbox_waypoints)

    print("Running Dagger")
    res = dagger(sim, exp_policy, corels_policy_generator, N=5)
    model = res['model']
    for i, m in enumerate(model):
        print(f'\nAction {i}')
        print(m.rl())
