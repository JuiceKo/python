N = 4
TERMINAL_STATES = [0, 15]
ACTIONS = [0, 1, 2, 3]
GAMMAS = [1.0, 0.9, 0.8]
THETA = 1e-4


def get_next_state(s, action, n, terminal_states):
    if s in terminal_states:
        return s

    row = s // n
    col = s % n

    new_row = row
    new_col = col

    if action == 0:
        new_row -= 1
    elif action == 1:
        new_col += 1
    elif action == 2:
        new_row += 1
    elif action == 3:
        new_col -= 1

    if new_row < 0 or new_row >= n or new_col < 0 or new_col >= n:
        return s

    return new_row * n + new_col


def value_iteration(n, gamma, theta, terminal_states):
    num_states = n * n
    V = [0.0 for _ in range(num_states)]
    iteration = 0

    while True:
        delta = 0.0
        new_V = V[:]

        for s in range(num_states):
            if s in terminal_states:
                continue

            best_value = None

            for a in ACTIONS:
                next_s = get_next_state(s, a, n, terminal_states)
                reward = -1.0
                value = reward + gamma * V[next_s]

                if best_value is None or value > best_value:
                    best_value = value

            new_V[s] = best_value

            diff = abs(new_V[s] - V[s])
            if diff > delta:
                delta = diff

        V = new_V
        iteration += 1

        if delta <= theta:
            break

    policy = [-1 for _ in range(num_states)]

    for s in range(num_states):
        if s in terminal_states:
            policy[s] = None
            continue

        best_value = None
        best_action = 0

        for a in ACTIONS:
            next_s = get_next_state(s, a, n, terminal_states)
            reward = -1.0
            value = reward + gamma * V[next_s]
            if best_value is None or value > best_value:
                best_value = value
                best_action = a

        policy[s] = best_action

    return V, policy, iteration


def policy_evaluation(policy, n, gamma, theta, terminal_states, max_iterations=1000):
    num_states = n * n
    V = [0.0 for _ in range(num_states)]

    k = 0
    while True:
        delta = 0.0
        new_V = V[:]

        for s in range(num_states):
            if s in terminal_states:
                continue

            a = policy[s]
            next_s = get_next_state(s, a, n, terminal_states)
            reward = -1.0

            new_V[s] = reward + gamma * V[next_s]

            diff = abs(new_V[s] - V[s])
            if diff > delta:
                delta = diff

        V = new_V
        k += 1


        if delta <= theta or k >= max_iterations:
            break

    return V


def policy_iteration(n, gamma, theta, terminal_states):
    num_states = n * n

    policy = [0 for _ in range(num_states)]
    for ts in terminal_states:
        policy[ts] = None

    stable = False
    iterations = 0

    while not stable:
        iterations += 1

        V = policy_evaluation(policy, n, gamma, theta, terminal_states)

        stable = True

        for s in range(num_states):
            if s in terminal_states:
                continue

            old_action = policy[s]
            best_action = old_action
            best_value = None

            for a in ACTIONS:
                next_s = get_next_state(s, a, n, terminal_states)
                reward = -1.0
                value = reward + gamma * V[next_s]

                if best_value is None or value > best_value:
                    best_value = value
                    best_action = a

            policy[s] = best_action

            if best_action != old_action:
                stable = False

    return V, policy, iterations



def action_to_symbol(a):
    if a == 0:
        return "^"
    if a == 1:
        return ">"
    if a == 2:
        return "v"
    if a == 3:
        return "<"
    return "T"       # terminal


def print_values(V, n):
    for r in range(n):
        line = ""
        for c in range(n):
            s = r * n + c
            line += f"{V[s]:6.2f} "
        print(line)
    print()


def print_policy(policy, n):
    for r in range(n):
        line = ""
        for c in range(n):
            s = r * n + c
            if policy[s] is None:
                line += "  T  "
            else:
                line += "  " + action_to_symbol(policy[s]) + "  "
        print(line)
    print()


for gamma in GAMMAS:
    print("=====================================")
    print("VALUE ITERATION   - gamma =", gamma)
    print("=====================================")
    V_vi, pi_vi, it_vi = value_iteration(N, gamma, THETA, TERMINAL_STATES)
    print("Nombre d'itérations :", it_vi)
    print("Valeur optimale V(s) :")
    print_values(V_vi, N)
    print("Politique optimale π(s) :")
    print_policy(pi_vi, N)

for gamma in GAMMAS:
    print("=====================================")
    print("POLICY ITERATION  - gamma =", gamma)
    print("=====================================")
    V_pi, pi_pi, it_pi = policy_iteration(N, gamma, THETA, TERMINAL_STATES)
    print("Nombre d'itérations (policy iteration) :", it_pi)
    print("Valeur optimale V(s) :")
    print_values(V_pi, N)
    print("Politique optimale π(s) :")
    print_policy(pi_pi, N)

