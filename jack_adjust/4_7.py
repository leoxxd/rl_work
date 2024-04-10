import numpy as np
import pickle
import matplotlib.pyplot as plt

def poisson_calculator(Lambda=3):
    result = {}
    for i in range(0, 21):
        result[i] = max(np.finfo(float).eps, (np.power(Lambda, i) / (np.math.factorial(i))) * np.exp(-Lambda))
    return result

def P_calculate(all_possibility):
    for state_value_A in range(21):
        print("State " + str(state_value_A))
        for state_value_B in range(21):
            P = {}
            for action in range(-5, 6):
                temp = {}
                if action <= state_value_A and -action <= state_value_B and action + state_value_B <= 20 and -action + state_value_A <= 20:
                    for customer_A in range(21):
                        for customer_B in range(21):
                            for returned_car_A in range(21):
                                for returned_car_B in range(21):
                                    # Adjust for the free move from A to B by an employee
                                    if action == -1:  # 免费从A移动到B一辆车
                                        action_cost = 0
                                    elif action < 0:  # 从A到B移动多于一辆车
                                        action_cost = 2 * (abs(action) - 1)  # 第一辆免费，其余每辆收费2美元
                                    else:  # 从B到A移动车辆或者没有移动
                                        action_cost = 2 * abs(action)  # 每辆车收费2美元
                                    # Calculate extra parking cost
                                    extra_parking_cost_A = 4 if state_value_A - action > 10 else 0
                                    extra_parking_cost_B = 4 if state_value_B + action > 10 else 0
                                    
                                    value_A_Changed = min(20, state_value_A - min(customer_A, state_value_A - action) + returned_car_A - action)
                                    value_B_Changed = min(20, state_value_B - min(customer_B, state_value_B + action) + returned_car_B + action)
                                    reward = 10 * min(customer_A, state_value_A - action) + \
                                             10 * min(customer_B, state_value_B + action) - \
                                             action_cost - extra_parking_cost_A - extra_parking_cost_B
                                    
                                    temp[((value_A_Changed, value_B_Changed),reward)] = temp.get((value_A_Changed, value_B_Changed), 0) + all_possibility[(customer_A, returned_car_A, customer_B, returned_car_B)]
                    P[action] = temp
            with open('P' + str(state_value_A)+str('_')+str(state_value_B), 'wb') as f:
                pickle.dump(P, f, protocol=-1)
def policy_evaluation(V, pi, Theta):
    counter = 1
    while True:
        Delta = 0
        print("Calculating loop " + str(counter))
        for i in range(21):
            print("----Calculating " + str(i))
            for j in range(21):
                with open('P' + str(i)+str('_')+str(j), 'rb') as f:
                    p = pickle.load(f)
                a = pi[(i, j)]
                p = p[a]
                old_value = V[(i, j)]
                V[(i, j)] = 0
                for keys, values in p.items():
                    (states, reward) = keys
                    possibility = values
                    V[(i, j)] += (reward + 0.9 * V[states]) * possibility
                Delta = max(Delta, abs(V[(i, j)] - old_value))
        print("Delta = " + str(Delta))
        if Delta < Theta:
            return V
        counter += 1
# The rest of the functions remain unchanged
def policy_improvement(V, pi={}):
    counter = 1
    while True:
        print("Calculating policy loop " + str(counter))
        policy_stable = True
        for keys, old_action in pi.items():
            with open('P' + str(keys[0])+str('_')+str(keys[1]), 'rb') as f:
                p = pickle.load(f)
            possible_q = [0] * 11
            [state_value_A, state_value_B] = keys
            for possible_action in range(-5, 6):
                index = possible_action + 5
                if possible_action <= state_value_A and -possible_action <= state_value_B and possible_action + state_value_B <= 20 and -possible_action + state_value_A <= 20:
                    # print(possible_action)
                    # print(state_value_A, state_value_B)
                    p_a = p[possible_action]
                    for p_keys, values in p_a.items():
                        [states, reward]=p_keys
                        possibility = values
                        possible_q[index] += (reward + 0.9 * V[states]) * possibility
                else:
                    possible_q[index] = -999
            pi[keys] = np.argmax(possible_q) - 5
            if pi[keys] != old_action:
                policy_stable = False
        if policy_stable:
            return pi
        counter += 1                

def init():
    # Initialize possibilities for customers and returns
    customer_A = poisson_calculator(3)
    customer_B = poisson_calculator(4)
    return_A = poisson_calculator(3)
    return_B = poisson_calculator(2)
    all_possibility = {}
    for i in range(21):
        for j in range(21):
            for m in range(21):
                for n in range(21):
                    all_possibility[(i, j, m, n)] = customer_A[i] * return_A[j] * customer_B[m] * return_B[n]
    with open('all_possibility', 'wb') as f:
        pickle.dump(all_possibility, f, protocol=-1)
    P_calculate(all_possibility)

def train():
    V = {}
    for i in range(21):
        for j in range(21):
            V[(i, j)] = 10 * np.random.random()
    pi = {}
    for i in range(21):
        for j in range(21):
            pi[(i, j)] = 0
    for q in range(15):
        print("Big loop "+str(q))
        V = policy_evaluation(V, pi, Theta=0.01)
        pi = policy_improvement(V, pi)
        with open('pi'+str(q), 'wb') as f:
            pickle.dump(pi, f, protocol=-1)
        with open('V'+str(q), 'wb') as v:
            pickle.dump(V, v, protocol=-1)
        print("================")
        for i in range(21):
            print("i = " + str(i))
            for j in range(21):
                print("  " + str(pi[i, j]))

def main():
    init() 
    train()

if __name__ == "__main__":
    main()
