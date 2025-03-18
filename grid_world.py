import numpy as np
import random
import matplotlib.pyplot as plt
from decorator import measure_performance

class GridWorld():
    def __init__(self):
        self.gamma = 0.9
        self.action_list = {0: "↑", 1: ">", 2: "↓", 3: "<"}
        self.action_map = [[-1, 0], [0, 1], [1, 0], [0, -1]]  # ↑, >, ↓, <
        self.value_map = np.zeros((4, 4))  # 가치 맵
        self.policy_map = np.full((4, 4), " ", dtype=object)  # 정책 맵
        self.reward_map = np.array([
            [0,  1, -1,  0],
            [0,  1,  0,  0],
            [0, -1, -2,  0],
            [0,  0,  0,  4]
        ])
        self.theta = 1e-4  
        self.value_history = []
    
    def update_state(self, state, action):
        """ 주어진 상태(state)에서 action을 수행한 후의 새로운 상태 반환 """
        new_x = state[0] + action[0]
        new_y = state[1] + action[1]

        if new_x < 0 or new_x >= 4 or new_y < 0 or new_y >= 4:
            return state, -1  # 벗어나면 원래 상태로 돌아옴
        
        return [new_x, new_y], 0  # 새로운 상태

    @measure_performance
    def default_iteration(self):
        """ 무작위 정책 하에서 가치 반복법 (기대값 기반) """
        for _ in range(200):  
            new_value_map = self.value_map.copy()  
            delta = 0

            for x in range(4):
                for y in range(4):  
                    state = [x, y]
                    expect_value = 0
                    
                    # 모든 행동에 대해 기대값 계산 (무작위 정책: 0.25 확률)
                    for act in self.action_list.keys():
                        next_state, penalty = self.update_state(state, self.action_map[act])  
                        reward = self.reward_map[next_state[0], next_state[1]] + penalty 
                        expect_value += 0.25 * (reward + self.gamma * self.value_map[next_state[0], next_state[1]])
                    
                    new_value_map[x, y] = expect_value
                    delta = max(delta, abs(expect_value - self.value_map[x, y]))
            
            self.value_map = new_value_map
            if delta < self.theta:
                break
            
        return self.value_map
    
    @measure_performance
    def value_iteration(self):
        """ 가치 반복법 (최적 가치 함수 계산) """
        while True:
            delta = 0  
            new_value_map = self.value_map.copy()  

            for x in range(4):
                for y in range(4):  
                    state = [x, y]
                    best_value = -np.inf  

                    for act in self.action_list.keys():
                        next_state, penalty = self.update_state(state, self.action_map[act])  
                        reward = self.reward_map[next_state[0], next_state[1]] + penalty 
                        expect_value = reward + self.gamma * self.value_map[next_state[0], next_state[1]]

                        if expect_value > best_value:
                            best_value = expect_value

                    new_value_map[x, y] = best_value
                    delta = max(delta, abs(best_value - self.value_map[x, y]))  

            self.value_map = new_value_map
            if delta < self.theta:  
                break 
        return self.value_map  

    @measure_performance
    def policy_iteration(self):
        """ 정책 반복법 """
        # 초기 무작위 정책 설정
        policy = np.random.randint(0, 4, size=(4, 4))  # 0~3 사이의 행동
        
        while True:
            # 1. 정책 평가 (Policy Evaluation)
            while True:
                delta = 0
                new_value_map = self.value_map.copy()
                
                for x in range(4):
                    for y in range(4):
                        state = [x, y]
                        act = policy[x, y]
                        next_state, penalty = self.update_state(state, self.action_map[act])
                        reward = self.reward_map[next_state[0], next_state[1]] + penalty
                        value = reward + self.gamma * self.value_map[next_state[0], next_state[1]]
                        
                        new_value_map[x, y] = value
                        delta = max(delta, abs(value - self.value_map[x, y]))
                
                self.value_map = new_value_map
                if delta < self.theta:
                    break
            
            # 2. 정책 개선 (Policy Improvement)
            policy_stable = True
            for x in range(4):
                for y in range(4):
                    state = [x, y]
                    old_action = policy[x, y]
                    best_action = old_action
                    best_value = -np.inf
                    
                    for act in self.action_list.keys():
                        next_state, penalty = self.update_state(state, self.action_map[act])
                        reward = self.reward_map[next_state[0], next_state[1]] + penalty
                        value = reward + self.gamma * self.value_map[next_state[0], next_state[1]]
                        
                        if value > best_value:
                            best_value = value
                            best_action = act
                    
                    policy[x, y] = best_action
                    if old_action != best_action:
                        policy_stable = False
            
            if policy_stable:
                break
        
        # 정책 맵을 문자열로 변환
        for x in range(4):
            for y in range(4):
                self.policy_map[x, y] = self.action_list[policy[x, y]]
        
        return self.value_map

    def draw_policy_map(self, value_map):
        for x in range(4):
            for y in range(4):
                state = [x, y]
                best_value = float('-inf')
                best_actions = []  # 최대 가치를 가진 행동들을 저장

                for act in range(4):  
                    next_state, _ = self.update_state(state, self.action_map[act])
                    # 벽에 부딪혀 제자리로 돌아오는 경우 제외
                    if next_state == state:
                        continue
                    value = value_map[next_state[0], next_state[1]]

                    if value > best_value:
                        best_value = value
                        best_actions = [act]  # 새로운 최대 가치 발견 시 리스트 초기화
                    

                # 최대 가치가 있는 모든 행동을 문자열로 결합
                if best_actions:  # 유효한 행동이 있는 경우만
                    self.policy_map[x, y] = ''.join(self.action_list[act] for act in best_actions)
                else:
                    self.policy_map[x, y] = " "  # 유효한 이동이 없으면 공백

        return self.policy_map

    def show_grid(self, title, value_map, policy_map):
        print(f"\n{title} Optimal Value Map:")
        print(np.round(value_map, 2))
        print(f"\n{title} Optimal Policy:")
        for row in policy_map:
            print(" ".join(row))

if __name__ == "__main__":
    grid = GridWorld()
    
    # Default Iteration
    default_value_map = grid.default_iteration()
    default_policy_map = grid.draw_policy_map(default_value_map)
    grid.show_grid(title="Default Iteration", value_map=default_value_map, policy_map=default_policy_map)

    # Value Iteration
    grid.value_map = np.zeros((4, 4))  # 가치 맵 초기화
    value_iteration_map = grid.value_iteration()
    value_iteration_policy_map = grid.draw_policy_map(value_iteration_map)
    grid.show_grid(title="Value Iteration", value_map=value_iteration_map, policy_map=value_iteration_policy_map)

    # Policy Iteration
    grid.value_map = np.zeros((4, 4))  # 가치 맵 초기화
    policy_iteration_map = grid.policy_iteration()
    grid.show_grid(title="Policy Iteration", value_map=policy_iteration_map, policy_map=grid.policy_map)