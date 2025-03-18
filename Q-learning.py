import random
import numpy as np
from grid_world import GridWorld
from decorator import measure_performance

class QLearning(GridWorld):
    def __init__(self, episodes, gamma=0.9):
        super().__init__()
        self.episodes = episodes  # 에피소드 수
        self.gamma = gamma
        self.alpha = 0.01  # 학습률
        # 상태-행동 가치 함수 Q(s, a) 초기화 (4x4 그리드, 4방향 행동)
        self.q_values = {
            (x, y, act): 0.0 
            for x in range(4) 
            for y in range(4) 
            for act in self.action_list.keys()
        }

    def choose_action(self, state, epsilon):
        """ ε-greedy로 행동 선택 """
        if random.random() < epsilon:
            return random.choice(list(self.action_list.keys()))
        else:
            best_action = None
            best_value = -np.inf
            for act in self.action_list.keys():
                q_value = self.q_values[(state[0], state[1], act)]
                if q_value > best_value:
                    best_value = q_value
                    best_action = act
            return best_action

    def generate_episode(self, epsilon):
        """ Q-Learning용 에피소드 생성 (상태, 행동, 보상, 다음 상태 포함) """
        state = [0, 0]
        episode = []

        while self.reward_map[state[0], state[1]] != 4:  # 종료 상태 도달 전까지
            action = self.choose_action(state, epsilon)
            print(action)
            next_state, penalty = self.update_state(state, self.action_map[action])
            reward = self.reward_map[next_state[0], next_state[1]] + penalty
            episode.append((state, action, reward, next_state))
            state = next_state

        # 종료 상태 추가
        episode.append((state, None, self.reward_map[state[0], state[1]], None))
        return episode

    def update_policy(self):
        """ Q(s, a)를 기반으로 최적 정책 계산 및 policy_map에 저장 """
        for x in range(4):
            for y in range(4):
                state = [x, y]
                if self.reward_map[x, y] == 4:  # 종료 상태
                    self.policy_map[x, y] = "★"
                    continue
                
                best_action = None
                best_value = -np.inf
                for act in self.action_list.keys():
                    q_value = self.q_values[(x, y, act)]
                    if q_value > best_value:
                        best_value = q_value
                        best_action = act
                
                self.policy_map[x, y] = self.action_list[best_action] if best_action is not None else " "

    @measure_performance
    def q_learning(self):
        """ Q-Learning 알고리즘 """
        for episode_num in range(self.episodes):
            # epsilon을 에피소드 진행에 따라 감소 (최소 0.1)
            epsilon = max(0.1, 1.0 - episode_num / self.episodes)
            episode = self.generate_episode(epsilon)
            for t in range(len(episode) - 1):  # 마지막 상태 제외
                state, action, reward, next_state = episode[t]
                current_q = self.q_values[(state[0], state[1], action)]
                # 다음 상태에서 최대 Q값 찾기 (Off-policy 특징)
                max_next_q = max(self.q_values[(next_state[0], next_state[1], act)] 
                               for act in self.action_list.keys())
                # Q-Learning 업데이트 공식
                self.q_values[(state[0], state[1], action)] += self.alpha * (
                    reward + self.gamma * max_next_q - current_q
                )

        self.update_policy()  # 최적 정책 계산
        return self.q_values, self.policy_map

# 실행문
if __name__ == "__main__":
    # Q-Learning 인스턴스 생성
    qlearn = QLearning(episodes=1000, gamma=0.9)
    
    # Q-Learning 알고리즘 실행
    q_values, policy_map = qlearn.q_learning()
    
    # 1. Q(s, a) 맵 출력
    print("=== Q(s, a) Map ===")
    for x in range(4):
        for y in range(4):
            print(f"State ({x}, {y}): ", end="")
            for act in qlearn.action_list.keys():
                q_val = q_values[(x, y, act)]
                print(f"{act}: {q_val:.2f}  ", end="")
            print()
        print()
    
    # 2. Policy Map 출력
    print("=== Policy Map ===")
    for x in range(4):
        row = " ".join(policy_map[x, y] for y in range(4))
        print(row)