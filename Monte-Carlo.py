import numpy as np
import random
from decorator import measure_performance
from grid_world import GridWorld
class MonteCarlo(GridWorld):
    def __init__(self, episodes, gamma=0.9):
        super().__init__()
        self.episodes = episodes  # 에피소드 수
        self.gamma = gamma
        self.returns = {(x, y): [] for x in range(4) for y in range(4)}  # 리턴 저장
        self.alpha = 0.1  # TD 학습률

    def generate_episode(self):
        state = [0, 0]
        episode = []
        epsilon = 0.1
    
        while self.reward_map[state[0], state[1]] != 4:
            if random.random() < epsilon:
                action = random.choice(list(self.action_list.keys()))
            else:
                best_value = -np.inf
                best_action = 0
                for act in self.action_list.keys():
                    next_state, penalty = self.update_state(state, self.action_map[act])
                    value = self.reward_map[next_state[0], next_state[1]] + penalty + self.gamma * self.value_map[next_state[0], next_state[1]]
                    if value > best_value:
                        best_value = value
                        best_action = act
                action = best_action
            next_state, penalty = self.update_state(state, self.action_map[action])
            reward = self.reward_map[next_state[0], next_state[1]] + penalty
            episode.append((state, reward, action))
            state = next_state

        episode.append((state, self.reward_map[state[0], state[1]], None))
        return episode

    def update_policy(self):
        """ V(s)를 기반으로 최적 정책 계산 및 policy_map에 저장 """
        for x in range(4):
            for y in range(4):
                state = [x, y]
                if self.reward_map[x, y] == 4:  # 종료 상태
                    self.policy_map[x, y] = "★"
                    continue
                
                best_action = None
                best_value = -np.inf
                for act in self.action_list.keys():
                    next_state, penalty = self.update_state(state, self.action_map[act])
                    reward = self.reward_map[next_state[0], next_state[1]] + penalty
                    value = reward + self.gamma * self.value_map[next_state[0], next_state[1]]
                    if value > best_value:
                        best_value = value
                        best_action = act
                
                self.policy_map[x, y] = self.action_list[best_action] if best_action is not None else " "

    @measure_performance
    def first_visit_simulate(self):
        """ First-Visit Monte Carlo 방식 """
        for _ in range(self.episodes):
            episode = self.generate_episode()
            G = 0
            visited = set()

            for t in reversed(range(len(episode))):
                state, reward, action = episode[t]
                G = self.gamma * G + reward

                if tuple(state) not in visited:
                    visited.add(tuple(state))
                    self.returns[tuple(state)].append(G)
                    self.value_map[state[0], state[1]] = np.mean(self.returns[tuple(state)])

        self.update_policy()  # 최적 정책 계산
        return self.value_map, self.policy_map

    @measure_performance
    def multi_visit_simulate(self):
        """ Every-Visit Monte Carlo 방식 """
        for _ in range(self.episodes):
            episode = self.generate_episode()
            G = 0

            for t in reversed(range(len(episode))):
                state, reward, _ = episode[t]
                G = self.gamma * G + reward
                self.returns[tuple(state)].append(G)
                self.value_map[state[0], state[1]] = np.mean(self.returns[tuple(state)])

        self.update_policy()  # 최적 정책 계산
        return self.value_map, self.policy_map

    @measure_performance
    def temporal_difference(self):
        """ TD(0) 방식 """
        for _ in range(self.episodes):
            episode = self.generate_episode()
            for t in range(len(episode) - 1):  # 마지막 상태 제외
                state, reward, action = episode[t]
                next_state, _ = self.update_state(state, self.action_map[action])
                current_value = self.value_map[state[0], state[1]]
                next_value = self.value_map[next_state[0], next_state[1]]
                self.value_map[state[0], state[1]] += self.alpha * (
                    reward + self.gamma * next_value - current_value
                )

        self.update_policy()  # 최적 정책 계산
        return self.value_map, self.policy_map

if __name__ == "__main__":
    # Monte Carlo 테스트
    # First-Visit Monte Carlo
    monte = MonteCarlo(episodes=3000)
    v_map, p_map = monte.first_visit_simulate()
    monte.show_grid("First-Visit Monte Carlo", v_map, p_map)

    # Every-Visit Monte Carlo
    #monte = MonteCarlo(episodes=2000)  # 새 인스턴스
    #v_map, p_map = monte.multi_visit_simulate()
    #monte.show_grid("Every-Visit Monte Carlo", v_map, p_map)
    
    # Temporal Difference
    monte = MonteCarlo(episodes=1000)  # 새 인스턴스
    v_map, p_map = monte.temporal_difference()
    monte.show_grid("Temporal Difference", v_map, p_map)