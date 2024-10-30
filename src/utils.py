import torch
import numpy as np
from queue import Queue
from world.utils import RenderedEnvWrapper
from src.preprocess import preprocess_data, get_adjacent_cells



class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

def record(global_ep, global_ep_r, episode_score, my_reward, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        global_ep_r.value = episode_score
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:",
        global_ep.value,
        "| Ep_r: %.2f" % global_ep_r.value,
        "| Ep_score: %.2f" % my_reward.mean(),
    )

def evaluate_policy(agent, env, device, episodes=5):
    env = RenderedEnvWrapper(env)
    returns = []
    for i in range(episodes):
        done = False
        state, info = env.reset()

        while not done:
            state = preprocess_data(state, info)
            state, done, new_info = env.step(agent.act(state.to(device)))

            # reward = calculate_reward(info, new_info)

            # total_reward += reward

            info = new_info.copy()
        env.render(f"./Episode_{i+1}")
    return info["scores"][0]

def action_coord_change(position, action) -> tuple[int, int]:
        y, x = position[0], position[1]
        tx, ty = x, y
        if action == 1:
            tx = (x + 1) % 40
        elif action == 2:
            tx = (40 + x - 1) % 40
        elif action == 3:
            ty = (40 + y - 1) % 40
        elif action == 4:
            ty = (y + 1) % 40
        return ty, tx
    
def calculate_reward(processed_state, next_processed_state, old_info, new_info, actions) -> np.ndarray:
    # eaten словарь из пойманных существ.
    # Ключи - номер команды и индекс пойманного существа,
    # значение - номер команды и индекс существа, которое его поймало
    
    old_distances = np.array([__get_n_nearest_targets(obst_mask, centered_preys_msk) for centered_preys_msk, _, obst_mask, _ in processed_state])
    new_distances = np.array([__get_n_nearest_targets(obst_mask, centered_preys_msk) for centered_preys_msk, _, obst_mask, _ in next_processed_state])

    isnan = np.logical_or(np.isnan(old_distances), np.isnan(new_distances))
    old_distances[isnan] = 0
    new_distances[isnan] = 0

    dist_difference = new_distances - old_distances

    kills = get_kills(old_info, new_info)
    
    dist_difference[kills == 1] = 0

    sudden_change = np.logical_or(dist_difference > 2,
                                      dist_difference < -2)
    dist_difference[sudden_change] = 0

    dist_difference = np.clip(dist_difference, -1, 1)

    result = dist_difference * -0.5 + kills

    stands_still = check_for_standing_still(old_info, new_info)
    result[stands_still == 1] = -0.7
    
    return result

# def get_potential(centered_obstacles_mask: np.ndarray, centered_preys_mask: np.ndarray):

#     # targets = __get_n_nearest_targets(centered_obstacles_mask, centered_preys_mask)
#     distance = __get_n_nearest_targets(centered_obstacles_mask, centered_preys_mask)
#     if not distance:
#         return None
#     return 1 - distance[0][-1] /(40 + 40 - 2)

def check_for_standing_still(info, next_info):
    out = []
    for predator_info, next_predator_info in zip(info['predators'], next_info['predators']):
        out.append(
            predator_info['x'] == next_predator_info['x'] and
            predator_info['y'] == next_predator_info['y']
        )
    return np.array(out)

def get_kills(info, next_info):
    """Returns prey kills and enemy kills for each predator of team 0 during the step"""
    n = len(next_info['predators'])
    prey_team_id = next_info['preys'][0]['team']
    prey_kills = np.zeros(n)
    # enemy_kills = np.zeros(n)

    for killed, killer in next_info['eaten'].items():
        if killer[0] != 0:
            continue

        if killed[0] == prey_team_id:
            prey_kills[killer[1]] = 1

    return prey_kills

def __get_n_nearest_targets(obstacles_mask, centered_preys_mask, src = (20, 20), n=1):
    """Returns list of tuples (x, y, dst) of n nearest targets 
    or more if n nearest targets are bonuses or enemies"""

    out = []

    queue = Queue()
    queue.put(src)

    distance_mask = np.empty_like(obstacles_mask, dtype=np.float32)
    distance_mask.fill(np.nan)
    distance_mask[src[1], src[0]] = 0

    contains_preys = False

    while not queue.empty():
        x, y = queue.get()

        for nx, ny in get_adjacent_cells(x, y, obstacles_mask, distance_mask):
            queue.put((nx, ny))
            distance_mask[ny, nx] = distance_mask[y, x] + 1

            if centered_preys_mask[ny, nx] == 1:
                contains_preys = True
                out.append((nx, ny, distance_mask[ny, nx]))

            if len(out) >= n and contains_preys:
                break
    if not out:
        return np.nan
    else:
        return out[0][2]