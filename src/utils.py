import numpy as np
from src.preprocess import get_adjacent_cells
from numba import jit
    
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

@jit(nopython=True, parallel=True)
def __get_n_nearest_targets(obstacles_mask, centered_preys_mask, src = (20, 20), n=1):
    """Returns list of tuples (x, y, dst) of n nearest targets 
    or more if n nearest targets are bonuses or enemies"""
    
    out = []

    queue = []
    queue.append(src) 

    distance_mask = np.empty_like(obstacles_mask, dtype=np.float32)
    distance_mask.fill(np.nan)
    distance_mask[src[1], src[0]] = 0

    contains_preys = False

    while queue:
        x, y = queue.pop(0)

        for nx, ny in get_adjacent_cells(x, y, obstacles_mask, distance_mask):
            queue.append((nx, ny))
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