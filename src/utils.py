import numpy as np
from numba import jit
    
def calculate_reward(old_info, new_info, distance_map) -> np.ndarray:
    # eaten словарь из пойманных существ.
    # Ключи - номер команды и индекс пойманного существа,
    # значение - номер команды и индекс существа, которое его поймало
    
    old_hunter_coords = np.array([(agent["y"], agent["x"]) for agent in old_info["predators"]])
    new_hunter_coords = np.array([(agent["y"], agent["x"]) for agent in new_info["predators"]])

    enemy_coords = np.array([(enemy["y"], enemy["x"]) for enemy in old_info["enemy"] if enemy["alive"]])
    pray_coords = np.array([(prey["y"], prey["x"]) for prey in old_info["preys"] if prey["alive"]])

    eatable_targets = np.concatenate((pray_coords, enemy_coords))

    old_distances = get_targets_distance(old_hunter_coords, eatable_targets, distance_map)
    new_distances = get_targets_distance(new_hunter_coords, eatable_targets, distance_map)

    values = np.ones(len(eatable_targets))
    values[-len(enemy_coords):] = values[-len(enemy_coords):] * 3 # As hunter kill equals 3 points +-

    density = get_targets_density(eatable_targets, distance_map, values, radius=5)
    
    enemy_bonus_counts = np.array([enemy['bonus_count'] for enemy in old_info['enemy'] if enemy["alive"]])
    agents_bonus_counts = get_bonus_counts(old_info)

    # Because we can't kill instantly and need 1 + bonus turns to achieve
    density[-len(enemy_coords):] = density[-len(enemy_coords):] * (1 / (enemy_bonus_counts + 1)) 

    metric = density[None, :] / (old_distances + 1) # [5, n_pray] and fixes div 0 error

    # We should not eat enemy without a bonus
    metric[:, -len(enemy_coords):] = metric[:, -len(enemy_coords):] * (agents_bonus_counts > 0)[:, None]

    best_pray_index = np.argmax(metric, axis=1)

    old_distances = np.take_along_axis(old_distances, best_pray_index[:, None], axis=1).squeeze()
    new_distances = np.take_along_axis(new_distances, best_pray_index[:, None], axis=1).squeeze()

    isnan = np.logical_or(np.isnan(old_distances), np.isnan(new_distances))
    old_distances[isnan] = 0
    new_distances[isnan] = 0

    dist_difference = new_distances - old_distances

    prey_kills, enemy_kills, bonus_kills = get_kills(old_info, new_info)
    
    killed_anybody = np.logical_or(prey_kills, enemy_kills, bonus_kills)

    dist_difference[killed_anybody == 1] = 0

    sudden_change = np.logical_or(dist_difference > 2,
                                      dist_difference < -2)
    dist_difference[sudden_change] = 0

    dist_difference = np.clip(dist_difference, -1, 1)

    result = dist_difference * -0.5 + prey_kills + bonus_kills * 1.3 + enemy_kills * 3 * (agents_bonus_counts > 0)

    stands_still = check_for_standing_still(old_info, new_info)
    result[stands_still == 1] = -0.7
    
    return result

def check_for_standing_still(info, next_info):
    out = []
    for predator_info, next_predator_info in zip(info['predators'], next_info['predators']):
        out.append(
            predator_info['x'] == next_predator_info['x'] and
            predator_info['y'] == next_predator_info['y']
        )
    return np.array(out)

def get_bonus_counts(info):
    return np.array([p['bonus_count'] for p in info['predators']])

def get_kills(info, next_info):
    """Returns prey kills and enemy kills for each predator of team 0 during the step"""
    n = len(next_info['predators'])
    prey_team_id = next_info['preys'][0]['team']
    prey_kills = np.zeros(n)
    enemy_kills = np.zeros(n)

    for killed, killer in next_info['eaten'].items():
        if killer[0] != 0:
            continue

        if killed[0] == prey_team_id:
            prey_kills[killer[1]] = 1
        else:
            enemy_kills[killer[1]] = 1

    bonus_counts = get_bonus_counts(info)
    bonus_counts_next = get_bonus_counts(next_info)
    bonus_kills = (bonus_counts_next > bonus_counts).astype(int)

    return prey_kills, enemy_kills, bonus_kills

@jit(nopython=True, parallel=True)
def get_targets_distance(hunters_coordinates: np.ndarray, 
                         preys_coords: np.ndarray, 
                         distance_map: np.ndarray):
    targets_distance = np.empty(shape=(5, len(preys_coords)), dtype=np.int64)
    for i, hunter_coordinates in enumerate(hunters_coordinates):
        distances = distance_map[hunter_coordinates[0]*40 + hunter_coordinates[1], :]
        flat_indices = np.array([prey_coords[0]*40 + prey_coords[1] for prey_coords in preys_coords])
        distances = distances[flat_indices]
        targets_distance[i] = distances

    return targets_distance

@jit(nopython=True, parallel=True)
def get_targets_density(preys_coords: np.ndarray, 
                        distance_map: np.ndarray, 
                        values: np.ndarray,
                        radius):
    density = []
    flat_indices = np.array([prey_coords[0]*40 + prey_coords[1] for prey_coords in preys_coords])
    for prey_coord in preys_coords:
        distances = distance_map[prey_coord[0]*40 + prey_coord[1]]
        distances = distances[flat_indices]
        density.append(np.sum(values[distances < radius]))
    return np.array(density)