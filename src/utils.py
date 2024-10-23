"""
Functions that use multiple times
"""

from collections import defaultdict

import torch
import numpy as np
from world.utils import RenderedEnvWrapper

my_hash = defaultdict(int)


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

def calculate_reward(state, old_info, new_info) -> np.ndarray:
    # eaten словарь из пойманных существ.
    # Ключи - номер команды и индекс пойманного существа,
    # значение - номер команды и индекс существа, которое его поймало
    team_indices = [predator["id"] for predator in new_info["predators"]]
    team_coordinates = np.array([(predator["y"], predator["x"]) for predator in new_info["predators"]])
    old_team_coordinates = np.array([(predator["y"], predator["x"]) for predator in old_info["predators"]])

    new_preys_coordinates = np.array([(prey["y"], prey["x"]) for prey_index, prey in enumerate(new_info["preys"]) if old_info["preys"][prey_index]["alive"]])
    old_preys_coordinates = np.array([(prey["y"], prey["x"]) for prey in old_info["preys"] if prey["alive"]])

    # eaten = np.array(list(new_info["eaten"].values()))
    rewards = []

    for hunter_index, hunter_position, old_hunter_position in zip(team_indices, team_coordinates, old_team_coordinates):
        # distance_to_closest_prey = np.linalg.norm(hunter_position[None, :] - preys_coordinates, axis=1).min()
        # todo: add eaten hunter handling
        # if len(eaten):
        #     eaten_by_current_hunter = eaten[(eaten == (0, hunter_index)).all(axis=1)].shape[0]
        # else:
        #     eaten_by_current_hunter = 0
        # my_hash[tuple(hunter_position)] += 1
        # rewards.append(
        #     10 * eaten_by_current_hunter
        #     + 5 / (distance_to_closest_prey + 1e-9)
        #     + 3 * (my_hash[tuple(hunter_position)]) ** (-1 / 2)
        # )
        current_potential = get_potential(state, hunter_position, new_preys_coordinates)
        previous_potential = get_potential(state, old_hunter_position, old_preys_coordinates)

        rewards.append(0.99 * current_potential - previous_potential)
    return np.array(rewards)

def get_potential(state, hunter_position: np.ndarray, preys_coordinates: np.ndarray):
    closest_pray_coords = preys_coordinates[np.argsort(np.sum(np.abs(preys_coordinates - hunter_position), axis=-1))]

    distance = find_path(state, hunter_position, closest_pray_coords, 5)
    return 1 - distance /(40 + 40 - 2)

def record(global_ep, global_ep_r, episode_score, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        global_ep_r.value = episode_score
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:",
        global_ep.value,
        "| Ep_r: %.3f" % global_ep_r.value,
        # "| Ep_score: %.0f" % episode_score,
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

def preprocess_data(state: np.ndarray, info: dict) -> torch.Tensor:
    state = torch.tensor(state)

    hunters_coordinates = torch.tensor(np.array([(agent["y"], agent["x"]) for agent in info["predators"]]))
    prey_id = state[:,:, 0].max()
    prey_mask = (state[:,:, 0] == prey_id)
    hunter_mask = (state[:,:, 0] == 0)
    state[prey_mask] = prey_id * torch.ones_like(state[prey_mask])

    states = []
    for hunter_coordinates in hunters_coordinates:
        new_state = state.clone()
        new_state[hunter_mask] = 5 * torch.ones_like(new_state[hunter_mask])
        new_state[[*hunter_coordinates]] = torch.ones_like(new_state[[*hunter_coordinates]]) * 15
        states.append(new_state.permute(2,0,1).float().unsqueeze(0))
    
    return torch.cat(states)

from collections import deque

# To store matrix cell coordinates
class Point:
    def __init__(self,x: int, y: int):
        self.x = x
        self.y = y
 
# A data structure for queue used in BFS
class queueNode:
    def __init__(self,pt: Point, dist: int):
        self.pt = pt  # The coordinates of the cell
        self.dist = dist  # Cell's distance from the source
 
# Check whether given cell(row,col)
# is a valid cell or not
def isValid(row: int, col: int, mask):
    if row < 0:
        row = 39
    if col < 0:
        col = 39
    if row >= 40:
        row = 0
    if col >= 40:
        col = 0
    
    return mask[row, col], row, col
 
# These arrays are used to get row and column 
# numbers of 4 neighbours of a given cell 
rowNum = [-1, 0, 0, 1]
colNum = [0, -1, 1, 0]
 
# Function to find the shortest path between 
# a given source cell to a destination cell. 
def BFS(mat, src: Point, dest: Point):
     
    # check source and destination cell 
    # of the matrix have value 1 
    if mat[src.x][src.y]!=1 or mat[dest.x][dest.y]!=1:
        return -1
     
    visited = [[False for i in range(40)] 
                       for j in range(40)]
     
    # Mark the source cell as visited 
    visited[src.x][src.y] = True
     
    # Create a queue for BFS 
    q = deque()
     
    # Distance of source cell is 0
    s = queueNode(src,0)
    q.append(s) #  Enqueue source cell
     
    # Do a BFS starting from source cell 
    while q:
 
        curr = q.popleft() # Dequeue the front cell
         
        # If we have reached the destination cell, 
        # we are done 
        pt = curr.pt
        if pt.x == dest.x and pt.y == dest.y:
            return curr.dist
         
        # Otherwise enqueue its adjacent cells 
        for i in range(4):
            row = pt.x + rowNum[i]
            col = pt.y + colNum[i]
             
            # if adjacent cell is valid, has path  
            # and not visited yet, enqueue it.
            valid, row, col = isValid(row,col, mat)
            if (valid and not visited[row][col]):
                visited[row][col] = True
                Adjcell = queueNode(Point(row,col),
                                    curr.dist+1)
                q.append(Adjcell)
     
    # Return -1 if destination cannot be reached 
    return -1

def find_path(state, hunter_coods, preys_coords, top_nearest = 5):
    preys_coords = preys_coords[:top_nearest]
    state = state.squeeze()

    mask = (~((state[:, :, 0] == -1) * (state[:, :, 1] == -1) 
              + (state[:, :, 0] == 0))).astype(np.int32)
    mask[hunter_coods[0], hunter_coods[1]] = 1
    source = Point(*hunter_coods)
    distance = np.inf
    for nearest_prey_coords in preys_coords:
        dest = Point(*nearest_prey_coords)
        distance = min(distance, BFS(mask,source,dest))
        if distance == 0:
            break
        if distance == -1:
            distance = 40 + 40 - 2
            break
    return distance