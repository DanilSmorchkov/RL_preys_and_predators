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

class SharedRMSprop(torch.optim.RMSprop):
    """Implements RMSprop algorithm with shared states.
    """

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0):
        super(SharedRMSprop, self).__init__(params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=0, centered=False)

        # State initialisation (must be done before step, else will not be shared between threads)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = p.data.new().resize_(1).zero_()
                state['square_avg'] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['square_avg'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # g = αg + (1 - α)Δθ^2
                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)
                # θ ← θ - ηΔθ/√(g + ε)
                avg = square_avg.sqrt().add_(group['eps'])
                p.data.addcdiv_(-group['lr'], grad, avg)

        return loss

def calculate_reward(state, old_info, new_info) -> np.ndarray:
    # eaten словарь из пойманных существ.
    # Ключи - номер команды и индекс пойманного существа,
    # значение - номер команды и индекс существа, которое его поймало
    team_indices = [predator["id"] for predator in new_info["predators"]]
    team_coordinates = np.array([(predator["x"], predator["y"]) for predator in new_info["predators"]])
    old_team_coordinates = np.array([(predator["x"], predator["y"]) for predator in old_info["predators"]])

    new_preys_coordinates = np.array([(prey["x"], prey["y"]) for prey_index, prey in enumerate(new_info["preys"]) if old_info["preys"][prey_index]["alive"]])
    old_preys_coordinates = np.array([(prey["x"], prey["y"]) for prey in old_info["preys"] if prey["alive"]])

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
        current_potential = get_potential(state, hunter_position, old_preys_coordinates)
        previous_potential = get_potential(state, old_hunter_position, old_preys_coordinates)

        rewards.append(0.9 * current_potential - previous_potential)
    return np.array(rewards)

def get_potential(state, hunter_position: np.ndarray, preys_coordinates: np.ndarray):
    closest_pray_coords = preys_coordinates[np.argmin(np.sum(np.abs(preys_coordinates - hunter_position), axis=-1))]

    distance = find_path(state, hunter_position, closest_pray_coords)
    return 1 - distance /(40+40)

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
        "| Ep_r: %.0f" % global_ep_r.value,
        # "| Ep_score: %.0f" % episode_score,
    )


def evaluate_policy(agent, env, device, episodes=5):
    env = RenderedEnvWrapper(env)
    returns = []
    for i in range(episodes):
        done = False
        state, info = env.reset()

        while not done:
            state, additional_info = preprocess_data(state, info)
            state, done, new_info = env.step(agent.act(state.to(device), additional_info.to(device)))

            # reward = calculate_reward(info, new_info)

            # total_reward += reward

            info = new_info.copy()
        env.render(f"./Episode_{i+1}")
    return info["scores"][0]

def preprocess_data(state: np.ndarray, info: dict) -> tuple[torch.Tensor, torch.Tensor]:
    state = torch.tensor(state).unsqueeze(0).permute(0, 3, 1, 2).float()

    hunters_coordinates = torch.tensor(np.array([(agent["x"], agent["y"]) for agent in info["predators"]]))
    preys_coordinates = torch.tensor(np.array([(prey["x"], prey["y"]) for prey in info["preys"] if prey["alive"]]))
    result_directions = []
    # todo: ПОДНЯТЬСЯ
    for hunter_coordinates in hunters_coordinates:
        vectors = hunter_coordinates[None, :] - preys_coordinates
        closest_prey_index = torch.linalg.norm(vectors.float(), dim=1).argmin()
        result_directions.append(vectors[closest_prey_index].unsqueeze(0) / torch.linalg.norm(vectors[closest_prey_index].float()))
    return state, torch.cat(result_directions).unsqueeze(0)

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

def find_path(state, hunter_coods, nearest_prey_coords):
    state = state.squeeze()
    mask = ((state[:, :, 0] != -1) + (state[:, :, 1] != -1)).astype(np.int32)
    source = Point(*hunter_coods)
    dest = Point(*nearest_prey_coords)

    return BFS(mask,source,dest)