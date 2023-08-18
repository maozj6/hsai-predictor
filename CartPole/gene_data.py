import cv2
import argparse
import gym
import math
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from collections import deque
from tqdm import tqdm
import os

import tkinter
USE_CUDA = True # If we want to use GPU (powerful one needed!)
env = gym.make('CartPole-v0').unwrapped

GRAYSCALE = True # False is RGB
RESIZE_PIXELS = 60 # Downsample image to this number of pixels
FRAMES = 2 # state is the number of last frames: the more frames,
device = torch.device("cuda" if (torch.cuda.is_available() and USE_CUDA) else "cpu")

if GRAYSCALE == 0:
    resize = T.Compose([T.ToPILImage(),
                        T.Resize(RESIZE_PIXELS, interpolation=Image.CUBIC),
                        T.ToTensor()])

    nn_inputs = 3 * FRAMES  # number of channels for the nn
else:
    resize = T.Compose([T.ToPILImage(),
                        T.Resize(RESIZE_PIXELS, interpolation=Image.CUBIC),
                        T.Grayscale(),
                        T.ToTensor()])
    nn_inputs = FRAMES  # number of channels for the nn

# ---- CONVOLUTIONAL NEURAL NETWORK ----
HIDDEN_LAYER_1 = 16
HIDDEN_LAYER_2 = 32
HIDDEN_LAYER_3 = 32
KERNEL_SIZE = 5 # original = 5
STRIDE = 2 # original = 2
class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(nn_inputs, HIDDEN_LAYER_1, kernel_size=KERNEL_SIZE, stride=STRIDE)
        self.bn1 = nn.BatchNorm2d(HIDDEN_LAYER_1)
        self.conv2 = nn.Conv2d(HIDDEN_LAYER_1, HIDDEN_LAYER_2, kernel_size=KERNEL_SIZE, stride=STRIDE)
        self.bn2 = nn.BatchNorm2d(HIDDEN_LAYER_2)
        self.conv3 = nn.Conv2d(HIDDEN_LAYER_2, HIDDEN_LAYER_3, kernel_size=KERNEL_SIZE, stride=STRIDE)
        self.bn3 = nn.BatchNorm2d(HIDDEN_LAYER_3)
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        nn.Dropout()
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

n_actions = env.action_space.n

def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)
env.reset()
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape
stop_training = False

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Collecting data of RacingCar')
    parser.add_argument('-n', '--number', default=30000, help='number of samples')
    parser.add_argument('-d', '--dir', default='data/train/controller_1/', help='output path, dir\'s name')
    parser.add_argument('-c', '--controller',default='models/policy_net_best1.pt', help='path of DQN models')
    parser.add_argument('-s', '--seed',default=0, help='random seed')

    args = parser.parse_args()
    train_model=[args.controller]
    print(train_model)
    outdir = args.dir
    rseed = int(args.seed)

    env.seed(rseed)
    np.random.seed(rseed)
    total = int(args.number)
    theta_thre=6 * 2 * math.pi / 360
    model1 = torch.load(args.controller)
    model = DQN(screen_height, screen_width, n_actions).to(device)
    model.load_state_dict(model1)
    collect = 0
    npz_guard = 0
    outdir = args.dir
    if not os.path.exists(outdir + '/'):
        os.makedirs(outdir + '/')

    pbar = tqdm(total=total,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
    pbar.set_description("Collecting files " + str(npz_guard))

    while collect < total:
        env.reset()
        guard = 0
        init_screen = get_screen()
        screens = deque([init_screen] * FRAMES, FRAMES)
        END = False
        recording_safe = []
        recording_action = []
        recording_obs = []
        recording_label=[]
        while not END:
            state = torch.cat(list(screens), dim=1)
            action=model(state).max(1)[1].view(1, 1)
            state_variables, _, done, theta = env.step(action.item())
            END= done
            screens.append(get_screen())
            recording_action.append(action.cpu().detach().numpy().item())
            img = state[0][1].cpu().detach().numpy()
            obs = cv2.resize(img, (64, 64))
            recording_obs.append(obs)
            next_state = torch.cat(list(screens), dim=1)
            state = next_state
            if (theta < -theta_thre or theta > theta_thre):
                safe = 0
            else:
                safe= 1
            recording_safe.append(safe)
            guard = guard+1
        if guard>15:
            for big_i in range(len(recording_safe) - 10):
                labels = []
                initial_safe = True
                small_i = 0
                while len(labels) < 10:
                    if recording_safe[big_i + small_i] == 0:
                        for little_j in range(10 - len(labels)):
                            labels.append(0)
                        break
                    labels.append(recording_safe[big_i + small_i])
                    small_i = small_i + 1
                recording_label.append(labels)
            np.savez_compressed(outdir + "/" + str(npz_guard) + ".npz", obs=recording_obs,
                                action=recording_action, safe=recording_safe,label=recording_label,
                               )
            npz_guard =npz_guard+1
            collect=collect+guard
            print(collect)
            pbar.update(guard)
    pbar.close()


