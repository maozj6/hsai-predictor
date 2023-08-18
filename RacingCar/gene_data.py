from CarRacingDQNAgent import CarRacingDQNAgent
from common_functions import process_state_image
from common_functions import generate_state_frame_stack_from_queue
import math
import argparse
import os
import gym
from collections import deque
import cv2
import numpy as np
import time

np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)
def process_frame(frame):
  obs = frame[0:84, :, :].astype(np.float)
  obs = cv2.resize(obs,(64,64))
  obs = ((obs) ).round().astype(np.uint8)
  state = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
  state = state.astype(float)
  state /= 255.0
  return state
class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getx(self):
        return self.x

    def gety(self):
        return self.y


def GetCross(p1, p2, p):
    return (p2.x - p1.x) * (p.y - p1.y) - (p.x - p1.x) * (p2.y - p1.y)


class Getlen:
    def __init__(self, p1, p2):
        self.x = p1.getx() - p2.getx()
        self.y = p1.gety() - p2.gety()
        # use math.sqrt() to get the square root
        self.len = math.sqrt((self.x ** 2) + (self.y ** 2))
    # define the function of getting the length of line
    def getlen(self):
        return self.len

def IsPointInMatrix(p1, p2, p3, p4, p):
    isPointIn = GetCross(p1, p2, p) * GetCross(p3, p4, p) >= 0 and GetCross(p2, p3, p) * GetCross(p4, p1, p) >= 0
    return isPointIn

def getDis(p1, p2, p3, p4, p):
    # define the object
    l1 = Getlen(p1, p2)
    l2 = Getlen(p1, p3)
    l3 = Getlen(p2, p3)
    # get the length of two points
    d1 = l1.getlen()
    d2 = l2.getlen()
    d3 = l3.getlen()

def isInTrack(position, trackList):
    x, y = position
    pp = Point(x, y)
    for i in range(len(trackList)):
        p1 = Point(trackList[i][0][0][0], trackList[i][0][0][1])
        p2 = Point(trackList[i][0][1][0], trackList[i][0][1][1])
        p3 = Point(trackList[i][0][2][0], trackList[i][0][2][1])
        p4 = Point(trackList[i][0][3][0], trackList[i][0][3][1])
        if IsPointInMatrix(p1, p2, p3, p4, pp):
            return True
    return False

if __name__ == '__main__':


    name = 0
    parser = argparse.ArgumentParser(description='Collecting data of RacingCar')
    parser.add_argument('-n', '--number', default=80000, help='number of samples')
    parser.add_argument('-d', '--dir', default='data/test', help='output path, dir\'s name')
    parser.add_argument('-c', '--controller',default='models/trial_600.h5', help='path of DQN models')
    parser.add_argument('-s', '--seed',default=0, help='random seed')

    args = parser.parse_args()

    train_model=[args.controller]
    print(train_model)
    outdir = args.dir
    rseed = int(args.seed)

    if not os.path.exists(outdir + '/'):
        os.makedirs(outdir + '/')
    env = gym.make('CarRacing-v0')
    env.seed(rseed)
    np.random.seed(rseed)
    agent = CarRacingDQNAgent(epsilon=0)  # Set epsilon to 0 to ensure all actions are instructed by the agent
    serise=[]
    for i in range(len(train_model)):
        each_model_number = int(args.number)
        agent.load(train_model[i])
        guard=0
        safety=[]
        recording_imgs = []
        recording_obs = []
        recording_action = []
        recording_safe = []
        recording_position = []
        recording_map = []
        collect=0
        npz_guard=0
        while collect < each_model_number :
            npz_guard=npz_guard+1
            safe_frame=False
            single_count = 0
            end_count=0
            init_state = env.reset()
            init_state = process_state_image(init_state)
            total_reward = 0
            punishment_counter = 0
            state_frame_stack_queue = deque([init_state] * agent.frame_stack_num, maxlen=agent.frame_stack_num)
            time_frame_counter = 1
            for j in range(50):
                env.render()
                current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
                action = agent.act(current_state_frame_stack)
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                next_state = process_state_image(next_state)
                state_frame_stack_queue.append(next_state)
            while True:
                env.render()
                current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
                action = agent.act(current_state_frame_stack)
                recording_action.append(action)
                next_state, reward, done, info = env.step(action)
                ctrl_obs=process_state_image(next_state)
                recording_obs.append(ctrl_obs)
                obs64=process_frame(next_state)
                recording_imgs.append(obs64)
                single_count=single_count+1

                name = name + 1
                guard=guard+1
                posx, posy = info[0]
                safety = isInTrack(info[0], info[1])
                if safety:
                    recording_safe.append(1)
                else:
                    recording_safe.append(0)
                recording_position.append(info[0])
                total_reward += reward
                next_state = process_state_image(next_state)
                state_frame_stack_queue.append(next_state)
                if done:
                    serise.append(len(recording_obs))
                    recording_map = (info[1])
                    recording_obs = np.array(recording_obs, dtype=np.float16)
                    recording_action = np.array(recording_action, dtype=np.float16)
                    recording_position=np.array(recording_position, dtype=np.float16)
                    recording_map=np.array(recording_map)
                    tmp=time.strftime("%Y%m%d%H%M%S", time.localtime())
                    recording_label=[]
                    for big_i in range(len(recording_safe)-200):
                        labels=[]
                        initial_safe=True
                        small_i=0
                        while len(labels)<21:
                            print(big_i)
                            print("b and s")
                            print(small_i)
                            if recording_safe[big_i+small_i]==False:
                                for little_j in range(21-len(labels)):
                                    labels.append(0)
                                break
                            if small_i%10==0:
                                labels.append(recording_safe[big_i+small_i])
                            small_i=small_i+1
                        recording_label.append(labels)
                    collect = collect+len(recording_label)
                    np.savez_compressed(outdir+"/"+str(npz_guard)+".npz", obs=recording_obs, imgs=recording_imgs,action=recording_action,safe=recording_safe,label=recording_label,serise=serise,map=recording_map,position=recording_position,model= train_model[0])
                    recording_imgs = []
                    recording_obs = []
                    recording_action = []
                    recording_safe = []
                    recording_position = []
                    recording_map = []
                    break
                time_frame_counter += 1
