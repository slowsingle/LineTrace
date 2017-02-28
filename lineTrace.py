# -*- coding: utf-8 -*-

import gym
import gym.spaces
import cv2
import numpy as np


class Agent(object):
    def __init__(self, field, n_state, n_action, init_x=320.0, init_y=80.0):
        self.x = init_x
        self.y = init_y
        self.vx = 1.0
        self.vy = 0.0
        self.body_r = 20
        self.body_color = (100, 255, 100)
        self.field = cv2.cvtColor(field, cv2.COLOR_BGR2GRAY)
        self.n_state = n_state
        self.n_action = n_action

    # エージェントは各行動に応じて移動する
    def update(self, act_num):
        # (vx, vy)
        # act number に応じて方向転換する
        act_list = (np.pi / 3.0, np.pi/ 4.0, np.pi/ 6.0, 0.0, -np.pi / 6.0, -np.pi/ 4.0, -np.pi / 3.0)
        tmp_vx = np.cos(act_list[act_num]) * self.vx - np.sin(act_list[act_num]) * self.vy
        tmp_vy = np.sin(act_list[act_num]) * self.vx + np.cos(act_list[act_num]) * self.vy
        self.vx = tmp_vx
        self.vy = tmp_vy

        self.x += 2.0 * self.vx
        self.y += 2.0 * self.vy

    # 現在の状態をバーチャルセンサから取得する
    def get_state(self):
        # センサ中心
        sensor_center_x = self.x + 2.0 * self.vx
        sensor_center_y = self.y + 2.0 * self.vy
        # ラインセンサの並ぶ方向のベクトル
        sensor_direct_x = - self.vy
        sensor_direct_y = self.vx
        state = list()
        for i in range(-(self.n_state // 2), -(self.n_state // 2) + self.n_state):
            s_x = int(round(sensor_center_x + 3.0 * i * sensor_direct_x))
            s_y = int(round(sensor_center_y + 3.0 * i * sensor_direct_y))
            try:
                # print s_x, s_y
                # 黒線(0)と白い領域(1)を検出
                sensor_val = self.field[s_y, s_x]
                if sensor_val == 0:
                    state.append(0.0)
                elif sensor_val == 255:
                    state.append(1.0)
                else:
                    print "error (sensor_val is not 0 nor 255)"
                    return None
            except Exception as e:
                print "except"
                return None
        return np.array(state)


class LineTrace(gym.core.Env):
    def __init__(self, obs_space=9, act_space=7):
        self.obs_space = obs_space
        self.act_space = act_space
        self.observation_space = gym.spaces.Box(low=-0.0, high=1.0, shape=(obs_space,))
        self.action_space = gym.spaces.Discrete(act_space)
        self.reward_range = (-1.0, 1.0)

        # フィールド画像の読み込み
        # フィールド画像には白の中に黒線があるものを想定する
        self.field = cv2.imread("field/field.png")
        self.agent = Agent(field=self.field, n_state=self.obs_space, n_action=self.act_space)
        self.n_step = 0
        self.max_step = 10000

    # 各stepごとに呼ばれる
    # actionを受け取り、次のstateとreward、episodeが終了したかどうかを返すように実装
    def _step(self, action):
        self.agent.update(act_num=action)
        state = self.agent.get_state()

        done = False
        counts = np.where(state == 0.0)[0]
        len_counts = counts.shape[0]

        if len_counts == 0:
            reward = -1.0
        else:
            point = np.mean(counts)
            center = float(self.obs_space) / 2.0
            reward = (center - abs(point - center)) / center
            #print point, center, reward

        # 場外へ出そうになったらエピソード終了
        if self.agent.x < 20.0 or self.agent.x > 620 or self.agent.y < 20.0 or self.agent.y > 460:
            done = True
            reward = -1.0

        # 一定以上走れたらエピソード終了
        if self.n_step > self.max_step:
            done = True
            reward = 1.0
        else:
            self.n_step += 1

        return state, reward, done, {}

    # cv2の関数を利用してエージェントの動きを描画する
    def viewUseCv2(self, isWaitKey=False):
        tmp_field = np.copy(self.field)
        cv2.circle(tmp_field, center=(int(self.agent.x), int(self.agent.y)), \
                   radius=self.agent.body_r, color=self.agent.body_color, thickness=-1)
        cv2.line(tmp_field, pt1=(int(self.agent.x), int(self.agent.y)), \
                 pt2=(int(self.agent.x + 20 * self.agent.vx), int(self.agent.y + 20 * self.agent.vy)), \
                 color=(255, 100, 100), thickness=3)
        cv2.imshow('render', tmp_field)
        if isWaitKey:
            cv2.waitKey(0)
        else:
            cv2.waitKey(10)

    def _reset(self):
        self.agent.x = 320.0
        self.agent.y = 80.0
        self.agent.vx = 1.0
        self.agent.vy = 0.0
        self.n_step = 0
        cv2.destroyAllWindows()

        return self.agent.get_state()
