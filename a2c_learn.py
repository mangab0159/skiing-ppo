# 필요한 패키지 임포트
import tensorflow as tf

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import adam_v2; Adam = adam_v2.Adam

import numpy as np
import matplotlib.pyplot as plt


## A2C 액터 신경망
class Actor(Model):

    # def __init__(self, action_dim, action_bound):
    def __init__(self, action_dim):
        super(Actor, self).__init__()

        self.h1 = Dense(64, activation='relu')
        self.h2 = Dense(32, activation='relu')
        self.h3 = Dense(16, activation='relu')
        self.mu = Dense(action_dim, activation='linear')

    def call(self, state):
        x = self.h1(state)
        x = self.h2(x)
        x = self.h3(x)
        mu = self.mu(x)

        return mu


## A2C 크리틱 신경망
class Critic(Model):

    def __init__(self):
        super(Critic, self).__init__()

        self.h1 = Dense(64, activation='relu')
        self.h2 = Dense(32, activation='relu')
        self.h3 = Dense(16, activation='relu')
        self.v = Dense(1, activation='linear')

    def call(self, state):
        x = self.h1(state)
        x = self.h2(x)
        x = self.h3(x)
        v = self.v(x)
        return v


## A2C 에이전트 클래스
class A2Cagent(object):

    def __init__(self, env):

        # 하이퍼파라미터
        self.GAMMA = 0.95
        self.BATCH_SIZE = 32
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001
        self.EPS = 1e-25

        # 환경
        self.env = env
        # 상태변수 차원
        self.state_dim = env.observation_space.shape[0]
        # 행동 차원
        self.action_dim = env.action_space.n

        # 액터 신경망 및 크리틱 신경망 생성
        self.actor = Actor(self.action_dim)
        self.critic = Critic()
        self.actor.build(input_shape=(None, self.state_dim))
        self.critic.build(input_shape=(None, self.state_dim))

        self.actor.summary()
        self.critic.summary()

        # 옵티마이저 설정
        self.actor_opt = Adam(self.ACTOR_LEARNING_RATE)
        self.critic_opt = Adam(self.CRITIC_LEARNING_RATE)

        # 에프소드에서 얻은 총 보상값을 저장하기 위한 변수
        self.save_epi_reward = []


    ## 액터 신경망에서 행동 샘플링
    def get_action(self, state):
        mu_a = self.actor(state)
        
        probs = tf.nn.softmax(mu_a)
        action = np.argmax(probs, 1)[0]
        return action


    ## 액터 신경망 학습
    def actor_learn(self, states, actions, advantages):

        with tf.GradientTape() as tape:
            # 로그 정책 함수
            action_approxes = self.actor(states, training=True)
            probs = tf.nn.softmax(action_approxes)
            prob = probs

            actions = tf.stack([tf.range(tf.shape(actions)[0]),actions[:,0]],axis=-1)
            prob = tf.gather_nd(probs, actions)
            log_policy = tf.math.log(prob)

            # 손실함수
            loss_policy = log_policy * advantages
            loss = tf.reduce_sum(-loss_policy)

        # 그래디언트
        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))


    ## 크리틱 신경망 학습
    def critic_learn(self, states, td_targets):
        with tf.GradientTape() as tape:
            td_hat = self.critic(states, training=True)
            loss = tf.reduce_mean(tf.square(td_targets-td_hat))

        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_variables))


    ## 시간차 타깃 계산
    def td_target(self, rewards, next_v_values, dones):
        y_i = np.zeros(next_v_values.shape)
        for i in range(next_v_values.shape[0]):
            if dones[i]:
                y_i[i] = rewards[i]
            else:
                y_i[i] = rewards[i] + self.GAMMA * next_v_values[i]
        return y_i


    ## 배치에 저장된 데이터 추출
    def unpack_batch(self, batch):
        unpack = batch[0]
        for idx in range(len(batch)-1):
            unpack = np.append(unpack, batch[idx+1], axis=0)

        return unpack


    ## 에이전트 학습
    def train(self, max_episode_num):

        # 에피소드마다 다음을 반복
        for ep in range(int(max_episode_num)):

            # 배치 초기화
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = [], [], [], [], []
            # 에피소드 초기화
            time, episode_reward, done = 0, 0, False
            # 환경 초기화 및 초기 상태 관측
            state = self.env.reset()

            while not done:

                # 행동 샘플링
                action = self.get_action(tf.convert_to_tensor([state], dtype=tf.float32))
                # 다음 상태, 보상 관측
                next_state, reward, done, _ = self.env.step(action)
                # shape 변환
                state = np.reshape(state, [1, self.state_dim])
                # action = np.reshape(action, [1, self.action_dim])
                action = np.reshape(action, [1, 1])
                reward = np.reshape(reward, [1, 1])
                next_state = np.reshape(next_state, [1, self.state_dim])
                done = np.reshape(done, [1, 1])
                train_reward = reward

                # 배치에 저장
                batch_state.append(state)
                batch_action.append(action)
                batch_reward.append(train_reward)
                batch_next_state.append(next_state)
                batch_done.append(done)

                # 배치가 채워질 때까지 학습하지 않고 저장만 계속
                if len(batch_state) < self.BATCH_SIZE:
                    # 상태 업데이트
                    state = next_state[0]
                    episode_reward += reward[0]
                    time += 1
                    continue

                # 배치가 채워지면 학습 진행
                # 배치에서 대이터 추출
                states = self.unpack_batch(batch_state)
                actions = self.unpack_batch(batch_action)
                train_rewards = self.unpack_batch(batch_reward)
                next_states = self.unpack_batch(batch_next_state)
                dones = self.unpack_batch(batch_done)

                # 배치 비움
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = [], [], [], [], []

                # TD(0) 타깃 계산
                next_v_values = self.critic(tf.convert_to_tensor(next_states, dtype=tf.float32))
                td_targets = self.td_target(train_rewards, next_v_values.numpy(), dones)

                # 크리틱 신경망 업데이트
                self.critic_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                                  tf.convert_to_tensor(td_targets, dtype=tf.float32))

                # 어드밴티지 계산
                v_values = self.critic(tf.convert_to_tensor(states, dtype=tf.float32))
                next_v_values = self.critic(tf.convert_to_tensor(next_states, dtype=tf.float32))
                advantages = train_rewards + self.GAMMA * next_v_values - v_values

                # 액터 신경망 업데이트
                self.actor_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                                 tf.convert_to_tensor(actions, dtype=tf.int32),
                                 tf.convert_to_tensor(advantages, dtype=tf.float32))

                # 상태 업데이트
                state = next_state[0]
                episode_reward += reward[0]
                time += 1


            # 에피소드마다 결과 출력
            print('Episode: ', ep+1, 'Time: ', time, 'Reward: ', episode_reward)

            self.save_epi_reward.append(episode_reward)


        # 학습이 끝난 후, 누적 보상값 저장
        np.savetxt('./save_weights/Skiing_epi_reward.txt', self.save_epi_reward)
        print(self.save_epi_reward)


    ## 에피소드와 누적 보상값을 그려주는 함수
    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()

