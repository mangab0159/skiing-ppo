from a2c_learn import A2Cagent
import gym

def main():

    max_episode_num = 1000   # 최대 에피소드 설정
    env_name = 'Skiing-ram-v4'
    env = gym.make(env_name)  # 환경으로 OpenAI Gym의 Skiing-ram-v4 설정
    agent = A2Cagent(env)   # A2C 에이전트 객체

    # 학습 진행
    agent.train(max_episode_num)

    # 학습 결과 도시
    agent.plot_result()

if __name__=="__main__":
    main()
