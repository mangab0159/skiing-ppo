# skiing-ppo

- [openai gym skiing](https://gym.openai.com/envs/Skiing-v0/) 게임 agent 구현

  <img width="150" src="https://user-images.githubusercontent.com/39723283/158948457-c111f740-b570-4e50-8bbe-31b6e800462f.png" />

- 환경 설정
  - 관련 패키지 설치
  ```shell
  pip install tensorflow
  pip install gym
  pip install atari_py
  ```
  - ROM 파일 import
    - [다운로드 링크](http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html)
  ```shell
  python -m atari_py.import_roms <path to folder>
  ```
