from gymnasium.envs.registration import register

register(
     id="SlipperyDistShift-v0",
     entry_point="src.envs:SlipperyDistShift",
     max_episode_steps=300,
)
register(
        id="DynamicObstaclesSwitch-8x8-v0",
        entry_point="src.envs:DynamicObstaclesSwitchEnv",
        max_episode_steps=200,
        kwargs = {"penalty":0.05, "size":8}
)
register(
        id="DynamicObstaclesSwitch-6x6-v0",
        entry_point="src.envs:DynamicObstaclesSwitchEnv",
        max_episode_steps=200,
        kwargs = {"penalty":0.1, "size":6}
)

