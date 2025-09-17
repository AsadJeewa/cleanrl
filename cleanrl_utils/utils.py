def get_base_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env