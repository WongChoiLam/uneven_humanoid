import torch
import wandb
import functools
import statistics

def wandb_env_wrapper(env):
    step = env.step
    @functools.wraps(step)
    def wrapper(self, *args, **kwargs):
        print("Before calling method")
        result = step (self, *args, **kwargs)
        print("After calling method")
        return result
    env.step = wrapper
    return env

def runner_wrapper(runner, name : str, group : str, mode : str, dir : str):
    _log = runner.log
    wandb.init(name=name, group=group, mode=mode, dir=dir)
    @functools.wraps(_log)
    def wrapper(*args, **kwargs):
        result = _log (*args, **kwargs)
        locs = args[0]
        wandb_log = {}
        if len(locs['rewbuffer']) > 0:
            wandb_log = {
                'train/mean_reward'         : statistics.mean(locs['rewbuffer']),\
                'train/mean_episode_length' : statistics.mean(locs['rewbuffer']),\
            }
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=runner.device)
                for ep_info in locs['ep_infos']:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(runner.device)))
                value = torch.mean(infotensor).item()
                log_key = 'ep_infos/mean_' + key
                wandb_log[log_key] = value
        wandb.log(wandb_log, locs['it'])
        return result
    runner.log = wrapper
    return runner
