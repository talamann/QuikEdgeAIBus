"""
Orchestration evaluation driver for the SimEdgeEnv.

Runs a trained IMPALA checkpoint for num_episodes episodes of episode_length steps and
emits (1) a per-step states.csv, (2) a per-episode summary.csv/json and (3) an
across-episode aggregate summary.json in data/testresults/IMPALA/containers/<N>/tests/<k>/.

Unlike test.py (which naively picks folders[0]), this script selects the trial folder
deterministically via --trial (exact folder name match; fallback: the trial whose name
contains the run start year passed via --year, i.e. the DLinear runs). Everything else
(env construction, checkpoints, flatten()) is identical to the baseline test.py.
"""
import os
import sys
import pickle
import json
import time
import click
from copy import deepcopy
import pandas as pd
from pprint import PrettyPrinter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datacenter.Datacenter import *  # noqa: F401,F403
from workload.Workload_v2 import *  # noqa: F401,F403

from utils.constants import (
    TRAIN_RESULTS_PATH,
    TESTS_RESULTS_PATH,
    DATASETS_PATH,
    SCHEDULER_PATH,
    ENVSMAP,
)
from utils.class_builder import make_env_class

pp = PrettyPrinter(indent=4)


def flatten(raw_obs, action, reward, info):
    return {
        'action': action,
        'num_consolidated': info['num_consolidated'],
        'num_moves': info['num_moves'],
        'num_overloaded': info['num_overloaded'],
        'mean_accuracy': info['mean_accuracy'],
        'num_model_switches': info['num_model_switches'],
        'num_slav': info['num_slav'],
        'cpu_conserved_cost': info['cpu_conserved_cost'],
        'mean_cluster_cpu_util': info['mean_cluster_cpu_util'],
        'mean_cluster_mem_util': info['mean_cluster_mem_util'],
        'oversub_cores': info['oversub_cores'],
        'reward_sla': info['rewards']['reward_sla'],
        'reward_accuracy': info['rewards']['reward_accuracy'],
        'reward_illegal': info['rewards']['reward_illegal'],
        'reward_consolidation': info['rewards']['reward_consolidation'],
        'reward': reward,
        'inference_time_ms': info.get('inference_time_ms', None),
    }


def pick_trial(experiment_folder, trial, year):
    items = os.listdir(experiment_folder)
    folders = [item for item in items if os.path.isdir(os.path.join(experiment_folder, item))]
    if trial is not None:
        matches = [f for f in folders if trial in f]
        if not matches:
            raise RuntimeError(f"--trial '{trial}' matched no folder in {experiment_folder}")
        return matches[0]
    for marker in (year, str(year)):
        for f in folders:
            if marker in f:
                return f
    return folders[0]


def summarize_episode(df):
    sums = {
        'num_consolidated': 'sum',
        'num_moves': 'sum',
        'num_overloaded': 'sum',
        'num_model_switches': 'sum',
        'num_slav': 'sum',
        'cpu_conserved_cost': 'sum',
        'oversub_cores': 'sum',
        'reward_sla': 'sum',
        'reward_accuracy': 'sum',
        'reward_illegal': 'sum',
        'reward_consolidation': 'sum',
        'reward': 'sum',
    }
    means = {
        'mean_accuracy': 'mean',
        'mean_cluster_cpu_util': 'mean',
        'mean_cluster_mem_util': 'mean',
    }
    row = {k: float(df[k].sum()) for k in sums}
    row.update({k: float(df[k].mean()) for k in means if k in df.columns})
    row['steps'] = int(len(df))
    row['mean_inference_time_ms'] = float(df['inference_time_ms'].astype(float).mean()) \
        if 'inference_time_ms' in df.columns else None
    return row


@click.command()
@click.option('--config-file', type=str, default='datacenter_test')
@click.option('--local-mode', type=bool, default=True)
@click.option('--num_containers', required=True, type=int, default=6)
@click.option('--type-env', required=True,
              type=click.Choice(['sim-edge']),
              default='sim-edge')
@click.option('--episode-length', required=False, type=int, default=60)
@click.option('--num-episodes', required=False, type=int, default=1)
@click.option('--checkpoint-to-load', required=False, type=str, default='last')
@click.option('--trial', required=False, type=str, default=None,
              help='substring matched against trial folder name')
@click.option('--year', required=False, type=str, default='2026',
              help='when --trial is unset, pick the trial containing this year marker')

def main(config_file: str, local_mode: bool, type_env: str,
         num_episodes: int, episode_length: int, num_containers: int,
         checkpoint_to_load: str, trial: str, year: str):
    import ray
    from gymnasium import make as gym_make
    from ray.rllib.algorithms.algorithm import Algorithm

    experiment_folder = os.path.join(TRAIN_RESULTS_PATH, f"IMPALA_{num_containers}")
    config_path = os.path.join(experiment_folder, f"{config_file}.json")

    with open(config_path) as cf:
        config_file = json.loads(cf.read())

    generator_config = deepcopy(config_file)
    del generator_config['notes']
    bitbrains_path = os.path.join(DATASETS_PATH, "bitbrains/rnd")
    yolo_path = os.path.join(DATASETS_PATH, "yolo")
    generator_config.update({'type_env': type_env})

    datacenter = DatacenterGeneration(generator_config)
    workload = WorkloadGenerator()
    env_config = generator_config['env_config_base']
    env_config.update({
        "episode_length": episode_length,
        'datacenter': datacenter,
        'datasets': {'bitbrains_path': bitbrains_path, 'yolo_path': yolo_path},
        'scheduler_path': SCHEDULER_PATH,
        'overload_threshold': workload,
    })

    ray.init(local_mode=local_mode)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    experiment_str = pick_trial(experiment_folder, trial, year)
    print(f"Using trial: {experiment_str}")

    pp.pprint(env_config)

    if type_env not in ['CartPole-v0', 'Pendulum-v0']:
        env = gym_make(ENVSMAP[type_env], config=env_config)
    else:
        raise RuntimeError("unsupported type_env")

    checkpoints = sorted(
        s for s in filter(
            lambda x: 'checkpoint' in x,
            os.listdir(os.path.join(experiment_folder, experiment_str))))
    if checkpoint_to_load == 'last':
        checkpoint_string = checkpoints[-1]
        checkpoint_path = os.path.join(experiment_folder, experiment_str, checkpoint_string)
        checkpoint_to_load_info = int(checkpoint_string.replace('checkpoint_', ''))
    else:
        cp_index = int(checkpoint_to_load)
        checkpoint_string = f"checkpoint_{cp_index:06d}"
        checkpoint_path = os.path.join(experiment_folder, experiment_str, checkpoint_string)
        checkpoint_to_load_info = cp_index

    agent = Algorithm.from_checkpoint(checkpoint_path)
    episodes = []
    episode_summaries = []
    for i in range(0, num_episodes):
        print(f"---- \nepisode: {i} ----\n")
        episode_reward = 0
        done = truncate = False
        states = []
        obs, info = env.reset()
        iter = 0
        while not done:
            start_time = time.perf_counter()
            action = agent.compute_single_action(obs)
            obs, reward, done, truncate, info = env.step(action)
            state = flatten(env.observation, action, reward, info)
            state['inference_time_ms'] = (time.perf_counter() - start_time) * 1000
            states.append(state)
            episode_reward += reward
            iter += 1
        states = pd.DataFrame(states)
        states['episode'] = i
        states['step'] = states.index
        print(f"episode reward: {episode_reward}")
        episodes.append(states)
        episode_summaries.append(summarize_episode(states))

    info = {
        'type_env': type_env,
        'checkpoint': checkpoint_to_load_info,
        'experiment_str': experiment_str,
        'episode_length': episode_length,
        'num_episodes': num_episodes,
        'algorithm': generator_config['run_or_experiment'],
        'model_type': env_config.get('model_type'),
        'penalty_accuracy': env_config['penalty_accuracy'],
        'penalty_sla': env_config['penalty_sla'],
        'penalty_consolidation': env_config['penalty_consolidation'],
        'num_workers': 3,
    }

    test_series_path = os.path.join(
        TESTS_RESULTS_PATH, generator_config['run_or_experiment'],
        'containers', str(num_containers), 'tests')
    if not os.path.isdir(test_series_path):
        os.makedirs(test_series_path)
    content = os.listdir(test_series_path)
    new_test = len(content)
    this_test_folder = os.path.join(test_series_path, str(new_test))
    os.makedirs(this_test_folder)

    all_states = pd.concat(episodes, ignore_index=True)
    all_states.to_csv(os.path.join(this_test_folder, 'states.csv'))

    summ = pd.DataFrame(episode_summaries)
    summ.to_csv(os.path.join(this_test_folder, 'summary.csv'), index=False)

    aggregate = {}
    numeric_cols = [c for c in summ.columns if summ[c].dtype == 'float64' or summ[c].dtype == 'int64']
    for c in numeric_cols:
        aggregate[c + '_mean'] = float(summ[c].mean())
        aggregate[c + '_std'] = float(summ[c].std()) if len(summ) > 1 else 0.0
    aggregate['num_episodes'] = len(summ)

    with open(os.path.join(this_test_folder, 'info.json'), 'x') as out_file:
        json.dump(info, out_file, indent=4)
        json.dump(config_file, out_file, indent=4)
    with open(os.path.join(this_test_folder, 'episodes.pickle'), 'wb') as out_pickle:
        pickle.dump(episodes, out_pickle)
    with open(os.path.join(this_test_folder, 'summary.json'), 'w') as out_summary:
        json.dump(aggregate, out_summary, indent=4)

    print(f"\nresults in: {this_test_folder}")
    print("\nper-episode summary:")
    print(summ.to_string(index=False))
    print("\nacross-episode aggregate:")
    pp.pprint(aggregate)


if __name__ == "__main__":
    main()