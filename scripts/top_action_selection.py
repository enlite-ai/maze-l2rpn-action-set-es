"""Contains methods for retrospective action set reduction from historic trajectories."""
import argparse
import os
from collections import Counter

import grid2op
import matplotlib.pyplot as plt
import numpy as np
from grid2op.Converter import IdToAct
from maze.core.trajectory_recording.datasets.in_memory_dataset import InMemoryDataset
from maze.core.trajectory_recording.datasets.trajectory_processor import IdentityTrajectoryProcessor


def parse_args():
    """Argument parser for composing submission"""
    parser = argparse.ArgumentParser(description="Retrospective action set reduction from historic trajectories.")
    parser.add_argument("--trajectory_data", default=None, help="Path to trajectory dumps.", type=str)
    parser.add_argument("--n_workers", default=1, help="Number of workers for parallel data loading.", type=int)
    parser.add_argument("--keep_k", default=None, help="Number of top actions to keep.", type=int)
    parser.add_argument("--dump_file", default=None, help=".npy file where to dump the action candidates.", type=str)
    parser.add_argument("--power_grid", default=None,
                        help="Corresponding grid2op power_grid dataset. (e.g., rte_case14_realistic)", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    """ main """

    # parse arguments
    args = parse_args()

    # check input directory
    input_directory = args.trajectory_data
    assert os.path.exists(input_directory)

    # instantiate dataset
    dataset = InMemoryDataset(input_data=input_directory,
                              n_workers=args.n_workers,
                              conversion_env_factory=None,
                              deserialize_in_main_thread=False,
                              trajectory_processor=IdentityTrajectoryProcessor())

    # collect actions
    actions_taken = []
    for i in range(len(dataset)):
        _, actions, _ = dataset[i]
        actions_taken.append(actions[0]['action'])

    # count distinct actions
    counts = Counter(actions_taken)

    # ensure proper sorting
    action_ids = []
    action_counts = []
    for k, v in counts.items():
        action_ids.append(k)
        action_counts.append(v)

    action_ids = np.asarray(action_ids, dtype=np.int)
    action_counts = np.asarray(action_counts, dtype=np.int)
    sorted_idxs = np.argsort(action_counts)[::-1]
    action_ids = action_ids[sorted_idxs]
    action_counts = action_counts[sorted_idxs]

    # dump top candidates
    if args.dump_file:
        assert args.keep_k is not None, "Select the number of <args.keep_k> top actions to keep!"

        # keep only top k actions
        actions_to_keep = action_ids[:args.keep_k]

        # dump top k action indices to npy file
        action_selection_vector = np.asarray(actions_to_keep, dtype=np.int)
        np.save(args.dump_file, action_selection_vector)

    # visualize action histogram
    plt.figure("Action Distribution")
    plt.clf()
    plt.bar(range(len(action_counts)), action_counts)
    if args.keep_k:
        plt.plot([args.keep_k + 0.5] * 2, [0, max(action_counts)], "k-", linewidth=2, alpha=0.5)
        plt.text(args.keep_k + 0.5, max(action_counts) * 1.01, f"<- {args.keep_k} actions",
                 ha="center", va="bottom", color="k")
    ax = plt.gca()
    ax.set_xticks(range(len(action_counts)))
    tick_labels = [f"{action_id}\n({i + 1})" for i, action_id in enumerate(action_ids)]
    ax.set_xticklabels(tick_labels)
    ax.yaxis.grid()
    plt.xlabel("Action ID")
    plt.ylabel("Count")
    plt.title(f"Action Histogram | overall distinct actions {len(action_counts)}")

    # substation analysis
    if args.power_grid is not None:
        env = grid2op.make(args.power_grid)
        id_to_act = IdToAct(env.action_space)
        id_to_act.init_converter(set_line_status=False,
                                 change_line_status=False,
                                 set_topo_vect=True,
                                 change_bus_vect=False,
                                 redispatch=False,
                                 curtail=False,
                                 storage=False)

        # map actions to sub_stations
        action_id_to_sub = dict()
        for action_id, action in enumerate(id_to_act.all_actions):
            action_dict = action.as_dict()
            if "set_bus_vect" not in action_dict:
                action_id_to_sub[action_id] = None
            else:
                action_id_to_sub[action_id] = int(action_dict["set_bus_vect"]["modif_subs_id"][0])

        sub_station_sets = []
        sub_station_counts = []
        for num_actions in range(len(action_ids)):
            contained_action_ids = action_ids[:num_actions+1]
            sub_stations = set([action_id_to_sub[action_id] for action_id in contained_action_ids
                                if action_id_to_sub[action_id] is not None])
            sub_station_sets.append(sub_stations)
            sub_station_counts.append(len(sub_stations))

        plt.figure("Substation Counts")
        plt.plot(range(len(action_ids)), sub_station_counts, "b-", alpha=0.5)

        # visualize substation ids
        prev_count = 0
        for i, count in enumerate(sub_station_counts):
            if count > prev_count:
                plt.text(i, count, sub_station_sets[i], va="center", ha="right")
                prev_count = count
                print(f"{count}: {sub_station_sets[i]}")

        ax = plt.gca()
        ax.set_xticks(range(len(action_counts)))
        tick_labels = [f"{action_id}\n({i+1})" for i, action_id in enumerate(action_ids)]
        ax.set_xticklabels(tick_labels)
        plt.grid()
        plt.xlabel("Action ID")
        plt.ylabel("Substation Count")
        plt.title(f"Overall number of substations {env.n_sub}")

    # show resulting plots
    plt.show(block=True)
