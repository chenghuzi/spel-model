from collections import defaultdict

import numpy as np


def get_collective_hs(exp_data_groups):
    # Concatenate the paired session populations without changing their original order.
    collective_exp_hs_dict_mean_list_pre = defaultdict(list)
    collective_exp_hs_dict_mean_list_bal = defaultdict(list)
    collective_exp_hs_dict_raw_lists_pre = defaultdict(list)
    collective_exp_hs_dict_raw_lists_bal = defaultdict(list)

    ng_names = list(exp_data_groups.keys())
    four_conditions = exp_data_groups[ng_names[0]]["pre"][1].keys()

    for cond in four_conditions:
        nnids_all = []
        for ng_name in ng_names:
            nnids_all.append(exp_data_groups[ng_name]["nnids"])
            collective_exp_hs_dict_mean_list_pre[cond].append(
                exp_data_groups[ng_name]["pre"][1][cond]["mean"]
            )
            collective_exp_hs_dict_raw_lists_pre[cond].append(
                exp_data_groups[ng_name]["pre"][1][cond]["raw"]
            )

            collective_exp_hs_dict_mean_list_bal[cond].append(
                exp_data_groups[ng_name]["bal"][1][cond]["mean"]
            )
            collective_exp_hs_dict_raw_lists_bal[cond].append(
                exp_data_groups[ng_name]["bal"][1][cond]["raw"]
            )

        nnids_all = np.concatenate(nnids_all)
        collective_exp_hs_dict_mean_list_bal[cond] = np.concatenate(
            collective_exp_hs_dict_mean_list_bal[cond], axis=1
        )

        collective_exp_hs_dict_mean_list_bal[cond] = {
            "mean": collective_exp_hs_dict_mean_list_bal[cond],
        }
        collective_exp_hs_dict_mean_list_pre[cond] = np.concatenate(
            collective_exp_hs_dict_mean_list_pre[cond], axis=1
        )
        collective_exp_hs_dict_mean_list_pre[cond] = {
            "mean": collective_exp_hs_dict_mean_list_pre[cond]
        }
    return (
        collective_exp_hs_dict_mean_list_pre,
        collective_exp_hs_dict_mean_list_bal,
        collective_exp_hs_dict_raw_lists_pre,
        collective_exp_hs_dict_raw_lists_bal,
        nnids_all,
    )
