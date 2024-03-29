import importlib
import os
import sys

import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn import metrics


def compute_metrics(params):
    if params.metric == "custom":
        metric_file = hf_hub_download(
            repo_id=params.competition_id,
            filename="metric.py",
            token=params.token,
            repo_type="dataset",
        )
        sys.path.append(os.path.dirname(metric_file))
        metric = importlib.import_module("metric")
        evaluation = metric.compute(params)
    else:
        solution_file = hf_hub_download(
            repo_id=params.competition_id,
            filename="solution.csv",
            token=params.token,
            repo_type="dataset",
        )

        solution_df = pd.read_csv(solution_file)

        submission_filename = f"submissions/{params.team_id}-{params.submission_id}.csv"
        submission_file = hf_hub_download(
            repo_id=params.competition_id,
            filename=submission_filename,
            token=params.token,
            repo_type="dataset",
        )
        submission_df = pd.read_csv(submission_file)

        public_ids = solution_df[solution_df.split == "public"][params.submission_id_col].values
        private_ids = solution_df[solution_df.split == "private"][params.submission_id_col].values

        public_solution_df = solution_df[solution_df[params.submission_id_col].isin(public_ids)]
        public_submission_df = submission_df[submission_df[params.submission_id_col].isin(public_ids)]

        private_solution_df = solution_df[solution_df[params.submission_id_col].isin(private_ids)]
        private_submission_df = submission_df[submission_df[params.submission_id_col].isin(private_ids)]

        public_solution_df = public_solution_df.sort_values(params.submission_id_col).reset_index(drop=True)
        public_submission_df = public_submission_df.sort_values(params.submission_id_col).reset_index(drop=True)

        private_solution_df = private_solution_df.sort_values(params.submission_id_col).reset_index(drop=True)
        private_submission_df = private_submission_df.sort_values(params.submission_id_col).reset_index(drop=True)

        _metric = getattr(metrics, params.metric)
        target_cols = [col for col in solution_df.columns if col not in [params.submission_id_col, "split"]]
        public_score = _metric(public_solution_df[target_cols], public_submission_df[target_cols])
        private_score = _metric(private_solution_df[target_cols], private_submission_df[target_cols])

        # scores can also be dictionaries for multiple metrics
        evaluation = {
            "public_score": {
                params.metric: public_score,
            },
            "private_score": {
                params.metric: private_score,
            },
        }

    # check all keys in public_score and private_score are same
    if evaluation["public_score"].keys() != evaluation["private_score"].keys():
        raise ValueError("Public and private scores have different keys")
    return evaluation
