import pandas as pd
from huggingface_hub import hf_hub_download

def _metric(solution_df,submission_df, mode = "top_level"):
    """
    This function calculates the accuracy of the generated predictions.

    Parameters
    ----------
    solution_df : pandas.DataFrame
        The dataframe containing the solution data.
    submission_df : pandas.DataFrame
        The dataframe containing the submission data.
    mode : str, optional
        The mode of evaluation. Can be "top_level" or "bottom_level". The default is "top_level".
    
    Returns
    -------
    None.
    """


    solution_df["submission_pred"] = submission_df["pred"]
    cols = ["split","pred","source"]


    solution_df["correct"] = solution_df["pred"] == solution_df["submission_pred"]
    accuracy = solution_df.groupby(cols)["correct"].mean().to_frame("accuracy").reset_index()
    accuracy["score_name"] = accuracy["pred"] +"_"+ accuracy["source"]
    
    evaluation = {}
    
    for split,temp in accuracy.groupby("split"):
        scores_by_source = temp.set_index("score_name")["accuracy"].sort_index()
        scores_by_source["generated_accuracy"] = temp.query("pred=='generated'")["accuracy"].mean()
        scores_by_source["pristine_accuracy"] = temp.query("pred=='pristine'")["accuracy"].mean()
        scores_by_source["balanced_accuracy"] = (scores_by_source["generated_accuracy"] + scores_by_source["pristine_accuracy"])/2.
        if mode == "top_level":
            scores_to_save = ["generated_accuracy", "pristine_accuracy", "balanced_accuracy"]
            evaluation[f"{split}_score"] = scores_by_source.loc[scores_to_save].to_dict()
        else:
            evaluation[f"{split}_score"] = scores_by_source.to_dict()

    if "time" in submission_df.columns:
        solution_df["submission_time"] = submission_df["time"]
        for split, temp in solution_df.groupby("split"):
            evaluation[f"{split}_score"]["total_time"] = float(temp["submission_time"].sum())

    # evaluation = {
    #     "public_score": {
    #         "metric1": public_score,
    #     },
    #     "private_score": {
    #         "metric1": private_score,
    #     }
    # }
    return evaluation



def compute(params):
    solution_file = hf_hub_download(
        repo_id=params.competition_id,
        filename="solution.csv",
        token=params.token,
        repo_type="dataset",
    )

    solution_df = pd.read_csv(solution_file).set_index(params.submission_id_col)

    submission_filename = f"submissions/{params.team_id}-{params.submission_id}.csv"
    submission_file = hf_hub_download(
        repo_id=params.competition_id,
        filename=submission_filename,
        token=params.token,
        repo_type="dataset",
    )
    
    submission_df = pd.read_csv(submission_file).set_index(params.submission_id_col)

    return _metric(solution_df,submission_df)