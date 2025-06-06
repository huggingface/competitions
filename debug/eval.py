import pandas as pd
from metric import _metric

solution_file = "/tmp/data/solution.csv"
solution_df = pd.read_csv(solution_file).set_index("id")

submission_file = "/tmp/model/submission.csv"
submission_df = pd.read_csv(submission_file).set_index("id")

evaluation = _metric(solution_df, submission_df)

print(evaluation)

