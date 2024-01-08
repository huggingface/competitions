import os
from typing import List

from pydantic import BaseModel


class EvalParams(BaseModel):
    competition_id: str
    competition_type: str
    metric: str
    token: str
    team_id: str
    submission_id: str
    submission_id_col: str
    submission_cols: List[str]
    submission_rows: int
    output_path: str
    submission_repo: str
    time_limit: int

    class Config:
        protected_namespaces = ()

    def save(self, output_dir):
        """
        Save parameters to a json file.
        """
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "params.json")
        # save formatted json
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=4))
