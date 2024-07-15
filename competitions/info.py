import io
import json
from dataclasses import dataclass
from datetime import datetime

from huggingface_hub import HfApi, hf_hub_download


@dataclass
class CompetitionInfo:
    competition_id: str
    autotrain_token: str

    def __post_init__(self):
        config_fname = hf_hub_download(
            repo_id=self.competition_id,
            filename="conf.json",
            use_auth_token=self.autotrain_token,
            repo_type="dataset",
        )
        competition_desc = hf_hub_download(
            repo_id=self.competition_id,
            filename="COMPETITION_DESC.md",
            use_auth_token=self.autotrain_token,
            repo_type="dataset",
        )
        dataset_desc = hf_hub_download(
            repo_id=self.competition_id,
            filename="DATASET_DESC.md",
            use_auth_token=self.autotrain_token,
            repo_type="dataset",
        )
        self.config = self.load_config(config_fname)
        self.competition_desc = self.load_md(competition_desc)
        self.dataset_desc = self.load_md(dataset_desc)
        try:
            submission_desc = hf_hub_download(
                repo_id=self.competition_id,
                filename="SUBMISSION_DESC.md",
                use_auth_token=self.autotrain_token,
                repo_type="dataset",
            )
            self.submission_desc = self.load_md(submission_desc)
        except Exception:
            self.submission_desc = None

        try:
            rules_md = hf_hub_download(
                repo_id=self.competition_id,
                filename="RULES.md",
                use_auth_token=self.autotrain_token,
                repo_type="dataset",
            )
            self.rules_md = self.load_md(rules_md)
        except Exception:
            self.rules_md = None

        if self.config["EVAL_METRIC"] == "custom":
            if "SCORING_METRIC" not in self.config:
                raise ValueError(
                    "For custom metrics, please provide a single SCORING_METRIC name in the competition config file: conf.json"
                )

    def load_md(self, md_path):
        with open(md_path, "r", encoding="utf-8") as f:
            md = f.read()
        return md

    def load_config(self, config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config

    @property
    def submission_limit(self):
        return self.config["SUBMISSION_LIMIT"]

    @property
    def selection_limit(self):
        return self.config["SELECTION_LIMIT"]

    @property
    def end_date(self):
        e_d = self.config["END_DATE"]
        return datetime.strptime(e_d, "%Y-%m-%d")

    @property
    def eval_higher_is_better(self):
        hb = self.config["EVAL_HIGHER_IS_BETTER"]
        return True if int(hb) == 1 else False

    @property
    def competition_description(self):
        return self.competition_desc

    @property
    def submission_columns(self):
        return self.config["SUBMISSION_COLUMNS"].split(",")

    @property
    def submission_columns_raw(self):
        return self.config["SUBMISSION_COLUMNS"]

    @property
    def submission_description(self):
        return self.submission_desc

    @property
    def dataset_description(self):
        return self.dataset_desc

    @property
    def logo_url(self):
        return self.config["LOGO"]

    @property
    def competition_type(self):
        return self.config["COMPETITION_TYPE"].lower().strip()

    @property
    def metric(self):
        return self.config["EVAL_METRIC"]

    @property
    def submission_id_col(self):
        return self.config["SUBMISSION_ID_COLUMN"]

    @property
    def submission_cols(self):
        cols = self.config["SUBMISSION_COLUMNS"].split(",")
        cols = [c.strip() for c in cols]
        return cols

    @property
    def submission_rows(self):
        return self.config["SUBMISSION_ROWS"]

    @property
    def time_limit(self):
        return self.config["TIME_LIMIT"]

    @property
    def hardware(self):
        return self.config.get("HARDWARE", "cpu-basic")

    @property
    def dataset(self):
        return self.config.get("DATASET", "")

    @property
    def submission_filenames(self):
        return self.config.get("SUBMISSION_FILENAMES", ["submission.csv"])

    @property
    def scoring_metric(self):
        if self.config["EVAL_METRIC"] == "custom":
            if "SCORING_METRIC" not in self.config:
                raise Exception("Please provide a single SCORING_METRIC in the competition config file: conf.json")
            if self.config["SCORING_METRIC"] is None:
                raise Exception("Please provide a single SCORING_METRIC in the competition config file: conf.json")
            return self.config["SCORING_METRIC"]
        return self.config["EVAL_METRIC"]

    @property
    def rules(self):
        return self.rules_md

    def _save_md(self, md, filename, api):
        md = io.BytesIO(md.encode())
        api.upload_file(
            path_or_fileobj=md,
            path_in_repo=filename,
            repo_id=self.competition_id,
            repo_type="dataset",
        )

    def update_competition_info(self, config, markdowns, token):
        api = HfApi(token=token)
        conf_json = json.dumps(config, indent=4)
        conf_json_bytes = conf_json.encode("utf-8")
        conf_json_buffer = io.BytesIO(conf_json_bytes)
        api.upload_file(
            path_or_fileobj=conf_json_buffer,
            path_in_repo="conf.json",
            repo_id=self.competition_id,
            repo_type="dataset",
        )

        competition_desc = markdowns["competition_desc"]
        dataset_desc = markdowns["dataset_desc"]
        submission_desc = markdowns["submission_desc"]
        rules_md = markdowns["rules"]

        self._save_md(competition_desc, "COMPETITION_DESC.md", api)
        self._save_md(dataset_desc, "DATASET_DESC.md", api)
        self._save_md(submission_desc, "SUBMISSION_DESC.md", api)
        self._save_md(rules_md, "RULES.md", api)
