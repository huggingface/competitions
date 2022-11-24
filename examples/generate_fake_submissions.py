import io
import json
import os
import random
from datetime import datetime

from huggingface_hub import HfApi
from tqdm import tqdm


NUM_USERS = 1000
NUM_SUBMISSIONS = 10
MIN_SCORE = 0.0
MAX_SCORE = 1.0
START_DATE = datetime(2022, 1, 1, 0, 0, 0)
END_DATE = datetime(2022, 11, 23, 0, 0, 0)
COMPETITION_ID = os.environ.get("COMPETITION_ID")
AUTOTRAIN_TOKEN = os.environ.get("AUTOTRAIN_TOKEN")

if __name__ == "__main__":
    # example submission:
    # {"name": "abhishek", "id": "5fa19f4ba13e063b8b2b5e11", "submissions": [{"date": "2022-11-09", "time": "12:54:55", "submission_id": "c0eed646-838f-482f-bf4d-2c651f8de43b", "submission_comment": "", "status": "pending", "selected": true, "public_score": -1, "private_score": -1}, {"date": "2022-11-09", "time": "14:02:21", "submission_id": "bc6b08c2-c684-4ee1-9be2-35cf717ce618", "submission_comment": "", "status": "done", "selected": true, "public_score": 0.3333333333333333, "private_score": 0.3333333333333333}, {"date": "2022-11-17", "time": "21:31:04", "submission_id": "4a2984b1-de10-411d-be0f-9aa7be16245f", "submission_comment": "", "status": "done", "selected": false, "public_score": 0.3333333333333333, "private_score": 0.3333333333333333}]}

    for i in tqdm(range(NUM_USERS)):
        name = f"test_{i}"
        # generate random id
        id = "".join(random.choices("0123456789abcdef", k=24))
        submissions = []
        for j in range(NUM_SUBMISSIONS):
            date = START_DATE + (END_DATE - START_DATE) * random.random()
            time = date.strftime("%H:%M:%S")
            date = date.strftime("%Y-%m-%d")
            submission_id = "".join(random.choices("0123456789abcdef", k=36))
            submission_comment = ""
            status = "done"
            selected = False
            public_score = MIN_SCORE + (MAX_SCORE - MIN_SCORE) * random.random()
            private_score = MIN_SCORE + (MAX_SCORE - MIN_SCORE) * random.random()
            submission = {
                "date": date,
                "time": time,
                "submission_id": submission_id,
                "submission_comment": submission_comment,
                "status": status,
                "selected": selected,
                "public_score": public_score,
                "private_score": private_score,
            }
            submissions.append(submission)

        submission = {
            "name": name,
            "id": id,
            "submissions": submissions,
        }
        fname = f"{id}.json"
        user_submission_info_json = json.dumps(submission)
        user_submission_info_json_bytes = user_submission_info_json.encode("utf-8")
        user_submission_info_json_buffer = io.BytesIO(user_submission_info_json_bytes)
        api = HfApi()
        api.upload_file(
            path_or_fileobj=user_submission_info_json_buffer,
            path_in_repo=fname,
            repo_id=COMPETITION_ID,
            repo_type="dataset",
            token=AUTOTRAIN_TOKEN,
        )
