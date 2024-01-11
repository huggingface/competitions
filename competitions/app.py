import datetime
import os
import threading

import pandas as pd
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from huggingface_hub.utils import disable_progress_bars
from loguru import logger
from pydantic import BaseModel

from competitions.errors import AuthenticationError
from competitions.info import CompetitionInfo
from competitions.leaderboard import Leaderboard
from competitions.runner import JobRunner
from competitions.submissions import Submissions
from competitions.text import SUBMISSION_SELECTION_TEXT, SUBMISSION_TEXT


HF_TOKEN = os.environ.get("HF_TOKEN", None)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMPETITION_ID = os.getenv("COMPETITION_ID")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "/tmp/model")

disable_progress_bars()

COMP_INFO = CompetitionInfo(competition_id=COMPETITION_ID, autotrain_token=HF_TOKEN)


class User(BaseModel):
    user_token: str


def run_job_runner():
    job_runner = JobRunner(token=HF_TOKEN, competition_info=COMP_INFO, output_path=OUTPUT_PATH)
    job_runner.run()


thread = threading.Thread(target=run_job_runner)
thread.start()


app = FastAPI()
static_path = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")
templates_path = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=templates_path)


@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    """
    This function is used to render the HTML file
    :param request:
    :return:
    """
    if HF_TOKEN is None:
        return templates.TemplateResponse("error.html", {"request": request})
    context = {
        "request": request,
        "logo": COMP_INFO.logo_url,
        "competition_type": COMP_INFO.competition_type,
    }
    return templates.TemplateResponse("index.html", context)


@app.get("/competition_info", response_class=JSONResponse)
async def get_comp_info(request: Request):
    info = COMP_INFO.competition_desc
    # info = markdown.markdown(info)
    resp = {"response": info}
    return resp


@app.get("/dataset_info", response_class=JSONResponse)
async def get_dataset_info(request: Request):
    info = COMP_INFO.dataset_desc
    # info = markdown.markdown(info)
    resp = {"response": info}
    return resp


@app.get("/submission_info", response_class=JSONResponse)
async def get_submission_info(request: Request):
    info = COMP_INFO.submission_desc
    # info = markdown.markdown(info)
    resp = {"response": info}
    return resp


@app.get("/leaderboard/{lb}", response_class=JSONResponse)
async def get_leaderboard(request: Request, lb: str):
    leaderboard = Leaderboard(
        end_date=COMP_INFO.end_date,
        eval_higher_is_better=COMP_INFO.eval_higher_is_better,
        max_selected_submissions=COMP_INFO.selection_limit,
        competition_id=COMPETITION_ID,
        token=HF_TOKEN,
    )
    if lb == "private":
        current_utc_time = datetime.datetime.utcnow()
        if current_utc_time < COMP_INFO.end_date:
            return {"response": "Private leaderboard will be available after the competition ends."}
    df = leaderboard.fetch(private=lb == "private")
    logger.info(df)
    resp = {"response": df.to_markdown(index=False)}
    return resp


@app.post("/my_submissions", response_class=JSONResponse)
async def my_submissions(request: Request, user: User):
    sub = Submissions(
        end_date=COMP_INFO.end_date,
        submission_limit=COMP_INFO.submission_limit,
        competition_id=COMPETITION_ID,
        token=HF_TOKEN,
        competition_type=COMP_INFO.competition_type,
    )
    try:
        success_subs, failed_subs = sub.my_submissions(user.user_token)
    except AuthenticationError:
        return {
            "response": {
                "submissions": "**Invalid token**",
                "submission_text": SUBMISSION_TEXT.format(COMP_INFO.submission_limit),
            }
        }
    subs = pd.concat([success_subs, failed_subs], axis=0)
    subs = subs.to_markdown(index=False)
    if len(subs.strip()) == 0:
        subs = "You have not made any submissions yet."
        failed_subs = ""
    submission_text = SUBMISSION_TEXT.format(COMP_INFO.submission_limit)
    submission_selection_text = SUBMISSION_SELECTION_TEXT.format(COMP_INFO.selection_limit)

    resp = {
        "response": {
            "submissions": subs,
            "submission_text": submission_text + submission_selection_text,
        }
    }
    return resp


@app.post("/new_submission", response_class=JSONResponse)
async def new_submission(
    submission_file: UploadFile = File(None),
    hub_model: str = Form(...),
    token: str = Form(...),
    submission_comment: str = Form(...),
):
    sub = Submissions(
        end_date=COMP_INFO.end_date,
        submission_limit=COMP_INFO.submission_limit,
        competition_id=COMPETITION_ID,
        token=HF_TOKEN,
        competition_type=COMP_INFO.competition_type,
    )
    try:
        if COMP_INFO.competition_type == "generic":
            resp = sub.new_submission(token, submission_file, submission_comment)
            return {"response": f"Success! You have {resp} submissions remaining today."}
        elif COMP_INFO.competition_type == "code":
            resp = sub.new_submission(token, hub_model, submission_comment)
            return {"response": f"Success! You have {resp} submissions remaining today."}
    except AuthenticationError:
        return {"response": "Invalid token"}
    return {"response": "Invalid competition type"}
