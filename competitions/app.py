import datetime
import os
import threading

import requests
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars
from huggingface_hub.utils._errors import EntryNotFoundError
from loguru import logger
from pydantic import BaseModel

from competitions import utils
from competitions.errors import AuthenticationError
from competitions.info import CompetitionInfo
from competitions.leaderboard import Leaderboard
from competitions.oauth import attach_oauth
from competitions.runner import JobRunner
from competitions.submissions import Submissions
from competitions.text import SUBMISSION_SELECTION_TEXT, SUBMISSION_TEXT


HF_TOKEN = os.environ.get("HF_TOKEN", None)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMPETITION_ID = os.environ.get("COMPETITION_ID")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/tmp/model")
START_DATE = os.environ.get("START_DATE", "2000-12-31")
DISABLE_PUBLIC_LB = int(os.environ.get("DISABLE_PUBLIC_LB", 0))
USE_OAUTH = int(os.environ.get("USE_OAUTH", 1))
VERSION_COMMIT_ID = os.environ.get("VERSION_COMMIT_ID", "0687567")

disable_progress_bars()

COMP_INFO = CompetitionInfo(competition_id=COMPETITION_ID, autotrain_token=HF_TOKEN)
RULES_AVAILABLE = COMP_INFO.rules is not None

try:
    REQUIREMENTS_FNAME = hf_hub_download(
        repo_id=COMPETITION_ID,
        filename="requirements.txt",
        token=HF_TOKEN,
        repo_type="dataset",
    )
except EntryNotFoundError:
    REQUIREMENTS_FNAME = None

if REQUIREMENTS_FNAME:
    logger.info("Uninstalling and installing requirements")
    utils.uninstall_requirements(REQUIREMENTS_FNAME)
    utils.install_requirements(REQUIREMENTS_FNAME)


class UserTeamNameUpdate(BaseModel):
    user_token: str
    new_team_name: str


class User(BaseModel):
    user_token: str


class UserLB(BaseModel):
    user_token: str
    lb: str


class UserSubmissionUpdate(BaseModel):
    user_token: str
    submission_ids: str


def run_job_runner():
    job_runner = JobRunner(token=HF_TOKEN, competition_info=COMP_INFO, output_path=OUTPUT_PATH)
    job_runner.run()


thread = threading.Thread(target=run_job_runner)
thread.start()


app = FastAPI()
if USE_OAUTH == 1:
    attach_oauth(app)

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
        "version_commit_id": VERSION_COMMIT_ID[:7],
        "rules_available": RULES_AVAILABLE,
    }
    return templates.TemplateResponse("index.html", context)


@app.get("/oauth_login", response_class=HTMLResponse)
async def oauth_login(request: Request):
    return RedirectResponse("/login/huggingface")


@app.get("/logout", response_class=HTMLResponse)
async def oauth_logout(request: Request):
    """Endpoint that logs out the user (e.g. delete cookie session)."""
    request.session.pop("oauth_info", None)
    context = {
        "request": request,
        "logo": COMP_INFO.logo_url,
        "competition_type": COMP_INFO.competition_type,
    }

    return templates.TemplateResponse("index.html", context)


@app.get("/use_oauth", response_class=JSONResponse)
async def use_oauth(request: Request):
    if USE_OAUTH == 1:
        if request.session.get("oauth_info") is not None:
            try:
                utils.user_authentication(request.session.get("oauth_info")["access_token"])
                return {"response": 2}
            except requests.exceptions.JSONDecodeError:
                request.session.pop("oauth_info", None)
                return {"response": USE_OAUTH}
    return {"response": USE_OAUTH}


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


@app.get("/rules", response_class=JSONResponse)
async def get_rules(request: Request):
    if COMP_INFO.rules is not None:
        return {"response": COMP_INFO.rules}
    return {"response": "No rules available."}


@app.get("/submission_info", response_class=JSONResponse)
async def get_submission_info(request: Request):
    info = COMP_INFO.submission_desc
    # info = markdown.markdown(info)
    resp = {"response": info}
    return resp


@app.post("/leaderboard", response_class=JSONResponse)
async def fetch_leaderboard(request: Request, user_lb: UserLB):
    if USE_OAUTH == 1:
        if request.session.get("oauth_info") is not None:
            user_lb.user_token = request.session.get("oauth_info")["access_token"]

    is_user_allowed = utils.can_user_see_private_lb(user_lb.user_token, COMPETITION_ID)

    if DISABLE_PUBLIC_LB == 1 and user_lb.lb == "public" and not is_user_allowed:
        return {"response": "Public leaderboard is disabled by the competition host."}

    leaderboard = Leaderboard(
        end_date=COMP_INFO.end_date,
        eval_higher_is_better=COMP_INFO.eval_higher_is_better,
        max_selected_submissions=COMP_INFO.selection_limit,
        competition_id=COMPETITION_ID,
        token=HF_TOKEN,
        scoring_metric=COMP_INFO.scoring_metric,
    )
    if user_lb.lb == "private":
        current_utc_time = datetime.datetime.now()
        if current_utc_time < COMP_INFO.end_date and not is_user_allowed:
            return {"response": "Private leaderboard will be available after the competition ends."}
    df = leaderboard.fetch(private=user_lb.lb == "private")

    if len(df) == 0:
        return {"response": "No teams yet. Why not make a submission?"}
    resp = {"response": df.to_markdown(index=False)}
    return resp


@app.post("/my_submissions", response_class=JSONResponse)
async def my_submissions(request: Request, user: User):
    if USE_OAUTH == 1:
        if request.session.get("oauth_info") is not None:
            user.user_token = request.session.get("oauth_info")["access_token"]

    sub = Submissions(
        end_date=COMP_INFO.end_date,
        submission_limit=COMP_INFO.submission_limit,
        competition_id=COMPETITION_ID,
        token=HF_TOKEN,
        competition_type=COMP_INFO.competition_type,
        hardware=COMP_INFO.hardware,
    )
    try:
        subs = sub.my_submissions(user.user_token)
    except AuthenticationError:
        return {
            "response": {
                "submissions": "",
                "submission_text": SUBMISSION_TEXT.format(COMP_INFO.submission_limit),
                "error": "**Invalid token**",
                "team_name": "",
            }
        }
    subs = subs.to_dict(orient="records")
    logger.info(subs)
    error = ""
    if len(subs) == 0:
        error = "**You have not made any submissions yet.**"
        subs = ""
    submission_text = SUBMISSION_TEXT.format(COMP_INFO.submission_limit)
    submission_selection_text = SUBMISSION_SELECTION_TEXT.format(COMP_INFO.selection_limit)

    team_name = utils.get_team_name(user.user_token, COMPETITION_ID, HF_TOKEN)

    resp = {
        "response": {
            "submissions": subs,
            "submission_text": submission_text + submission_selection_text,
            "error": error,
            "team_name": team_name,
        }
    }
    return resp


@app.post("/new_submission", response_class=JSONResponse)
async def new_submission(
    request: Request,
    submission_file: UploadFile = File(None),
    hub_model: str = Form(...),
    token: str = Form(None),
    submission_comment: str = Form(None),
):
    if submission_comment is None:
        submission_comment = ""

    if USE_OAUTH == 1:
        if request.session.get("oauth_info") is not None:
            token = request.session.get("oauth_info")["access_token"]

    if token is None:
        return {"response": "Invalid token"}

    todays_date = datetime.datetime.now()
    start_date = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
    if todays_date < start_date:
        comp_org = COMPETITION_ID.split("/")[0]
        if not utils.can_user_submit_before_start(token, comp_org):
            return {"response": "Competition has not started yet!"}

    sub = Submissions(
        end_date=COMP_INFO.end_date,
        submission_limit=COMP_INFO.submission_limit,
        competition_id=COMPETITION_ID,
        token=HF_TOKEN,
        competition_type=COMP_INFO.competition_type,
        hardware=COMP_INFO.hardware,
    )
    try:
        if COMP_INFO.competition_type == "generic":
            resp = sub.new_submission(token, submission_file, submission_comment)
            return {"response": f"Success! You have {resp} submissions remaining today."}
        if COMP_INFO.competition_type == "script":
            resp = sub.new_submission(token, hub_model, submission_comment)
            return {"response": f"Success! You have {resp} submissions remaining today."}
    except AuthenticationError:
        return {"response": "Invalid token"}
    return {"response": "Invalid competition type"}


@app.post("/update_selected_submissions", response_class=JSONResponse)
def update_selected_submissions(request: Request, user_sub: UserSubmissionUpdate):
    if USE_OAUTH == 1:
        if request.session.get("oauth_info") is not None:
            user_sub.user_token = request.session.get("oauth_info")["access_token"]

    sub = Submissions(
        end_date=COMP_INFO.end_date,
        submission_limit=COMP_INFO.submission_limit,
        competition_id=COMPETITION_ID,
        token=HF_TOKEN,
        competition_type=COMP_INFO.competition_type,
        hardware=COMP_INFO.hardware,
    )
    submission_ids = user_sub.submission_ids.split(",")
    submission_ids = [s.strip() for s in submission_ids]
    if len(submission_ids) > COMP_INFO.selection_limit:
        return {
            "success": False,
            "error": f"Please select at most {COMP_INFO.selection_limit} submissions.",
        }
    sub.update_selected_submissions(user_token=user_sub.user_token, selected_submission_ids=submission_ids)
    return {"success": True, "error": ""}


@app.post("/update_team_name", response_class=JSONResponse)
def update_team_name(request: Request, user_team: UserTeamNameUpdate):
    if USE_OAUTH == 1:
        if request.session.get("oauth_info") is not None:
            user_team.user_token = request.session.get("oauth_info")["access_token"]

    try:
        utils.update_team_name(user_team.user_token, user_team.new_team_name, COMPETITION_ID, HF_TOKEN)
        return {"success": True, "error": ""}
    except Exception as e:
        return {"success": False, "error": str(e)}
