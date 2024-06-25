import datetime
import os
import threading

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars
from huggingface_hub.utils._errors import EntryNotFoundError
from loguru import logger
from pydantic import BaseModel

from competitions import __version__, utils
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

disable_progress_bars()

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


class LeaderboardRequest(BaseModel):
    lb: str


class UpdateSelectedSubmissionsRequest(BaseModel):
    submission_ids: str


class UpdateTeamNameRequest(BaseModel):
    new_team_name: str


def run_job_runner():
    job_runner = JobRunner(
        competition_id=COMPETITION_ID,
        token=HF_TOKEN,
        output_path=OUTPUT_PATH,
    )
    job_runner.run()


def start_job_runner_thread():
    thread = threading.Thread(target=run_job_runner)
    thread.daemon = True
    thread.start()
    return thread


_ = start_job_runner_thread()


app = FastAPI()
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
        return HTTPException(status_code=500, detail="HF_TOKEN is not set.")
    competition_info = CompetitionInfo(competition_id=COMPETITION_ID, autotrain_token=HF_TOKEN)
    context = {
        "request": request,
        "logo": competition_info.logo_url,
        "competition_type": competition_info.competition_type,
        "version": __version__,
        "rules_available": competition_info.rules is not None,
    }
    return templates.TemplateResponse("index.html", context)


@app.get("/login_status", response_class=JSONResponse)
async def use_oauth(request: Request, user_token: str = Depends(utils.user_authentication)):
    if user_token:
        return {"response": 2}
    return {"response": 1}


@app.get("/logout", response_class=HTMLResponse)
async def user_logout(request: Request):
    """Endpoint that logs out the user (e.g. delete cookie session)."""

    if "oauth_info" in request.session:
        request.session.pop("oauth_info", None)

    competition_info = CompetitionInfo(competition_id=COMPETITION_ID, autotrain_token=HF_TOKEN)
    context = {
        "request": request,
        "logo": competition_info.logo_url,
        "competition_type": competition_info.competition_type,
        "__version__": __version__,
        "rules_available": competition_info.rules is not None,
    }

    return templates.TemplateResponse("index.html", context)


@app.get("/competition_info", response_class=JSONResponse)
async def get_comp_info(request: Request):
    competition_info = CompetitionInfo(competition_id=COMPETITION_ID, autotrain_token=HF_TOKEN)
    info = competition_info.competition_desc
    resp = {"response": info}
    return resp


@app.get("/dataset_info", response_class=JSONResponse)
async def get_dataset_info(request: Request):
    competition_info = CompetitionInfo(competition_id=COMPETITION_ID, autotrain_token=HF_TOKEN)
    info = competition_info.dataset_desc
    resp = {"response": info}
    return resp


@app.get("/rules", response_class=JSONResponse)
async def get_rules(request: Request):
    competition_info = CompetitionInfo(competition_id=COMPETITION_ID, autotrain_token=HF_TOKEN)
    if competition_info.rules is not None:
        return {"response": competition_info.rules}
    return {"response": "No rules available."}


@app.get("/submission_info", response_class=JSONResponse)
async def get_submission_info(request: Request):
    competition_info = CompetitionInfo(competition_id=COMPETITION_ID, autotrain_token=HF_TOKEN)
    info = competition_info.submission_desc
    resp = {"response": info}
    return resp


@app.post("/leaderboard", response_class=JSONResponse)
async def fetch_leaderboard(
    request: Request, body: LeaderboardRequest, user_token: str = Depends(utils.user_authentication)
):
    lb = body.lb

    comp_org = COMPETITION_ID.split("/")[0]
    if user_token is not None:
        is_user_admin = utils.is_user_admin(user_token, comp_org)
    else:
        is_user_admin = False

    if DISABLE_PUBLIC_LB == 1 and lb == "public" and not is_user_admin:
        return {"response": "Public leaderboard is disabled by the competition host."}

    competition_info = CompetitionInfo(competition_id=COMPETITION_ID, autotrain_token=HF_TOKEN)
    leaderboard = Leaderboard(
        end_date=competition_info.end_date,
        eval_higher_is_better=competition_info.eval_higher_is_better,
        max_selected_submissions=competition_info.selection_limit,
        competition_id=COMPETITION_ID,
        token=HF_TOKEN,
        scoring_metric=competition_info.scoring_metric,
    )
    if lb == "private":
        current_utc_time = datetime.datetime.now()
        if current_utc_time < competition_info.end_date and not is_user_admin:
            return {"response": "Private leaderboard will be available after the competition ends."}
    df = leaderboard.fetch(private=lb == "private")

    if len(df) == 0:
        return {"response": "No teams yet. Why not make a submission?"}
    resp = {"response": df.to_markdown(index=False)}
    return resp


@app.post("/my_submissions", response_class=JSONResponse)
async def my_submissions(request: Request, user_token: str = Depends(utils.user_authentication)):
    competition_info = CompetitionInfo(competition_id=COMPETITION_ID, autotrain_token=HF_TOKEN)
    if user_token is None:
        return {
            "response": {
                "submissions": "",
                "submission_text": SUBMISSION_TEXT.format(competition_info.submission_limit),
                "error": "**Invalid token. Please login.**",
                "team_name": "",
            }
        }
    sub = Submissions(
        end_date=competition_info.end_date,
        submission_limit=competition_info.submission_limit,
        competition_id=COMPETITION_ID,
        token=HF_TOKEN,
        competition_type=competition_info.competition_type,
        hardware=competition_info.hardware,
    )
    try:
        subs = sub.my_submissions(user_token)
    except AuthenticationError:
        return {
            "response": {
                "submissions": "",
                "submission_text": SUBMISSION_TEXT.format(competition_info.submission_limit),
                "error": "**Invalid token. Please login.**",
                "team_name": "",
            }
        }
    subs = subs.to_dict(orient="records")
    error = ""
    if len(subs) == 0:
        error = "**You have not made any submissions yet.**"
        subs = ""
    submission_text = SUBMISSION_TEXT.format(competition_info.submission_limit)
    submission_selection_text = SUBMISSION_SELECTION_TEXT.format(competition_info.selection_limit)

    team_name = utils.get_team_name(user_token, COMPETITION_ID, HF_TOKEN)

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
    submission_comment: str = Form(None),
    user_token: str = Depends(utils.user_authentication),
):
    if submission_comment is None:
        submission_comment = ""

    if user_token is None:
        return {"response": "Invalid token"}

    todays_date = datetime.datetime.now()
    start_date = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
    if todays_date < start_date:
        comp_org = COMPETITION_ID.split("/")[0]
        if not utils.is_user_admin(user_token, comp_org):
            return {"response": "Competition has not started yet!"}

    competition_info = CompetitionInfo(competition_id=COMPETITION_ID, autotrain_token=HF_TOKEN)
    sub = Submissions(
        end_date=competition_info.end_date,
        submission_limit=competition_info.submission_limit,
        competition_id=COMPETITION_ID,
        token=HF_TOKEN,
        competition_type=competition_info.competition_type,
        hardware=competition_info.hardware,
    )
    try:
        if competition_info.competition_type == "generic":
            resp = sub.new_submission(user_token, submission_file, submission_comment)
            return {"response": f"Success! You have {resp} submissions remaining today."}
        if competition_info.competition_type == "script":
            resp = sub.new_submission(user_token, hub_model, submission_comment)
            return {"response": f"Success! You have {resp} submissions remaining today."}
    except AuthenticationError:
        return {"response": "Invalid token"}
    return {"response": "Invalid competition type"}


@app.post("/update_selected_submissions", response_class=JSONResponse)
def update_selected_submissions(
    request: Request, body: UpdateSelectedSubmissionsRequest, user_token: str = Depends(utils.user_authentication)
):
    submission_ids = body.submission_ids

    if user_token is None:
        return {"success": False, "error": "Invalid token, please login."}

    competition_info = CompetitionInfo(competition_id=COMPETITION_ID, autotrain_token=HF_TOKEN)
    sub = Submissions(
        end_date=competition_info.end_date,
        submission_limit=competition_info.submission_limit,
        competition_id=COMPETITION_ID,
        token=HF_TOKEN,
        competition_type=competition_info.competition_type,
        hardware=competition_info.hardware,
    )
    submission_ids = submission_ids.split(",")
    submission_ids = [s.strip() for s in submission_ids]
    if len(submission_ids) > competition_info.selection_limit:
        return {
            "success": False,
            "error": f"Please select at most {competition_info.selection_limit} submissions.",
        }
    sub.update_selected_submissions(user_token=user_token, selected_submission_ids=submission_ids)
    return {"success": True, "error": ""}


@app.post("/update_team_name", response_class=JSONResponse)
def update_team_name(
    request: Request, body: UpdateTeamNameRequest, user_token: str = Depends(utils.user_authentication)
):
    new_team_name = body.new_team_name

    if user_token is None:
        return {"success": False, "error": "Invalid token"}

    if str(new_team_name).strip() == "":
        return {"success": False, "error": "Team name cannot be empty."}

    try:
        utils.update_team_name(user_token, new_team_name, COMPETITION_ID, HF_TOKEN)
        return {"success": True, "error": ""}
    except Exception as e:
        return {"success": False, "error": str(e)}
