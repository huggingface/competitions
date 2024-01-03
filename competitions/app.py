import os

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from competitions.info import CompetitionInfo
from competitions.leaderboard import Leaderboard


HF_TOKEN = os.environ.get("HF_TOKEN", None)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMPETITION_ID = os.getenv("COMPETITION_ID")
COMP_INFO = CompetitionInfo(competition_id=COMPETITION_ID, autotrain_token=HF_TOKEN)

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
    context = {"request": request}
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


@app.get("/leaderboard/{lb}", response_class=JSONResponse)
async def get_leaderboard(request: Request, lb: str):
    leaderboard = Leaderboard(
        end_date=COMP_INFO.end_date,
        eval_higher_is_better=COMP_INFO.eval_higher_is_better,
        max_selected_submissions=COMP_INFO.selection_limit,
        competition_id=COMPETITION_ID,
        autotrain_token=HF_TOKEN,
    )
    df = leaderboard.fetch(private=lb == "private")
    resp = {"response": df.to_markdown(index=False)}
    return resp


@app.get("/submissions", response_class=JSONResponse)
async def get_submissions(request: Request, user_token: str):
    info = COMP_INFO.submission_desc
    # info = markdown.markdown(info)
    resp = {"response": info}
    return resp
