import asyncio
import os
import signal
import sqlite3
from contextlib import asynccontextmanager

import psutil
from fastapi import FastAPI
from loguru import logger

from competitions.utils import run_evaluation


def get_process_status(pid):
    try:
        process = psutil.Process(pid)
        proc_status = process.status()
        return proc_status
    except psutil.NoSuchProcess:
        logger.info(f"No process found with PID: {pid}")
        return "Completed"


def kill_process_by_pid(pid):
    """Kill process by PID."""
    os.kill(pid, signal.SIGTERM)


class JobDB:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.c = self.conn.cursor()
        self.create_jobs_table()

    def create_jobs_table(self):
        self.c.execute(
            """CREATE TABLE IF NOT EXISTS jobs
            (id INTEGER PRIMARY KEY, pid INTEGER)"""
        )
        self.conn.commit()

    def add_job(self, pid):
        sql = f"INSERT INTO jobs (pid) VALUES ({pid})"
        self.c.execute(sql)
        self.conn.commit()

    def get_running_jobs(self):
        self.c.execute("""SELECT pid FROM jobs""")
        running_pids = self.c.fetchall()
        running_pids = [pid[0] for pid in running_pids]
        return running_pids

    def delete_job(self, pid):
        sql = f"DELETE FROM jobs WHERE pid={pid}"
        self.c.execute(sql)
        self.conn.commit()


PARAMS = os.environ.get("PARAMS")
DB = JobDB("job.db")


class BackgroundRunner:
    async def run_main(self):
        while True:
            running_jobs = DB.get_running_jobs()
            if running_jobs:
                for _pid in running_jobs:
                    proc_status = get_process_status(_pid)
                    proc_status = proc_status.strip().lower()
                    if proc_status in ("completed", "error", "zombie"):
                        logger.info(f"Process {_pid} is already completed. Skipping...")
                        try:
                            kill_process_by_pid(_pid)
                        except Exception as e:
                            logger.info(f"Error while killing process: {e}")
                        DB.delete_job(_pid)

            running_jobs = DB.get_running_jobs()
            if not running_jobs:
                logger.info("No running jobs found. Shutting down the server.")
                os.kill(os.getpid(), signal.SIGINT)
            await asyncio.sleep(30)


runner = BackgroundRunner()


@asynccontextmanager
async def lifespan(app: FastAPI):
    process_pid = run_evaluation(params=PARAMS)
    logger.info(f"Started training with PID {process_pid}")
    DB.add_job(process_pid)
    asyncio.create_task(runner.run_main())
    yield


api = FastAPI(lifespan=lifespan)


@api.get("/")
async def root():
    return "Your model is being evaluated..."


@api.get("/health")
async def health():
    return "OK"
