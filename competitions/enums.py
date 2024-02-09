import enum


class SubmissionStatus(enum.Enum):
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"


class CompetitionType(enum.Enum):
    GENERIC = 1
    SCRIPT = 2
