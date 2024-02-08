import enum


class SubmissionStatus(enum.Enum):
    PENDING = 0
    QUEUED = 1
    PROCESSING = 2
    SUCCESS = 3
    FAILED = 4


class CompetitionType(enum.Enum):
    GENERIC = 1
    SCRIPT = 2
