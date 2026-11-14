"""离线冻结独立知识任务。"""

import json
from pathlib import Path

from aftersalesagent.evaluation.e1_suite import build

if __name__ == "__main__":
    result = build(Path(__file__).resolve().parents[1])
    print(json.dumps({"task_count": result["task_count"], "splits": result["counts_by_split"]}))
