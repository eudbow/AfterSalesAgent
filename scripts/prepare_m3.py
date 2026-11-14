"""离线冻结54个本项目适配任务；已有不同版本时拒绝覆盖。"""

import json
from pathlib import Path

from aftersalesagent.evaluation.suite import freeze_suite

if __name__ == "__main__":
    result = freeze_suite(Path(__file__).resolve().parents[1])
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "suite_id",
                    "task_count",
                    "counts_by_split",
                    "counts_by_kind",
                    "source_commit",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
