"""生成并审计独立36任务；冻结文件不覆盖。"""

import json
from pathlib import Path

from aftersalesagent.evaluation.nl_suite import freeze

if __name__ == "__main__":
    print(json.dumps(freeze(Path(__file__).resolve().parents[1]), ensure_ascii=False, indent=2))
