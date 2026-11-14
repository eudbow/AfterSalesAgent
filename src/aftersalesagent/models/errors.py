"""厂商异常只暴露稳定分类；响应正文、Header与Key不进入日志或异常文本。"""

from aftersalesagent.schemas.contracts import ModelResponse


class ModelBackendFault(Exception):
    def __init__(
        self,
        code: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
        observation: ModelResponse | None = None,
    ):
        self.code, self.retryable, self.status_code = code, retryable, status_code
        self.observation = observation
        super().__init__(code)

    def public(self) -> dict:
        return {"code": self.code, "retryable": self.retryable, "status_code": self.status_code}
