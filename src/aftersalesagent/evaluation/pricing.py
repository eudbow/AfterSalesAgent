"""显式价格快照下的估算；缺usage、缓存分项或价格时返回未知，不推断零成本。"""

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class PriceSchedule(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    currency: str = "USD"
    input_hit_per_million: Decimal = Field(ge=0)
    input_miss_per_million: Decimal = Field(ge=0)
    output_per_million: Decimal = Field(ge=0)
    source_url: str
    verified_on: str
    rate_label: str


def estimate_cost(events, pricing: PriceSchedule | None):
    if pricing is None:
        return None
    cost = Decimal(0)
    observations = []
    for event in events:
        if event.kind == "model_response":
            observations.append(event.payload)
        elif event.kind == "model_error" and isinstance(event.payload.get("usage"), dict):
            observations.append(event.payload["usage"])
    if len(observations) != sum(e.kind == "model_request" for e in events):
        return None
    for item in observations:
        prompt, output = item.get("input_tokens"), item.get("output_tokens")
        if type(prompt) is not int or type(output) is not int:
            return None
        metadata = item.get("provider_metadata") or {}
        hit = metadata.get("prompt_cache_hit_tokens")
        if type(hit) is not int or not 0 <= hit <= prompt:
            return None
        miss = metadata.get("prompt_cache_miss_tokens", prompt - hit)
        if type(miss) is not int or hit + miss != prompt:
            return None
        cost += hit * pricing.input_hit_per_million + miss * pricing.input_miss_per_million
        cost += output * pricing.output_per_million
    return {
        "amount": str(cost / Decimal(1_000_000)),
        "currency": pricing.currency,
        "pricing": pricing.model_dump(mode="json"),
        "estimated_not_invoice": True,
    }
