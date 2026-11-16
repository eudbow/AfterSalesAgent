"""Hybrid retrieval plus explicit business dependencies over the unchanged v1 corpus."""

import hashlib
import json
import re

from aftersalesagent.knowledge.retail import RetailPolicyIndex
from aftersalesagent.policies.retail_v2 import capability_hash, capability_prompt

RELATIONS = {
    "return": {
        "pattern": r"退货|return",
        "required": ["POL-11", "05_returns", "06_refund_methods", "09_application_protocol"],
    },
    "cancel": {
        "pattern": r"取消|cancel",
        "required": ["POL-07", "03_cancellation", "09_application_protocol"],
    },
    "exchange": {
        "pattern": r"换货|exchange|variant|规格|颜色",
        "required": ["POL-12", "01_service_scope", "07_products_and_statuses"],
    },
    "address": {
        "pattern": r"地址|改址|address",
        "required": ["POL-08", "04_address_change", "01_service_scope"],
    },
    "payment": {
        "pattern": r"支付|payment|银行卡|bank card",
        "required": ["POL-09", "06_refund_methods", "01_service_scope"],
    },
    "unknown": {
        "pattern": r"赔偿|破损|损坏|运费|手续费|compensation|damag|fee",
        "required": ["08_exceptions_and_handoff", "09_application_protocol"],
    },
}


class RetailV2Index(RetailPolicyIndex):
    """Own config fingerprint while retaining the exact sourced corpus/version."""

    def __init__(self, **kwargs):
        kwargs.setdefault("method", "hybrid")
        super().__init__(**kwargs)
        self.fingerprint = hashlib.sha256(
            json.dumps(
                {"base": self.fingerprint, "relations": RELATIONS, "capability": capability_hash()},
                sort_keys=True,
            ).encode()
        ).hexdigest()

    def fixed_evidence(self):
        return {
            "APP-CAP-V2": dict(
                chunk_id="APP-CAP-V2",
                text=capability_prompt(),
                version="2.0.0",
                scope="retail_sim",
                origin="application_capability",
                source="application:retail-capabilities@2.0.0",
                section="应用能力与结果含义",
                rule_ids=["APP-CAP-V2"],
                supplied_by="server_fixed",
            )
        }

    def validate_snapshot(self, evidence):
        supplied = evidence.get("APP-CAP-V2")
        if supplied != self.fixed_evidence()["APP-CAP-V2"]:
            raise ValueError("Retail v2 fixed capability mismatch")
        super().validate_snapshot({k: v for k, v in evidence.items() if k != "APP-CAP-V2"})

    def search(self, args, *, session_id, call_id):
        data = super().search(args, session_id=session_id, call_id=call_id)
        query = args.query + " " + (args.topic or "")
        topics = [
            name for name, rule in RELATIONS.items() if re.search(rule["pattern"], query, re.I)
        ]
        required = list(
            dict.fromkeys(cid for name in topics for cid in RELATIONS[name]["required"])
        )
        original = {record["chunk_id"]: record for record in data["evidence"]}
        evidence, used, missing = [], 0, []
        for cid in dict.fromkeys([*required, *original]):
            chunk = self.by_id[cid]
            if used + len(chunk.text) > self.char_budget:
                if cid in required:
                    missing.append(cid)
                continue
            used += len(chunk.text)
            evidence.append(
                original.get(cid)
                or dict(
                    chunk_id=cid,
                    text=chunk.text,
                    version=chunk.version,
                    scope=chunk.scope,
                    origin=chunk.origin,
                    source=chunk.source,
                    section=chunk.section,
                    rule_ids=self.coverage.get(cid, {}).get("rule_ids", [cid]),
                    supplied_by="dependency",
                )
            )
        data["diagnostics"] = {
            "method": self.method,
            "topics": topics,
            "ranked_bundle_ids": list(original),
            "required_ids": required,
            "missing_required_ids": missing,
            "definition": "Structural topic coverage, not semantic entailment",
        }
        data["evidence"] = evidence
        if missing or len(evidence) < len(set(required) | set(original)):
            data["coverage_status"] = "truncated"
        elif evidence:
            data["coverage_status"] = "retrieved"
        return data
