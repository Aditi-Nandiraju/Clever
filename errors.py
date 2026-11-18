"""Map a failed attempt to one error category (first matching rule wins).

Read 20-30 real errors from your dev logs and adjust these patterns before the
final analysis; list the final rules in the report so the analysis is reproducible.
"""
import re

RULES = [
    ("parse_fail",     r"^parse_fail$|did not follow the required format"),
    ("banned",         r"Forbidden construct"),
    ("timeout",        r"^Timeout$|timed out|deterministic\) timeout|maximum recursion depth|heartbeats"),
    ("termination",    r"fail to show termination|structural recursion cannot|decreasing_by|termination_by"),
    ("test_failed",    r"#test|test failed|expected .* got|decide failed|evaluated to false"),
    ("syntax",         r"unexpected token|expected term|unexpected identifier|unterminated|invalid syntax|"
                       r"expected ':='|unknown command"),
    ("unknown_name",   r"unknown (identifier|constant|namespace|tactic)|invalid field|unknown declaration"),
    ("type_mismatch",  r"type mismatch|application type mismatch|failed to synthesize|cannot be coerced|"
                       r"function expected"),
    ("unsolved_goals", r"unsolved goals|no goals to be proved|`sorry`"),
    ("tactic_failed",  r"linarith failed|omega could not|simp made no progress|rfl failed|"
                       r"The rfl tactic failed|decide failed|tactic .* failed|failed to prove|"
                       r"motive is not type correct|did not find instance of the pattern|aesop: failed"),
]
CATEGORIES = [r[0] for r in RULES] + ["other"]


def classify(record):
    """Category of a failed attempt, or None if it was proved."""
    if record.get("proved"):
        return None
    if record.get("stage") == "parse":
        return "parse_fail"
    if record.get("stage") == "banned":
        return "banned"
    text = f"{record.get('feedback') or ''}\n{record.get('raw_error') or ''}"
    for label, pattern in RULES:
        if re.search(pattern, text, re.I | re.M):
            return label
    return "other"
