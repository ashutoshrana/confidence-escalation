"""Check an isolated installed base wheel without development dependencies."""

import asyncio
import compileall
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
import sys

import confidence_escalation as package
from confidence_escalation.async_middleware import AsyncConfidenceEscalationMiddleware

assert package.__version__ == version("confidence-escalation")
assert Path(package.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert find_spec("pytest") is None
assert compileall.compile_dir(Path(package.__file__).parent, quiet=1)

effects = []
gate = package.ConfidenceEscalationMiddleware(policy=package.ThresholdPolicy(threshold=0.6))
for evidence in [package.ConfidenceScore(0.2, package.ScoringMethod.COMPOSITE), gate.score()]:
    try:
        gate.call_guarded(effects.append, evidence, "blocked")
    except PermissionError:
        pass
    else:
        raise AssertionError("Low or missing evidence executed an action")
gate.call_guarded(effects.append, package.ConfidenceScore(0.8, package.ScoringMethod.COMPOSITE), "allowed")
assert effects == ["allowed"]


async def check_async():
    gate = AsyncConfidenceEscalationMiddleware(policy=package.ThresholdPolicy(threshold=0.6))
    effects = []

    async def action():
        effects.append("allowed")

    try:
        await gate.call_guarded(action, package.ConfidenceScore(0.2, package.ScoringMethod.COMPOSITE))
    except PermissionError:
        pass
    else:
        raise AssertionError("Async low confidence executed an action")
    assert effects == []
    await gate.call_guarded(action, package.ConfidenceScore(0.8, package.ScoringMethod.COMPOSITE))
    assert effects == ["allowed"]


asyncio.run(check_async())
print("Installed base runtime: sync/async permit, deny and missing-evidence checks passed")
