"""Explicit opt-in computational budgets, independent of mandatory RAM plans.

Exact tied/conditional likelihoods can require superlinear work. Protective
defaults may be raised deliberately through positive integer environment
settings. This never disables live-workspace, numerical or disk guards.
"""
from __future__ import annotations

import os

from openecon.analysis_contracts import AnalysisError


def work_budget(name:str,default:int)->int:
    value = os.environ.get(name,str(default)).strip()
    if not value.isascii() or not value.isdecimal() or int(value) <=0:
        raise AnalysisError("invalid_resource_budget",f"{name} must be a positive integer native-work budget; raising it permits more computation but never disables memory or numerical guards.")
    return int(value)
