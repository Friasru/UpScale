"""Calibration Engine v1: analyzes what UpScale's archived evidence (LIVE_FORWARD outcomes,
HISTORICAL_REPLAY samples) says about its signals, thresholds and decisions, and produces
EXPERIMENTAL findings and candidate configurations only.

It never changes production scoring, stages, thresholds, confidence or decisions, never
reads HOLDOUT outside an explicit final evaluation, never promotes a candidate, never
trades and never touches keys. See `engine` for the workflow.
"""
