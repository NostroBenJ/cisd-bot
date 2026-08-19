"""Trading bot: shadow by default, live only on evidence.

The organising rule of this package is that a signal may not trade until it
carries a passing ValidationRecord -- n, a t-statistic against the null, a
holdout result and a concentration check. Everything the research in
FINDINGS.md taught is encoded as a structural constraint here rather than as
a note somebody is supposed to remember.
"""
