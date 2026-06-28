from __future__ import annotations

from . import audit, build, plot, report, split, train_check


def main(*, run: bool = False, train: str = "none") -> None:
    build.main(run=run)
    split.main(run=run)
    if run:
        audit.main()
        plot.main()
        report.main()
    else:
        print("audit")
        print("plot")
        print("report")
    if train != "none":
        train_check.main(model=train, run=run)
