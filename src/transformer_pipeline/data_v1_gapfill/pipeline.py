from __future__ import annotations

from . import audit, build, plot, split, train_check


def main(*, run: bool = False, train: str = "none") -> None:
    build.main(run=run)
    split.main(run=run)
    if run:
        audit.main()
        plot.main()
    else:
        print("audit")
        print("plot")
    if train != "none":
        train_check.main(model=train, run=run)
