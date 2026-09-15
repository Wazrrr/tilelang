"""Frozen attempts: failures consume budget and never trigger replacement."""

from dataclasses import dataclass, field


@dataclass
class AttemptLedger:
    requested: int
    selected: list[int]
    outcomes: dict[int, str] = field(default_factory=dict)

    def __post_init__(self):
        if type(self.requested) is not int or self.requested <= 0:
            raise ValueError("requested budget must be positive")
        if len(self.selected) > self.requested or len(set(self.selected)) != len(self.selected):
            raise ValueError("frozen selection must fit the budget without replacement")
        if any(type(i) is not int or i < 0 for i in self.selected) or set(self.outcomes) - set(self.selected):
            raise ValueError("invalid selected/outcome identity")
        if any(status not in ("running", "passed", "failed", "interrupted") for status in self.outcomes.values()):
            raise ValueError("invalid attempt status")

    def start(self, index):
        if index not in self.selected or index in self.outcomes:
            raise ValueError("attempt must be selected and not previously consumed")
        self.outcomes[index] = "running"

    def finish(self, index, status):
        if self.outcomes.get(index) != "running" or status not in ("passed", "failed", "interrupted"):
            raise ValueError("only running attempts can finish")
        self.outcomes[index] = status

    def to_dict(self):
        return dict(
            version=1,
            requested=self.requested,
            selected=self.selected,
            outcomes={str(k): v for k, v in self.outcomes.items()},
            consumed=len(self.outcomes),
            remaining=len(self.selected) - len(self.outcomes),
            unused=self.requested - len(self.selected),
        )

    @classmethod
    def from_dict(cls, value):
        if value.get("version") != 1:
            raise ValueError("unsupported budget version")
        ledger = cls(value["requested"], value["selected"], {int(k): v for k, v in value["outcomes"].items()})
        if ledger.to_dict() != value:
            raise ValueError("budget accounting mismatch")
        return ledger
