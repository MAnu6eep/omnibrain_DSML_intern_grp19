from dataclasses import dataclass, field


@dataclass
class MemoryStore:
    records: list[dict[str, str]] = field(default_factory=list)

    def save(self, record: dict[str, str]) -> None:
        self.records.append(record)
