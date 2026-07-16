from dataclasses import dataclass


@dataclass
class LangfuseObserver:
    enabled: bool = False

    def trace(self, name: str, payload: dict[str, str]) -> dict[str, object]:
        return {"name": name, "payload": payload, "enabled": self.enabled}
