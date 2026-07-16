from dataclasses import dataclass


@dataclass
class VisionAgent:
    name: str = "vision_agent"

    def analyze(self, image_path: str) -> dict[str, str]:
        return {"agent": self.name, "image_path": image_path, "status": "placeholder"}
