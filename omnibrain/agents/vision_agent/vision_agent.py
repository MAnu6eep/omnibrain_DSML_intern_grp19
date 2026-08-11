from dataclasses import dataclass

from omnibrain.app.services.vision.vlm_reasoner import VLMReasoner


@dataclass
class VisionAgent:
    name: str = "vision_agent"

    def analyze(self, image_path: str, metadata: dict | None = None):
        """
        Analyze an extracted chart/table image using the VLM
        while preserving visual metadata.
        """

        reasoner = VLMReasoner()

        result = reasoner.analyze_image(image_path)

        return {
            "agent": self.name,
            "image_path": image_path,
            "result": result,
            "metadata": metadata or {},
            "status": "completed",
        }