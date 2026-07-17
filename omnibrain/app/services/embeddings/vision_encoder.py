from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor


class VisionEncoder:
    """
    Generates CLIP embeddings for extracted images.
    """

    def __init__(self):
        self.model_name = "openai/clip-vit-base-patch32"

        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name)

        self.model.eval()

    def encode_image(self, image_path):
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        with torch.no_grad():

            vision_outputs = self.model.vision_model(
                pixel_values=inputs["pixel_values"]
            )

            pooled = vision_outputs.pooler_output

            embedding = self.model.visual_projection(pooled)

        embedding = embedding.squeeze(0)

        return embedding.tolist()


if __name__ == "__main__":
    encoder = VisionEncoder()

    embedding = encoder.encode_image(
        "output/images/page_1_image_1.jpeg"
    )

    print(f"Embedding length: {len(embedding)}")