from PIL import Image
from sentence_transformers import SentenceTransformer


class VisionEncoder:
    """
    Generates 512-dimensional CLIP embeddings for extracted images.
    """

    def __init__(self):
        self.model_name = "clip-ViT-B-32"
        self.model = SentenceTransformer(self.model_name)

    def encode_image(self, image_path):
        """
        Generate a normalized 512-dimensional embedding.

        Args:
            image_path (str): Path to image.

        Returns:
            list: 512-dimensional embedding.
        """

        image = Image.open(image_path).convert("RGB")

        embedding = self.model.encode(
            image,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    def build_image_profile(self, image_data, image_id):
        """
        Build a structured image profile.

        Args:
            image_data (dict): Output from extractor.py
            image_id (str): Unique image id

        Returns:
            dict
        """

        embedding = self.encode_image(
            image_data["file_path"]
        )

        return {
            "image_id": image_id,
            "page": image_data["page"],
            "caption": image_data.get("caption"),
            "file_path": image_data["file_path"],
            "embedding": embedding,
        }