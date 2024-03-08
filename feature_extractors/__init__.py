from collections.abc import Sequence
from abc import ABC, abstractmethod
import torch
from PIL.Image import Image
from typing import Literal

class FeatureExtractor(ABC):
    @abstractmethod
    def encode_image(self, img_list: Sequence[Image]) -> torch.Tensor:
        """
        Encode the input images and return the corresponding embeddings.

        Args:
            img_list: A list of PIL.Image.Image objects.

        Returns:
            The embeddings of the input images. The shape should be (len(img_list), embedding_dim).
        """
        raise NotImplementedError

    @abstractmethod
    def encode_text(self, text_list: Sequence[str]) -> torch.Tensor:
        """
        Encode the input text data and return the corresponding embeddings.

        Args:
            text_list: A list of strings.

        Returns:
            The embeddings of the input text data. The shape should be (len(text_list), embedding_dim).
        """
        raise NotImplementedError

    @abstractmethod
    def encode_3D(self, pc_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encode the input 3D point cloud and return the corresponding embeddings.

        Args:
            pc_tensor: A tensor of shape (B, N, 3 + 3).
        
        Returns:
            The embeddings of the input 3D point cloud. The shape should be (B, embedding_dim).
        """
        raise NotImplementedError

    @abstractmethod
    def encode_query(self, queries: Sequence[str]) -> torch.Tensor:
        """Encode the queries and return the corresponding embeddings.

        Args:
            queries: A list of strings.

        Returns:
            The embeddings of the input text data. The shape should be (len(input_text), embedding_dim).
        """
        raise NotImplementedError