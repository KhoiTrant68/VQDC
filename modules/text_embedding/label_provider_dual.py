import torch
import torch.nn as nn
from typing import Tuple, Optional

class AbstractEncoder(nn.Module):
    """An abstract base class for encoders."""

    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        """Encodes the input data and returns the encoded data."""
        raise NotImplementedError

class PositionAwareSOSProvider(AbstractEncoder):
    """
    An encoder that provides start-of-sequence (SOS) tokens for unconditional training 
    with dynamic granularity quantized transformer.

    Attributes:
        coarse_sos (int): SOS token for coarse content.
        fine_sos (int): SOS token for fine content.
        coarse_pos_sos (int): SOS token for coarse position.
        fine_pos_sos (int): SOS token for fine position.
        coarse_seg_sos (int): SOS token for coarse segmentation.
        fine_seg_sos (int): SOS token for fine segmentation.
        activate_seg (bool): Whether to activate segmentation tokens.
    """

    def __init__(
        self,
        coarse_sos: int,
        coarse_pos_sos: int,
        fine_sos: Optional[int] = None,
        fine_pos_sos: Optional[int] = None,
        coarse_seg_sos: Optional[int] = None,
        fine_seg_sos: Optional[int] = None,
    ):
        """
        Initialize the PositionAwareSOSProvider.

        Args:
            coarse_sos (int): SOS token for coarse content.
            coarse_pos_sos (int): SOS token for coarse position.
            fine_sos (int, optional): SOS token for fine content.
            fine_pos_sos (int, optional): SOS token for fine position.
            coarse_seg_sos (int, optional): SOS token for coarse segmentation.
            fine_seg_sos (int, optional): SOS token for fine segmentation.
        """
        super().__init__()
        self.coarse_sos = coarse_sos
        self.fine_sos = fine_sos
        self.coarse_pos_sos = coarse_pos_sos
        self.fine_pos_sos = fine_pos_sos
        self.activate_seg = coarse_seg_sos is not None
        if self.activate_seg:
            self.coarse_seg_sos = coarse_seg_sos
            self.fine_seg_sos = fine_seg_sos

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Encodes the input data and returns the encoded data.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, ...]: A tuple of tensors containing SOS tokens.

       
        """
        batch_size = x.size(0)
        device = x.device

        c_coarse = torch.full((batch_size, 1), self.coarse_sos, device=device, dtype=torch.long)
        c_fine = torch.full((batch_size, 1), self.fine_sos, device=device, dtype=torch.long) if self.fine_sos is not None else None
        c_pos_coarse = torch.full((batch_size, 1), self.coarse_pos_sos, device=device, dtype=torch.long)
        c_pos_fine = torch.full((batch_size, 1), self.fine_pos_sos, device=device, dtype=torch.long) if self.fine_pos_sos is not None else None

        if self.activate_seg:
            c_seg_coarse = torch.full((batch_size, 1), self.coarse_seg_sos, device=device, dtype=torch.long)
            c_seg_fine = torch.full((batch_size, 1), self.fine_seg_sos, device=device, dtype=torch.long)
            return c_coarse, c_fine, c_pos_coarse, c_pos_fine, c_seg_coarse, c_seg_fine

        return c_coarse, c_fine, c_pos_coarse, c_pos_fine, None, None

class ClassForContentOnlyPositionAwareSOSProvider(AbstractEncoder):
    """
    An encoder that provides start-of-sequence (SOS) tokens for class-conditional training 
    with dynamic granularity quantized transformer. This encoder replaces the content SOS 
    tokens with class labels.

    Attributes:
        n_classes (int): Number of classes.
        threshold (int): Threshold value for class labels.
        coarse_pos_sos (int): SOS token for coarse position.
        fine_pos_sos (int): SOS token for fine position.
        coarse_seg_sos (int): SOS token for coarse segmentation.
        fine_seg_sos (int): SOS token for fine segmentation.
        activate_seg (bool): Whether to activate segmentation tokens.
    """

    def __init__(
        self,
        n_classes: int,
        threshold: int,
        coarse_pos_sos: int,
        fine_pos_sos: Optional[int] = None,
        coarse_seg_sos: Optional[int] = None,
        fine_seg_sos: Optional[int] = None,
    ):
        """
        Initialize the ClassForContentOnlyPositionAwareSOSProvider.

        Args:
            n_classes (int): Number of classes.
            threshold (int): Threshold value for class labels.
            coarse_pos_sos (int): SOS token for coarse position.
            fine_pos_sos (int, optional): SOS token for fine position.
            coarse_seg_sos (int, optional): SOS token for coarse segmentation.
            fine_seg_sos (int, optional): SOS token for fine segmentation.
        """
        super().__init__()
        self.n_classes = n_classes
        self.threshold = threshold
        self.coarse_pos_sos = coarse_pos_sos
        self.fine_pos_sos = fine_pos_sos
        self.activate_seg = coarse_seg_sos is not None
        if self.activate_seg:
            self.coarse_seg_sos = coarse_seg_sos
            self.fine_seg_sos = fine_seg_sos

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Encodes the input data and returns the encoded data.

        Args:
            x (torch.Tensor): Input tensor containing class labels.

        Returns:
            Tuple[torch.Tensor, ...]: A tuple of tensors containing encoded class labels and SOS tokens.

        """
        batch_size = x.size(0)
        device = x.device

        c_coarse = (x + self.threshold).unsqueeze(1)
        c_fine = c_coarse if self.fine_pos_sos is not None else None

        c_pos_coarse = torch.full((batch_size, 1), self.coarse_pos_sos, device=device, dtype=torch.long)
        c_pos_fine = torch.full((batch_size, 1), self.fine_pos_sos, device=device, dtype=torch.long) if self.fine_pos_sos is not None else None

        if self.activate_seg:
            c_seg_coarse = torch.full((batch_size, 1), self.coarse_seg_sos, device=device, dtype=torch.long)
            c_seg_fine = torch.full((batch_size, 1), self.fine_seg_sos, device=device, dtype=torch.long)
            return c_coarse, c_fine, c_pos_coarse, c_pos_fine, c_seg_coarse, c_seg_fine

        return c_coarse, c_fine, c_pos_coarse, c_pos_fine, None, None

class ClassAwareSOSProvider(AbstractEncoder):
    """
    A class-aware Start-of-Sequence (SOS) provider that encodes class labels with different thresholds
    for content, coarse position, and fine position.

    This provider generates six tensors:
    1. Class label + threshold_content
    2. Class label + threshold_content (repeated)
    3. Class label + threshold_coarse_position
    4. Class label + threshold_fine_position
    5. Coarse segment SOS token
    6. Fine segment SOS token

    Attributes:
        n_classes (int): Number of classes in the dataset.
        threshold_content (int): Threshold value for content encoding.
        threshold_coarse_position (int): Threshold value for coarse position encoding.
        threshold_fine_position (int): Threshold value for fine position encoding.
        coarse_seg_sos (int): SOS token for coarse segmentation.
        fine_seg_sos (int): SOS token for fine segmentation.
    """

    def __init__(
        self,
        n_classes: int,
        threshold_content: int,
        threshold_coarse_position: int,
        threshold_fine_position: int,
        coarse_seg_sos: int,
        fine_seg_sos: int,
    ):
        """
        Initialize the ClassAwareSOSProvider.

        Args:
            n_classes (int): Number of classes in the dataset.
            threshold_content (int): Threshold value for content encoding.
            threshold_coarse_position (int): Threshold value for coarse position encoding.
            threshold_fine_position (int): Threshold value for fine position encoding.
            coarse_seg_sos (int): SOS token for coarse segmentation.
            fine_seg_sos (int): SOS token for fine segmentation.
        """
        super().__init__()
        self.n_classes = n_classes
        self.threshold_content = threshold_content
        self.threshold_coarse_position = threshold_coarse_position
        self.threshold_fine_position = threshold_fine_position
        self.coarse_seg_sos = coarse_seg_sos
        self.fine_seg_sos = fine_seg_sos

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Encode the input tensor with class-aware SOS tokens.

        Args:
            x (torch.Tensor): Input tensor containing class labels.

        Returns:
            Tuple[torch.Tensor, ...]: A tuple of six tensors, each with shape (batch_size, 1),
            containing the encoded class labels and SOS tokens.

        
        """
        batch_size = x.shape[0]
        device = x.device

        content_sos = x + self.threshold_content
        coarse_pos_sos = x + self.threshold_coarse_position
        fine_pos_sos = x + self.threshold_fine_position

        coarse_seg_sos = torch.full((batch_size, 1), self.coarse_seg_sos, device=device)
        fine_seg_sos = torch.full((batch_size, 1), self.fine_seg_sos, device=device)

        return (
            content_sos.unsqueeze(1),
            content_sos.unsqueeze(1),
            coarse_pos_sos.unsqueeze(1),
            fine_pos_sos.unsqueeze(1),
            coarse_seg_sos,
            fine_seg_sos,
        )

def main():
    """
    Test function to demonstrate the usage of different SOS providers.
    This function creates instances of each provider class and tests their encode method
    with sample inputs, printing the results for verification.
    """
    # Test PositionAwareSOSProvider
    print("Testing PositionAwareSOSProvider:")
    pos_aware_provider = PositionAwareSOSProvider(
        coarse_sos=1, coarse_pos_sos=2, fine_sos=3, fine_pos_sos=4, coarse_seg_sos=5, fine_seg_sos=6
    )
    input_tensor = torch.randn(2, 10)  # Batch size of 2, arbitrary input size
    output = pos_aware_provider.encode(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output:", output)
    print("Explanation: Should return 6 tensors with shape (2, 1), containing the specified SOS tokens.")

    # Test ClassForContentOnlyPositionAwareSOSProvider
    print("\nTesting ClassForContentOnlyPositionAwareSOSProvider:")
    class_content_provider = ClassForContentOnlyPositionAwareSOSProvider(
        n_classes=10, threshold=100, coarse_pos_sos=2, fine_pos_sos=4, coarse_seg_sos=5, fine_seg_sos=6
    )
    class_input = torch.tensor([3, 7])  # Two class labels
    output = class_content_provider.encode(class_input)
    print("Input:", class_input)
    print("Output:", output)
    print("Explanation: Should return 6 tensors. The first two contain class labels + threshold (103 and 107),")
    print("             while others contain position and segment SOS tokens.")

    # Test ClassAwareSOSProvider
    print("\nTesting ClassAwareSOSProvider:")
    class_aware_provider = ClassAwareSOSProvider(
        n_classes=10, threshold_content=100, threshold_coarse_position=200,
        threshold_fine_position=300, coarse_seg_sos=5, fine_seg_sos=6
    )
    class_input = torch.tensor([3, 7])  # Two class labels
    output = class_aware_provider.encode(class_input)
    print("Input:", class_input)
    print("Output:", output)
    print("Explanation: Should return 6 tensors. The first four contain class labels + respective thresholds,")
    print("             while the last two contain segment SOS tokens.")


if __name__ == "__main__":
    main()