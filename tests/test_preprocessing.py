import numpy as np

from preprocessing.image_preprocessing import preprocess_frame


def test_preprocess_returns_expected_shapes():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    result = preprocess_frame(frame)
    assert result.color.ndim == 3
    assert result.gray.shape == result.blurred.shape
    assert result.edges.shape == result.gray.shape
