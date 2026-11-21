import pytest
import torch

from ml.inference.predict import load_link_checkpoint


def test_checkpoint_loader_reads_required_metadata(tmp_path):
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": {"weight": torch.tensor([1.0, 2.0])},
            "feature_names": ["source_sampoq"],
            "model_config": {"hidden_channels": 8},
            "dataset_version": "synthetic",
        },
        path,
    )
    checkpoint = load_link_checkpoint(path)
    assert checkpoint["feature_names"] == ["source_sampoq"]
    assert checkpoint["model_config"]["hidden_channels"] == 8
    assert checkpoint["dataset_version"] == "synthetic"


def test_checkpoint_loader_rejects_missing_keys(tmp_path):
    path = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": {}}, path)
    with pytest.raises(ValueError, match="missing keys"):
        load_link_checkpoint(path)
