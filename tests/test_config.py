from deepshield.utils.config import load_config


def test_inheritance_and_overrides(tmp_path):
    (tmp_path / "base.yaml").write_text("a: 1\nm: {x: 1, y: 2}\nr: {k: 1}\n")
    (tmp_path / "child.yaml").write_text("_base_: base.yaml\nm: {y: 3}\nr: {_replace_: true, z: 9}\n")
    cfg = load_config(tmp_path / "child.yaml", ["m.x=5", "new.deep=[1,2]"])
    assert cfg.a == 1 and cfg.m.x == 5 and cfg.m.y == 3
    assert cfg.r == {"z": 9}
    assert cfg.new.deep == [1, 2]


def test_repo_configs_load():
    from pathlib import Path

    for p in Path("configs").rglob("*.yaml"):
        load_config(p)


def test_experiment_models_build_and_forward():
    from pathlib import Path

    import torch

    from deepshield.models.factory import build_model

    for p in sorted(Path("configs/experiments").glob("*.yaml")):
        cfg = load_config(p)
        if cfg.model.get("arch") == "ensemble" or "ensemble" in cfg or "simclr" in cfg.model:
            continue
        m = dict(cfg.model)
        if "backbone" in m:
            m["backbone"] = dict(m["backbone"], pretrained=False, checkpoint=None)
        else:
            m["pretrained"] = False
        model = build_model(m).eval()
        S = cfg.data.image_size
        with torch.no_grad():
            out = model(torch.rand(1, 2, 3, S, S))
        assert out["logits"].shape == (1, 2), p
        if p.stem == "stage2_resnet_bilstm":
            assert model.backbone.out_dim == 2048  # PRD: 2048-d spatial embeddings
