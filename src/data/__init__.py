"""데이터셋 팩토리 — config의 data.source로 로더 선택 (기본: act_sim)."""


def get_dataset(cfg):
    if cfg["data"].get("source") == "libero":
        from data.libero import LiberoDataset
        return LiberoDataset(cfg)
    from data.act_sim import ActSimDataset
    return ActSimDataset(cfg)
