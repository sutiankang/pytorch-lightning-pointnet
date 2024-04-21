def build_model(cfg):
    if cfg.model_name == 'pointnet_cls':
        from models.pointnet_cls import PointNetCls
        model = PointNetCls(cfg)
    else:
        raise NotImplementedError
    return model


def build_dataset(cfg, split):
    if cfg.dataset_name in ['modelnet10', 'modelnet40']:
        from datasets.classification.modelnet import ModelNetDataset
        dataset = ModelNetDataset(cfg, split)
    else:
        raise NotImplementedError

    return dataset


def build_model_engine(cfg):
    if cfg.model_name == 'pointnet_cls':
        from models.engine import ClassifierEngine
        model_engine = ClassifierEngine(cfg)
    else:
        raise NotImplementedError
    return model_engine
