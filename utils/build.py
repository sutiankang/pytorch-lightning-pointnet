def build_model(cfg):
    if cfg.model_name == 'pointnet_cls':
        from models.classification.pointnet_cls import PointNetCls
        model = PointNetCls(cfg)
    elif cfg.model_name == 'pointnet_sem_seg':
        from models.segmentation.pointnet_sem_seg import PointNetSemSeg
        model = PointNetSemSeg(cfg)
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
        from models.classification.engine import ClassifierEngine
        model_engine = ClassifierEngine(cfg)
    else:
        raise NotImplementedError
    return model_engine
