import logging

from kaggen.train.pipeline import TrainPipeline


def main(train_config_path: str, class_config_path: str) -> None:
    logging.getLogger().setLevel(logging.INFO)
    train_pipeline = TrainPipeline(train_config_path, class_config_path, logger=get_logger())
    train_pipeline()


if __name__ == '__main__':
    train_config_path = 'configs/train.yaml'
    class_config_path = 'configs/classes.yaml'
    main(train_config_path, class_config_path)
