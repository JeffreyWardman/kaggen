import fire


def train(config_path: str):
    from kaggen.train import TrainPipeline

    return TrainPipeline(config_path)


def inference(config_path: str):
    from kaggen.inference import InferencePipeline

    return InferencePipeline(config_path)


def report(config_path: str):
    from kaggen.report import ReportPipeline

    return ReportPipeline(config_path)
