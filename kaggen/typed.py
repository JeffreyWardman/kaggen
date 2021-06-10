import importlib
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class LitModelConfig:
    model_args: Dict[str, Any]
    loss_args: Dict[str, Any]
    optimizer_args: Dict[str, Any]
    scheduler_args: Dict[str, Any]
    metrics_args: Dict[str, Any]


@dataclass
class ImportFuncConfig:
    module_name: str
    func_name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)
    call_function: bool = True

    def __post_init__(self):
        self.function = self.get_function()

    def get_function(self):
        func = getattr(importlib.import_module(self.module_name), self.func_name)
        if self.call_function:
            return func(**self.kwargs) if self.kwargs else func()
        return func


@dataclass
class DataModuleConfig(ImportFuncConfig):
    module_name: str
    func_name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)
    train_transforms: Dict[str, Any] = field(default_factory=dict)
    val_transforms: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super(DataModuleConfig, self).__post_init__()
        self.function.train_transforms = self.train_transforms
        self.function.val_transforms = self.val_transforms


@dataclass
class OptimizerConfig:
    model_params: Any
    module_name: str
    func_name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.function = self.get_function()

    def get_function(self):
        func = getattr(importlib.import_module(self.module_name), self.func_name)
        return func(self.model_params, **self.kwargs)


@dataclass
class SchedulerConfig:
    optimizer: Any
    module_name: str
    func_name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.function = self.get_function()

    def get_function(self):
        func = getattr(importlib.import_module(self.module_name), self.func_name)
        return func(self.optimizer, **self.kwargs)
