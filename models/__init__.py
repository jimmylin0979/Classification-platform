# import os
# import importlib

# # Registry
# MODEL_REGISTRY = {}

# # Decorator for registry


# def register_module(name):
#     def register_model(cls):
#         # Check there is no the same name already inside the MODEL_REGISTRY
#         if name in MODEL_REGISTRY:
#             raise ValueError(f"Cannot register duplicate moedl ({name})")

#         # Register model class instance with name into MODEL_REGISTRY
#         MODEL_REGISTRY[name] = cls
#         return cls

#     return register_model


# # Automatically import the models
# models_dir = os.path.dirname(__file__)
# for file in os.listdir(models_dir):
#     path = os.path.join(models_dir, file)
#     if (
#         not file.startswith("_")
#         and not file.startswith(".")
#         and (file.endswith(".py") or os.path.isdir(path))
#     ):
#         model_name = file[: file.find(".py")] if file.endswith(".py") else file
#         module = importlib.import_module(f"models.{model_name}")

#
from .convnext import convnext
from .swin_transformer import swin_transformer
