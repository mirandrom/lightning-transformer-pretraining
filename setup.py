from setuptools import setup
import pathlib
import re

here = pathlib.Path(__file__).parent.resolve()
readme = (here / "README.md").read_text(encoding="utf-8")
version = re.search(
    '__version__ = "([^"]+)"',
    (here / "ltp/__init__.py").read_text(encoding="utf-8")
).group(1)


setup(
    name="lightning-transformer-pretraining",
    version=version,
    author="mirandrom",
    description="Pretrain transformer language models with pytorch-lightning",
    long_description_content_type="text/markdown",
    long_description=readme,
    url="https://github.com/mirandrom/lightning-transformer-pretraining",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=['ltp'],
    python_requires=">=3.10",
    install_requires=[
        "torch ~= 1.11",
        "pytorch-lightning ~= 1.6",
        "transformers ~= 4.19",
        "datasets ~= 2.2",
        "wandb ~= 0.12",
        "deepspeed ~= 0.6"
        ],
)

