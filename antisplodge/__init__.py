#  expose version from _version file
from ._version import __version__

# only expose needed function in the init module
from .antisplodge import DeconvolutionExperiment, SingleCellDataset, train, predict
