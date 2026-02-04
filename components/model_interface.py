from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Any, Dict, Optional
from enum import Enum

class InsertPosition(Enum):
    Before = "Before"
    After = "After"
    Replace = "Replace"

# List of layer types from https://www.tensorflow.org/api_docs/python/tf/keras/layers
class LayerTypes(Enum):
    Activation = "Activation"
    ActivityRegularization = "ActivityRegularization"
    Add = "Add"
    AdditiveAttention = "AdditiveAttention"
    Attention = "Attention"
    Average = "Average"
    AveragePooling1D = "AveragePooling1D"
    AveragePooling2D = "AveragePooling2D"
    AveragePooling3D = "AveragePooling3D"
    BatchNormalization = "BatchNormalization"
    BatchNormalization1D = "BatchNormalization1D"
    BatchNormalization2D = "BatchNormalization2D"
    Bidirectional = "Bidirectional"
    CategoryEncoding = "CategoryEncoding"
    CenterCrop = "CenterCrop"
    Concatenate = "Concatenate"
    Conv1D = "Conv1D"
    Conv1DTranspose = "Conv1DTranspose"
    Conv2D = "Conv2D"
    Conv2DTranspose = "Conv2DTranspose"
    Conv3D = "Conv3D"
    Conv3DTranspose = "Conv3DTranspose"
    ConvLSTM1D = "ConvLSTM1D"
    ConvLSTM2D = "ConvLSTM2D"
    ConvLSTM3D = "ConvLSTM3D"
    Cropping1D = "Cropping1D"
    Cropping2D = "Cropping2D"
    Cropping3D = "Cropping3D"
    Dense = "Dense"
    DepthwiseConv1D = "DepthwiseConv1D"
    DepthwiseConv2D = "DepthwiseConv2D"
    Discretization = "Discretization"
    Dot = "Dot"
    Dropout = "Dropout"
    ELU = "ELU"
    EinsumDense = "EinsumDense"
    Embedding = "Embedding"
    Flatten = "Flatten"
    FlaxLayer = "FlaxLayer"
    GRU = "GRU"
    GRUCell = "GRUCell"
    GaussianDropout = "GaussianDropout"
    GaussianNoise = "GaussianNoise"
    GlobalAveragePooling1D = "GlobalAveragePooling1D"
    GlobalAveragePooling2D = "GlobalAveragePooling2D"
    GlobalAveragePooling3D = "GlobalAveragePooling3D"
    GlobalMaxPooling1D = "GlobalMaxPooling1D"
    GlobalMaxPooling2D = "GlobalMaxPooling2D"
    GlobalMaxPooling3D = "GlobalMaxPooling3D"
    GroupNormalization = "GroupNormalization"
    GroupQueryAttention = "GroupQueryAttention"
    HashedCrossing = "HashedCrossing"
    Hashing = "Hashing"
    Identity = "Identity"
    InputLayer = "InputLayer"
    InputSpec = "InputSpec"
    IntegerLookup = "IntegerLookup"
    JaxLayer = "JaxLayer"
    LSTM = "LSTM"
    LSTMCell = "LSTMCell"
    Lambda = "Lambda"
    Layer = "Layer"
    LayerNormalization = "LayerNormalization"
    LeakyReLU = "LeakyReLU"
    Masking = "Masking"
    MaxPooling1D = "MaxPooling1D"
    MaxPooling2D = "MaxPooling2D"
    MaxPooling3D = "MaxPooling3D"
    Maximum = "Maximum"
    MelSpectrogram = "MelSpectrogram"
    Minimum = "Minimum"
    MultiHeadAttention = "MultiHeadAttention"
    Multiply = "Multiply"
    Normalization = "Normalization"
    PReLU = "PReLU"
    Permute = "Permute"
    RNN = "RNN"
    RandomBrightness = "RandomBrightness"
    RandomContrast = "RandomContrast"
    RandomCrop = "RandomCrop"
    RandomFlip = "RandomFlip"
    RandomRotation = "RandomRotation"
    RandomTranslation = "RandomTranslation"
    RandomZoom = "RandomZoom"
    ReLU = "ReLU"
    RepeatVector = "RepeatVector"
    Rescaling = "Rescaling"
    Reshape = "Reshape"
    Resizing = "Resizing"
    SeLU = "SeLU"
    SeparableConv1D = "SeparableConv1D"
    SeparableConv2D = "SeparableConv2D"
    SiLU = "SiLU"
    Sigmoid = "Sigmoid"
    SimpleRNN = "SimpleRNN"
    SimpleRNNCell = "SimpleRNNCell"
    Softmax = "Softmax"
    SpatialDropout1D = "SpatialDropout1D"
    SpatialDropout2D = "SpatialDropout2D"
    SpatialDropout3D = "SpatialDropout3D"
    SpectralNormalization = "SpectralNormalization"
    StackedRNNCells = "StackedRNNCells"
    StringLookup = "StringLookup"
    Subtract = "Subtract"
    TFSMLayer = "TFSMLayer"
    TextVectorization = "TextVectorization"
    ThresholdedReLU = "ThresholdedReLU"
    TimeDistributed = "TimeDistributed"
    TorchModuleWrapper = "TorchModuleWrapper"
    UnitNormalization = "UnitNormalization"
    UpSampling1D = "UpSampling1D"
    UpSampling2D = "UpSampling2D"
    UpSampling3D = "UpSampling3D"
    Wrapper = "Wrapper"
    ZeroPadding1D = "ZeroPadding1D"
    ZeroPadding2D = "ZeroPadding2D"
    ZeroPadding3D = "ZeroPadding3D"

class Params:
    ACTIVATION = "activation"
    IN_CHANNELS = "in_channels"
    OUT_CHANNELS = "out_channels"
    KERNEL_SIZE = "kernel_size"
    STRIDE = "stride"
    PADDING = "padding"
    BIAS = "bias"
    NUM_FEATURES = "num_features"
    DROPOUT_RATE = "dropout_rate"
    IN_FEATURES = "in_features"
    OUT_FEATURES = "out_features"
    IN_HEIGHT = "in_height"
    IN_WIDTH = "in_width"
    OUT_HEIGHT = "out_height"
    OUT_WIDTH = "out_width"

class LayerSpec:
    type: LayerTypes
    name: str = None
    module: Any = None
    is_activation: bool = None

    def __init__(self, type: LayerTypes, name: str = None, module: Any = None, is_activation: bool = False, params: Optional[Dict[str, Any]] = None):
        self.type = type
        self.name = name
        self.module = module
        self.is_activation = is_activation
        self.params = params or {}

    def get(self, name: str, default: Any = None) -> Any:
        return self.params.get(name, default)
    
    def set(self, name: str, value: Any) -> None:
        self.params[name] = value

    def __str__(self):
        s = "Name: " + self.name + "\n"
        s += "Type: " + self.type.value + "\n"
        s += "Activation: " + str(self.is_activation) + "\n"
        s += "Params: " + str(self.params) + "\n"
        return s

class TunerModel(ABC):

    layers: Dict[str, LayerSpec] = {}

    def __str__(self):
        s = "\n"
        for layer in self.layers.values():
            s += str(layer) + "\n"

        return s

    @abstractmethod
    def create_specs(self):
        raise NotImplementedError
    
    @abstractmethod
    def add_layers(self, layers: List[LayerSpec], targets: List[LayerTypes], position: InsertPosition):
        raise NotImplementedError

    @abstractmethod
    def remove_layers(self, target: LayerSpec, linked_layers: List[LayerTypes], delimiter: bool, first_found: bool):
        raise NotImplementedError

    @abstractmethod
    def from_type(self, layer_type: LayerTypes):
        raise NotImplementedError

    @abstractmethod
    def to_type(self, cls: Any):
        raise NotImplementedError

    @abstractmethod
    def from_spec(self, layer_spec: LayerSpec):
        raise NotImplementedError
