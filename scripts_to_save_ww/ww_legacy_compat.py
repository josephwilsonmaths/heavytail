"""PyTorch-only import support for legacy WeightWatcher 0.2.7."""
import sys
import types
from importlib.machinery import ModuleSpec
import numpy as np

def _add_spec(module, name, is_package=False):
    """
    Give fake modules a valid importlib spec.
    This avoids torch._dynamo / importlib.find_spec failures.
    """
    module.__spec__ = ModuleSpec(name=name, loader=None, is_package=is_package)
    if is_package:
        module.__path__ = []
    return module

def install_tf_keras_stubs():
    """
    weightwatcher==0.2.7 imports/checks tensorflow/keras even for PyTorch models.
    This provides enough dummy Keras/TensorFlow structure for PyTorch-only analysis.
    """

    class DummyKerasModel:
        pass

    class DummyKerasLayer:
        pass

    def dummy_load_model(*args, **kwargs):
        raise RuntimeError(
            "Keras load_model was called, but this script is intended for "
            "PyTorch torchvision models only."
        )

    keras_models_stub = _add_spec(
        types.ModuleType("keras.models"),
        "keras.models",
        is_package=False,
    )
    keras_models_stub.Model = DummyKerasModel
    keras_models_stub.Sequential = DummyKerasModel
    keras_models_stub.load_model = dummy_load_model

    keras_core_stub = _add_spec(
        types.ModuleType("keras.layers.core"),
        "keras.layers.core",
        is_package=False,
    )
    keras_core_stub.Dense = DummyKerasLayer
    keras_core_stub.Flatten = DummyKerasLayer
    keras_core_stub.Dropout = DummyKerasLayer
    keras_core_stub.Activation = DummyKerasLayer

    keras_conv_stub = _add_spec(
        types.ModuleType("keras.layers.convolutional"),
        "keras.layers.convolutional",
        is_package=False,
    )
    keras_conv_stub.Conv1D = DummyKerasLayer
    keras_conv_stub.Conv2D = DummyKerasLayer
    keras_conv_stub.Conv3D = DummyKerasLayer

    keras_norm_stub = _add_spec(
        types.ModuleType("keras.layers.normalization"),
        "keras.layers.normalization",
        is_package=False,
    )
    keras_norm_stub.BatchNormalization = DummyKerasLayer

    keras_pool_stub = _add_spec(
        types.ModuleType("keras.layers.pooling"),
        "keras.layers.pooling",
        is_package=False,
    )
    keras_pool_stub.MaxPooling1D = DummyKerasLayer
    keras_pool_stub.MaxPooling2D = DummyKerasLayer
    keras_pool_stub.AveragePooling1D = DummyKerasLayer
    keras_pool_stub.AveragePooling2D = DummyKerasLayer
    keras_pool_stub.GlobalAveragePooling2D = DummyKerasLayer

    keras_backend_stub = _add_spec(
        types.ModuleType("keras.backend"),
        "keras.backend",
        is_package=False,
    )

    keras_layers_stub = _add_spec(
        types.ModuleType("keras.layers"),
        "keras.layers",
        is_package=True,
    )
    keras_layers_stub.Layer = DummyKerasLayer
    keras_layers_stub.Dense = DummyKerasLayer
    keras_layers_stub.Flatten = DummyKerasLayer
    keras_layers_stub.Dropout = DummyKerasLayer
    keras_layers_stub.Activation = DummyKerasLayer
    keras_layers_stub.Conv1D = DummyKerasLayer
    keras_layers_stub.Conv2D = DummyKerasLayer
    keras_layers_stub.Conv3D = DummyKerasLayer
    keras_layers_stub.BatchNormalization = DummyKerasLayer

    # Old Keras nested-module style
    keras_layers_stub.core = keras_core_stub
    keras_layers_stub.convolutional = keras_conv_stub
    keras_layers_stub.normalization = keras_norm_stub
    keras_layers_stub.pooling = keras_pool_stub

    keras_stub = _add_spec(
        types.ModuleType("keras"),
        "keras",
        is_package=True,
    )
    keras_stub.__version__ = "2.3.1"
    keras_stub.models = keras_models_stub
    keras_stub.layers = keras_layers_stub
    keras_stub.backend = keras_backend_stub

    tf_compat_v1_stub = _add_spec(
        types.ModuleType("tensorflow.compat.v1"),
        "tensorflow.compat.v1",
        is_package=False,
    )

    tf_compat_stub = _add_spec(
        types.ModuleType("tensorflow.compat"),
        "tensorflow.compat",
        is_package=True,
    )
    tf_compat_stub.v1 = tf_compat_v1_stub

    tf_stub = _add_spec(
        types.ModuleType("tensorflow"),
        "tensorflow",
        is_package=True,
    )
    tf_stub.__version__ = "2.0.0"
    tf_stub.keras = keras_stub
    tf_stub.compat = tf_compat_stub

    sys.modules["tensorflow"] = tf_stub
    sys.modules["tensorflow.compat"] = tf_compat_stub
    sys.modules["tensorflow.compat.v1"] = tf_compat_v1_stub

    sys.modules["keras"] = keras_stub
    sys.modules["keras.models"] = keras_models_stub
    sys.modules["keras.layers"] = keras_layers_stub
    sys.modules["keras.layers.core"] = keras_core_stub
    sys.modules["keras.layers.convolutional"] = keras_conv_stub
    sys.modules["keras.layers.normalization"] = keras_norm_stub
    sys.modules["keras.layers.pooling"] = keras_pool_stub
    sys.modules["keras.backend"] = keras_backend_stub

def import_legacy_weightwatcher():
    import importlib.metadata
    version = importlib.metadata.version('weightwatcher')
    if version != '0.2.7':
        raise RuntimeError(f'Use the ww_old_current_tv environment: expected WW 0.2.7, found {version}.')
    for name in ('NAN', 'NaN'):
        if not hasattr(np, name):
            setattr(np, name, np.nan)
    if not hasattr(np, 'Inf'):
        np.Inf = np.inf
    # The legacy package imports Keras even for PyTorch-only analysis.
    if 'weightwatcher' not in sys.modules:
        install_tf_keras_stubs()
    import weightwatcher
    if getattr(weightwatcher, '__version__', None) != '0.2.7':
        raise RuntimeError('A different WW version is already loaded; restart the 0.2.7 kernel.')
    return weightwatcher
