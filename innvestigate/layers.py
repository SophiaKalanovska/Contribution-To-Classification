from __future__ import annotations

from typing import Sequence

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as kbackend

import tensorflow_probability as tfp


import tensorflow_probability as tfp
from tensorflow.python.ops.numpy_ops import np_config

import tensorflow.keras.layers as klayers

import innvestigate.backend as ibackend
from innvestigate.backend.types import OptionalList, ShapeTuple, Tensor

__all__ = [
    "OnesLike",
    "AsFloatX",
    "FiniteCheck",
    "GradientWRT",
    "GreaterThanZero",
    "LessEqualThanZero",
    "Sum",
    "Identity",
    "Abs",
    "Square",
    "Clip",
    "Project",
    "SafeDivide",
    "Repeat",
    "ReduceMean",
    "Reshape",
    "AugmentationToBatchAxis",
    "AugmentationFromBatchAxis",
    "MultiplyWithLinspace",
    "AddGaussianNoise",
    "ExtractConv2DPatches",
    "RunningMeans",
    "Broadcast",
    "MaxNeuronSelection",
    "MaxNeuronIndex",
    "NeuronSelection",
]


###############################################################################
np_config.enable_numpy_behavior()
tfd = tfp.distributions
class OnesLike(klayers.Layer):
    """Create list of all-ones tensors of the same shapes as provided tensors."""

    def call(self, inputs: OptionalList[Tensor], *_args, **_kwargs) -> list[Tensor]:
        return [kbackend.ones_like(x) for x in ibackend.to_list(inputs)]


class AsFloatX(klayers.Layer):
    def call(self, inputs: OptionalList[Tensor], *_args, **_kwargs) -> list[Tensor]:
        return [ibackend.cast_to_floatx(x) for x in ibackend.to_list(inputs)]


class FiniteCheck(klayers.Layer):
    def call(self, inputs: OptionalList[Tensor], *_args, **_kwargs) -> list[Tensor]:
        return [
            kbackend.sum(ibackend.cast_to_floatx(ibackend.is_not_finite(x)))
            for x in ibackend.to_list(inputs)
        ]


###############################################################################


class GradientWRT(klayers.Layer):
    "Returns gradient wrt to another layer and given gradient,"
    " expects inputs+[output,]."

    def __init__(self, n_inputs, mask=None, dynamic= False, **kwargs):
        self.n_inputs = n_inputs
        self.mask = mask
        super(GradientWRT, self).__init__(**kwargs)

    def call(self, x):
        assert isinstance(x, (list, tuple))
        Xs, tmp_Ys = x[:self.n_inputs], x[self.n_inputs:]
        assert len(tmp_Ys) % 2 == 0
        len_Ys = len(tmp_Ys) // 2
        Ys, known_Ys = tmp_Ys[:len_Ys], tmp_Ys[len_Ys:]
        ret = tf.gradients(Ys, Xs, known_Ys)
        if self.mask is not None:
            ret = [x for c, x in zip(self.mask, ret) if c]
        self.__workaround__len_ret = len(ret)
        return ret

    def compute_output_shape(self, input_shapes):
        if self.mask is None:
            return input_shapes[:self.n_inputs]
        else:
            return [x for c, x in zip(self.mask, input_shapes[:self.n_inputs])
                    if c]

    # todo: remove once keras is fixed.
    # this is a workaround for cases when
    # wrapper and skip connections are used together.
    # bring the fix into keras and remove once
    # keras is patched.
    def compute_mask(self, inputs, mask=None):
        """Computes an output mask tensor.

        # Arguments
            inputs: Tensor or list of tensors.
            mask: Tensor or list of tensors.

        # Returns
            None or a tensor (or list of tensors,
                one per output tensor of the layer).
        """
        if not self.supports_masking:
            if mask is not None:
                if isinstance(mask, list):
                    if any(m is not None for m in mask):
                        raise TypeError('Layer ' + self.name +
                                        ' does not support masking, '
                                        'but was passed an input_mask: ' +
                                        str(mask))
                else:
                    raise TypeError('Layer ' + self.name +
                                    ' does not support masking, '
                                    'but was passed an input_mask: ' +
                                    str(mask))
            # masking not explicitly supported: return None as mask

            # this is the workaround for model.run_internal_graph.
            # it is required that there as many masks as outputs:
            return [None for _ in range(self.__workaround__len_ret)]
        # if masking is explicitly supported, by default
        # carry over the input mask
        return mask

class _Reduce(klayers.Layer):
    def __init__(
        self,
        *args,
        axis: OptionalList[int] | None = -1,
        keepdims: bool = False,
        **kwargs,
    ) -> None:
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(*args, **kwargs)

    def call(self, inputs: OptionalList[Tensor], *_args, **_kwargs) -> Tensor:
        return self._apply_reduce(inputs, axis=self.axis, keepdims=self.keepdims)

    def _apply_reduce(
        self,
        inputs: OptionalList[Tensor],
        axis: OptionalList[int] | None,
        keepdims: bool,
    ) -> Tensor:
        raise NotImplementedError()


class Sum(_Reduce):
    def _apply_reduce(
        self,
        inputs: OptionalList[Tensor],
        axis: OptionalList[int] | None,
        keepdims: bool,
    ) -> Tensor:
        return kbackend.sum(inputs, axis=axis, keepdims=keepdims)


###############################################################################

class Bn_forward(klayers.Layer):

    "Returns bn input with weighted beta"
    " expects inputs+[output,]."

    def __init__(self, n_inputs, **kwargs):
        self.n_inputs = n_inputs
        super(Bn_forward, self).__init__(**kwargs)

    def call(self, x):
        assert isinstance(x, (list, tuple))
        input_normalised, tmp_Ys = x[:self.n_inputs], x[self.n_inputs:]
        # Zs_pos_neg, tmp_Ys = tmp_Ys[:len(tmp_Ys) - 1], tmp_Ys[1:]
        # Zs_prime, tmp_Ys = tmp_Ys[:len(tmp_Ys)-1], tmp_Ys[1:]
        Rs, tmp_Ys = tmp_Ys[:1], tmp_Ys[1:]
        beta, _reverse_state = tmp_Ys[:1], tmp_Ys[1:]
        # contribution_normalised = contribution_normalised[0]
        # beta = beta[0]

        _reverse_state = _reverse_state[0]

        contribution_normalised = ibackend.apply(_reverse_state["layer"], Rs)

        contribution_normalised_minus_beta = tf.math.subtract(contribution_normalised, beta)[0]

        act_normalised_minus_beta = tf.math.subtract(input_normalised, beta)[0]
        hey = [tf.math.divide_no_nan(contribution_normalised_minus_beta, act_normalised_minus_beta)]

        ratio_con_act = [tf.clip_by_value(hey[0], clip_value_min=0, clip_value_max=1)]
        # multiplier = tf.constant(10 ** 3, dtype=hey[0].dtype)
        # hey = [tf.round(hey[0] * multiplier) / multiplier]


        weighted_output = tf.math.multiply(ratio_con_act, beta)[0]
        contribution = tf.math.add(contribution_normalised_minus_beta, weighted_output)

        y = tf.constant(0.0)
        mask_the_zeros = tf.math.not_equal(Rs[0], y)
        contribution = [tf.math.multiply(contribution, tf.cast(mask_the_zeros, tf.float32))]

        return contribution


    def compute_output_shape(self, input_shapes):
        return input_shapes[:self.n_inputs]


class Avgpool_forward(klayers.Layer):

    "Returns bn input with weighted beta"
    " expects inputs+[output,]."

    def __init__(self, reverse_state, **kwargs):
        self.reverse_state = reverse_state
        super(Avgpool_forward, self).__init__(**kwargs)

    def call(self, x):
        reverse_state = self.reverse_state[0]
        return ibackend.apply(reverse_state["layer"], x)

class ApplyLayerToList(klayers.Layer):

    "Returns bn input with weighted beta"
    " expects inputs+[output,]."

    def __init__(self, reverse_state, **kwargs):
        self.reverse_state = reverse_state
        super(ApplyLayerToList, self).__init__(**kwargs)

    def call(self, list_con):
        reverse_state = self.reverse_state[0]
        return [ibackend.apply(reverse_state, [a])[0] for a in list_con]

class ApplyLayerToTwoList(klayers.Layer):

    "Returns bn input with weighted beta"
    " expects inputs+[output,]."
    def call(self, layer, list_con):
        layer = layer[0]
        a, b = list_con
        if len(b) > 1 and len(a) > 1:
            return [layer([x, y])for x, y in zip(a, b)]
        elif len(a) == 1:
            return [layer([a[0], y]) for y in b]
        else:
            return [layer([x, b[0]]) for x in a]


class Default_forward(klayers.Layer):

    "Returns bn input with weighted beta"
    " expects inputs+[output,]."

    def __init__(self, n_inputs, **kwargs):
        self.n_inputs = n_inputs
        super(Default_forward, self).__init__(**kwargs)

    def call(self, x):
        assert isinstance(x, (list, tuple))
        Rs, reverse_state = x[:self.n_inputs], x[self.n_inputs:]

        reverse_state = reverse_state[0]
        new_Ys = ibackend.apply(reverse_state["layer"], Rs)

        return new_Ys

    def compute_output_shape(self, input_shapes):
        return input_shapes[:self.n_inputs]

class Max_neuron_forward(klayers.Layer):

    "Returns bn input with weighted beta"
    " expects inputs+[output,]."

    def __init__(self, n_inputs, **kwargs):
        self.n_inputs = n_inputs
        super(Max_neuron_forward, self).__init__(**kwargs)

    def call(self, x):
        assert isinstance(x, (list, tuple))
        Xs, tmp_Ys = x[:self.n_inputs], x[self.n_inputs:]
        Ys, tmp_Ys = tmp_Ys[:1], tmp_Ys[1:]
        reversed_Ys, reverse_state = tmp_Ys[:1], tmp_Ys[1:]

        reverse_state = reverse_state[0]

        grad = tf.gradients(Ys, Xs)
        max_activ = [tf.math.multiply(grad, reversed_Ys)[0]]

        absolute_Ys = ibackend.apply(reverse_state["layer"], tf.math.abs(max_activ))[0]
        non_Ys = ibackend.apply(reverse_state["layer"], max_activ)[0]
        new_Ys = tf.where(tf.equal(non_Ys, 0), -absolute_Ys, non_Ys)

        return new_Ys

    def compute_output_shape(self, input_shapes):
        return input_shapes[:self.n_inputs]

class _Map(klayers.Layer):
    def call(
        self, inputs: OptionalList[Tensor], *_args, **_kwargs
    ) -> OptionalList[Tensor]:
        if isinstance(inputs, list) and len(inputs) == 1:
            inputs = inputs[0]
        return self._apply_map(inputs)

    def _apply_map(self, X: Tensor):
        raise NotImplementedError()


class Identity(_Map):
    def _apply_map(self, X: Tensor) -> Tensor:
        return tf.identity(X)

class ReduceDimention(_Map):
    def _apply_map(self, X: Tensor) -> Tensor:
        return [x[0] for x in X]
class Abs(_Map):
    def _apply_map(self, X: Tensor) -> Tensor:
        return kbackend.abs(X)


class Square(_Map):
    def _apply_map(self, X: Tensor) -> Tensor:
        return kbackend.square(X)



# class Round(_Map):
#     def _apply_map(self, X: Tensor) -> Tensor:
#         multiplier = tf.constant(1000, dtype=X[0].dtype)
#         multiplied = tf.multiply(X, multiplier)
#         round = tf.round(multiplied)
#
#         return tf.math.divide_no_nan(round, multiplier)
        # return tf.round(X)

class Clip(_Map):
    def __init__(
        self, min_value: float | int | Tensor, max_value: float | int | Tensor
    ) -> None:
        self._min_value = min_value
        self._max_value = max_value
        super().__init__()

    def _apply_map(self, X: Tensor) -> Tensor:
        return tf.clip_by_value(X[0], clip_value_min= self._min_value, clip_value_max=self._max_value)


class Project(_Map):
    def __init__(self, output_range=False, input_is_postive_only: bool = False) -> None:
        # TODO: add type of output_range
        self._output_range = output_range
        self._input_is_positive_only = input_is_postive_only
        super().__init__()

    def _apply_map(self, X: Tensor):
        dims: tuple[int] = kbackend.int_shape(X)
        n_dim: int = len(dims)
        axes = tuple(range(1, n_dim))

        if len(axes) == 1:
            # TODO(albermax): this is only the case when the dimension in this
            # axis is 1, fix this.
            # Cannot reduce
            return X

        absmax = kbackend.max(kbackend.abs(X), axis=axes, keepdims=True)
        X = ibackend.safe_divide(X, absmax, factor=1)

        if self._output_range not in (False, True):  # True = (-1, +1)
            output_range = self._output_range

            if not self._input_is_positive_only:
                X = (X + 1) / 2
            X = kbackend.clip(X, 0, 1)

            X = output_range[0] + (X * (output_range[1] - output_range[0]))
        else:
            X = kbackend.clip(X, -1, 1)

        return X


###############################################################################


class GreaterThanZero(klayers.Layer):
    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        return kbackend.greater(inputs, kbackend.constant(0))


class LessEqualThanZero(klayers.Layer):
    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        return kbackend.less_equal(inputs, kbackend.constant(0))
class LessThanZero(klayers.Layer):
    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        return kbackend.less(inputs, kbackend.constant(0))
    
class LessThan(klayers.Layer):
    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        a, b = inputs
        return kbackend.less(a, b)    
    
class MoreThanZero(klayers.Layer):    
    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        return kbackend.less(kbackend.constant(0), inputs)    
    
class MoreThan(klayers.Layer):    
    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        a, b = inputs
        return kbackend.less(b, a)    
    

class MoreThanOne(klayers.Layer):    
    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        return kbackend.less(kbackend.constant(1), inputs)        
    

class MoreThanThree(klayers.Layer):    
    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        return kbackend.less(kbackend.constant(5), inputs)        
    
class Not_Equal_Zero(klayers.Layer):
    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        y = tf.constant(0.0)
        return tf.math.not_equal(inputs[0], y)
class Equal_Zero(klayers.Layer):
    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        y = tf.constant(0.0)
        return tf.math.equal(inputs[0], y)

class Absolut(klayers.Layer):
    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        return tf.math.abs(inputs[0])

class Squareroot(klayers.Layer):
    def call(self, X: Tensor, *_args, **_kwargs) -> Tensor:
        return tf.math.sqrt(X)
    
class Squeeze(klayers.Layer):
    def call(self, X: Tensor, *_args, **_kwargs) -> Tensor:
        return tf.squeeze(X)

class Squeeze_First_dim(klayers.Layer):
    def call(self, X: Tensor, *_args, **_kwargs) -> Tensor:
        return tf.squeeze(X, 0)
    
class Floor(klayers.Layer):
    def call(self, X: Tensor, *_args, **_kwargs) -> Tensor:
        return tf.floor(X)

class Ceil(klayers.Layer):
    def call(self, X: Tensor, *_args, **_kwargs) -> Tensor:
        return tf.math.ceil(X)
    
class Reduce_Sum(klayers.Layer):
    def call(self, X: Tensor, *_args, **_kwargs) -> Tensor:
        return tf.math.reduce_sum(X, 0)

class Bound_by_Zero_One(klayers.Layer):
    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        return tf.clip_by_value(inputs[0], clip_value_min=0, clip_value_max=1)
class Bound_by_x(klayers.Layer):
    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        return tf.clip_by_value(inputs[0], clip_value_min=0, clip_value_max=1)

class Bound_by_minus_one_to_zero(klayers.Layer):
    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        return tf.clip_by_value(inputs[0], clip_value_min=-1, clip_value_max=0)
class Bound_by_minus_one_to_one(klayers.Layer):
    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        return tf.clip_by_value(inputs[0], clip_value_min=-1, clip_value_max=1)

class ToSparse(klayers.Layer):
    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        return tf.sparse.from_dense(inputs[0])

class Squared_difference(klayers.Layer):
    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        a, b = inputs
        return tf.math.squared_difference(a, b)


# class Reduce_sum(klayers.Layer):
#     def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
#         return tf.math.reduce_sum(tf.math.reduce_sum(inputs[0], 2), 1)


class Reduce_mean(klayers.Layer):
    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        shape = inputs.shape
        if len(shape) != 1:
            inputs = tf.reshape(inputs[0], [-1, inputs[0].shape[-1]])
        if (inputs.shape[0] == 1):
            return tf.math.reduce_mean(inputs, -1)
        else:
            return tf.math.reduce_mean(inputs, 0)
        # return tfp.stats.percentile(flatten_a, 50.0, interpolation='midpoint')
    
class Reduce_min(klayers.Layer):
    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        flatten_a = tf.reshape(inputs[0], [-1, inputs[0].shape[-1]])
        return tf.reduce_min(flatten_a, axis=0)

class Reduce_max(klayers.Layer):
    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        if len(inputs.shape) != 1:
            inputs = tf.reshape(inputs[0], [-1, inputs[0].shape[-1]])
        return tf.reduce_max(inputs, axis=0)

class tensor_transformation(klayers.Layer):
    def call(self, scaled_value: Tensor, value: Tensor, ratio: Tensor, *_args, **_kwargs) -> Tensor:
        clip_threshold = 1.0
        gradients = tf.gradients(tf.abs(scaled_value), tf.abs(value))[0]
        clipped_gradients = tf.clip_by_value(gradients, -clip_threshold, clip_threshold)
        scaled_gradients = clipped_gradients / ratio
        scaled_value = tf.identity(value) + scaled_gradients
        return scaled_value



class Add_withoutLast(klayers.Layer):
    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        toReturn = None
        for i in range(len(inputs) - 2):
            if toReturn == None:
                toReturn = inputs[0]
            else:
                toReturn = tf.math.add(toReturn, inputs[i])

        return tf.math.divide_no_nan(toReturn, tf.cast(tf.constant(len(inputs) - 2), tf.float32))


class Reduce_mean_sparse(klayers.Layer):
    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        a, b = inputs

        if len(a.shape) != 1:
            a = tf.reshape(a, [-1, a.shape[-1]])
            b = tf.reshape(b, [-1, b.shape[-1]])


        if (a.shape[0] == 1):
            sum = tf.math.reduce_sum(a, -1)
            n = tf.math.reduce_sum(b, -1)
        else:
            sum = tf.math.reduce_sum(a, 0)
            n = tf.math.reduce_sum(b, 0)
        
        mean = tf.math.divide_no_nan(sum, n)
        return mean
class Average(klayers.Layer):
    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        inputs = inputs[1:]
        # flatten_a = tf.reshape(a, [-1, a.shape[-1]])
        # flatten_b = tf.reshape(b, [-1, b.shape[-1]])
        n = len(inputs)
        stacked = tf.stack(inputs)
        sum = tf.math.reduce_sum(stacked, 0)
        mean = tf.math.divide_no_nan(sum, n)

        return mean
class Reduce_var_to_input_sparse(klayers.Layer):
    def __init__(self, *args, activations: Tensor, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.activations = activations[0]

    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        a, b = inputs
        #
        flatten_b = tf.reshape(b, [-1, b.shape[-1]])

        n = tf.math.reduce_sum(flatten_b, 0)

        squared_difference_1 = tf.math.squared_difference(a, self.activations)

        squared_difference_1 = tf.math.multiply(squared_difference_1, b)

        squared_difference_flatten = tf.reshape(squared_difference_1, [-1, squared_difference_1.shape[-1]])

        sum = tf.math.reduce_sum(squared_difference_flatten, 0)

        var_1 = tf.math.divide_no_nan(sum, n)

        return var_1
    
class Colapse_Sum(klayers.Layer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        shape = inputs.shape

        if len(shape) != 1:
            a = Reshape([-1, shape[-1]])(inputs)
            sum = Reduce_Sum()(a)
        else:
            sum = Reduce_Sum()(inputs)
        return sum
    

class Colapse_Sum_Act(klayers.Layer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        a, b = inputs
       

        new_a = Multiply()([a, b])
        shape = new_a.shape
        if len(shape) != 1:
            new_a = Reshape([-1, shape[-1]])(new_a)
            sum = Reduce_Sum()(new_a)
        else:
            sum = Reduce_Sum()(new_a)
        return sum    


class Avg_Square(klayers.Layer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        a, b = inputs
        shape = a.shape

        if len(shape) != 1:
            a = Reshape([-1, shape[-1]])(a)
            b = Reshape([-1, shape[-1]])(b)

        sum = Reduce_Sum()(a)
        n = Reduce_Sum()(b)

        squared_difference_1 = Square()(a)

        if len(shape) != 1:
            squared_difference_1 = Reshape(shape =[-1, shape[-1]])(squared_difference_1)

        # log_of_ten = Where()([less_than_zero, b, tf.constant(0.0)]) 
        # for a, b in zip(not_equal, log_of_ten)]

        sum = Reduce_Sum()(squared_difference_1)
        var_1 = Divide_no_nan()([sum, n])
        std = Squareroot()(var_1)

        return std
    
class ShuffledChannelsLayer(tf.keras.layers.Layer):
    def __init__(self, seed=0, **kwargs):
        super(ShuffledChannelsLayer, self).__init__(**kwargs)
        self.seed = seed

    def call(self, inputs):
        shuffled_channels = []

        # Iterate over each channel
        if len(inputs.shape) > 2:
            for i in range(inputs.shape[-1]):
                # Extract the channel
                channel = inputs[:, :, :, i]

                # Flatten, shuffle, and then reshape the channel
                flattened_channel = tf.reshape(channel, [-1])
                shuffled_flattened_channel = tf.random.shuffle(flattened_channel, seed=self.seed)
                reshaped_dims = inputs.shape[:-1].as_list()
                reshaped_dims.append(1)
                reshaped_channel = tf.reshape(shuffled_flattened_channel, reshaped_dims)

                # Append the shuffled channel to the list
                shuffled_channels.append(reshaped_channel)
        else:
            
                shuffled_flattened_channel = tf.random.shuffle(inputs)    
                reshaped_channel = tf.reshape(shuffled_flattened_channel, inputs.shape)

                # Append the shuffled channel to the list
                shuffled_channels.append(reshaped_channel)

        # Stack the shuffled channels to reconstruct the tensor
        shuffled_tensor = tf.concat(shuffled_channels, axis=-1)

        return shuffled_tensor    
    
class Reduce_std_sparse(klayers.Layer):
    def __init__(self, mean = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mean = mean


    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        a, b = inputs
        shape = a.shape

        if len(shape) != 1:
            a = Reshape([-1, shape[-1]])(a)
            b = Reshape([-1, shape[-1]])(b)

        if self.mean == None:
            # mean = Fill()([a])
            mean = Reduce_mean_sparse()(inputs)
        else:
            mean = self.mean

        n = Reduce_Sum()(b)

        if len(mean.shape) >= 1:
            mean = Reshape(shape =[-1, shape[-1]])(mean)

        squared_difference_1 = Squared_difference()([a, mean])
        squared_difference_1 = Multiply()([squared_difference_1, b])

        if len(shape) != 1:
            squared_difference_1 = Reshape(shape =[-1, shape[-1]])(squared_difference_1)


        sum = Reduce_Sum()(squared_difference_1)
        var_1 = Divide_no_nan()([sum, n])
        std = Squareroot()(var_1)


        return std

class Reduce_var_sparse(klayers.Layer):
    def __init__(self, mean = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mean = mean


    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        a, b = inputs
        shape = a.shape

        if len(shape) != 1:
            a = Reshape([-1, shape[-1]])(a)
            b = Reshape([-1, shape[-1]])(b)

        if self.mean == None:
            # mean = Fill()([a])
            mean = Reduce_mean_sparse()(inputs)
        else:
            mean = self.mean

        n = Reduce_Sum()(b)

        if len(mean.shape) >= 1:
            mean = Reshape(shape =[-1, shape[-1]])(mean)

        squared_difference_1 = Squared_difference()([a, mean])
        squared_difference_1 = Multiply()([squared_difference_1, b])

        if len(shape) != 1:
            squared_difference_1 = Reshape(shape =[-1, shape[-1]])(squared_difference_1)


        sum = Reduce_Sum()(squared_difference_1)
        var = Divide_no_nan()([sum, n])

        return var
   
class Reduce_var_not_sparse(klayers.Layer):
    def __init__(self, mean = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mean = mean


    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        a = inputs
        shape = a.shape

        if len(shape) != 1:
            a = Reshape([-1, shape[-1]])(a)

        if self.mean == None:
            # mean = Fill()([a])
            mean = Reduce_mean_sparse()(inputs)
        else:
            mean = self.mean

        if len(mean.shape) >= 1:
            mean = Reshape(shape =[-1, shape[-1]])(mean)

        squared_difference_1 = Squared_difference()([a, mean])

        if len(shape) != 1:
            squared_difference_1 = Reshape(shape =[-1, shape[-1]])(squared_difference_1)
            sum = Reduce_Sum()(squared_difference_1)
            var = Divide_no_nan()([sum, tf.cast(shape[-1], tf.float32)])
        else:
            sum = Reduce_Sum()(squared_difference_1)
            var = Divide_no_nan()([sum, tf.cast(shape[-1], tf.float32)])

        return var

    
class Reduce_std(klayers.Layer):
    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        flatten_a = tf.reshape(inputs[0], [-1, inputs[0].shape[-1]])
        return tf.math.reduce_std(flatten_a, 0)

class Divide(klayers.Layer):
    def call(self, inputs: list[Tensor], *_args, **_kwargs) -> Tensor:
        if len(inputs) != 2:
            raise ValueError("A `Divide` layer should be called on exactly 2 inputs")
        a, b = inputs
        return a / b


class Gradient(klayers.Layer):
    def call(self, inputs: list[Tensor], *_args, **_kwargs) -> Tensor:
        if len(inputs) != 2:
            raise ValueError("A `grad` layer should be called on exactly 2 inputs")
        a, b = inputs
        return tf.gradients(a, b)
class Add(klayers.Layer):
    def call(self, inputs: list[Tensor], *_args, **_kwargs) -> Tensor:
        if len(inputs) != 2:
            raise ValueError("A `add` layer should be called on exactly 2 inputs")
        a, b = inputs
        return tf.math.add(a, b)
    
    
class Add_One(klayers.Layer):
    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        return tf.math.add(inputs, 1)


class ProbabilityInDistribution(klayers.Layer):
    def __init__(self, *args, activations , **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # self.mean =  tf.math.reduce_mean(activations)
        # self.std = tf.math.reduce_std(activations)
        self.activations = activations[0]
        shape = self.activations.shape
        if (len(shape) > 1):
            self.mean = Reduce_mean()(activations)
            self.std = Reduce_std()(activations)
        else:
            self.mean = tf.math.reduce_mean(activations)
            self.std = tf.math.reduce_std(activations)

    def call(self, inputs: list[Tensor], *_args, **_kwargs) -> Tensor:
        dist = tfd.Normal(loc=self.mean, scale=self.std)
        return dist.prob(inputs)

        # return tf.math.add(inputs[0], 1)
# class Add_Etha(klayers.Layer):
#     def call(self, inputs: list[Tensor], *_args, **_kwargs) -> Tensor:
#         return tf.math.add(inputs[0], tf.)
# class Add_Nine(klayers.Layer):
#     def call(self, inputs: list[Tensor], *_args, **_kwargs) -> Tensor:
#         return tf.math.add(inputs[0], [9])



class Log_Of_Two(klayers.Layer):
    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        # self.eps = kbackend.epsilon()
        # eps_like_input = tf.fill(dims=inputs[0].shape, value=self.eps)
        # add_var_eps = tf.add(inputs[0], eps_like_input)
        return tf.experimental.numpy.log2(inputs)
    
class Exponentaial(klayers.Layer):
    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        # self.eps = kbackend.epsilon()
        # eps_like_input = tf.fill(dims=inputs[0].shape, value=self.eps)
        # add_var_eps = tf.add(inputs[0], eps_like_input)
        return tf.math.exp(inputs)
    

class Log_Of_Ten(klayers.Layer):
    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        return tf.experimental.numpy.log10(inputs)
        # return tf.experimental.numpy.log(inputs[0]) / tf.experimental.numpy.log(100)


class Round(klayers.Layer):
    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        return tf.round(inputs)


class Clip_to_Max_One(klayers.Layer):
    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        return tf.clip_by_value(inputs, clip_value_min=0.0000001, clip_value_max=100000.0)

class Power(klayers.Layer):
    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        return tf.experimental.numpy.power([10], inputs)
    
class Where(klayers.Layer):
    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        if len(inputs) != 3:
            raise ValueError("A `Where` layer should be called on exactly 2 inputs")
        a, b, c = inputs
        return tf.where(a, b, c)

class Fill(klayers.Layer):
    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        return tf.fill(inputs[0].shape, 0.0)
    
class Cast_To_Float(klayers.Layer):
    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        return tf.cast(inputs, tf.float32)

class Cast_To_Bool(klayers.Layer):
    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        return tf.cast(inputs, tf.bool)

class Divide_no_nan(klayers.Layer):
    def call(self, inputs: list[Tensor], *_args, **_kwargs) -> Tensor:
        if len(inputs) != 2:
            raise ValueError("A `Divide` layer should be called on exactly 2 inputs")
        a, b = inputs
        return tf.math.divide_no_nan(a, b)

class SafeDivide(klayers.Layer):
    def __init__(self, *args, factor: float = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if factor is None:
            factor = kbackend.epsilon()
        self._factor = factor

    def call(self, inputs: list[Tensor], *_args, **_kwargs) -> Tensor:
        if len(inputs) != 2:
            raise ValueError(
                "A `SafeDivide` layer should be called on exactly 2 inputs"
            )
        a, b = inputs
        return ibackend.safe_divide(a, b, factor=self._factor)
class Squareroot_the_variance(klayers.Layer):
    def __init__(self, *args, eps: float = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eps = eps

    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        eps_like_input = tf.fill(dims=inputs.shape, value=self.eps)
        # power_of_two = tf.pow(inputs, 2)
        add_var_eps = tf.add(inputs, eps_like_input)
        return tf.math.sqrt(add_var_eps)
class Split(klayers.Layer):
    def __init__(self, *args, num_or_size_splits: int = None, axis:int = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if num_or_size_splits is None:
            num_or_size_splits = 2
        self.num_or_size_splits = num_or_size_splits

        if axis is None:
            axis = 0
        self.axis = axis

    def call(self, inputs: list[Tensor], *_args, **_kwargs) -> Tensor:
        return tf.split(inputs[0], num_or_size_splits=self.num_or_size_splits, axis=self.axis)
class Concat(klayers.Layer):
    def __init__(self, *args, axis: int = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if axis is None:
            axis = 0
        self.axis = axis

    def call(self, inputs: list[Tensor], *_args, **_kwargs) -> Tensor:
        return tf.concat(inputs, self.axis)

class Substract(klayers.Layer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def call(self, inputs: list[Tensor], *_args, **_kwargs) -> Tensor:
        if len(inputs) != 2:
            raise ValueError(
                "A `subtract` layer should be called on exactly 2 inputs"
            )
        a, b = inputs
        return tf.math.subtract(a, b)


class Multiply(klayers.Layer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def call(self, inputs: list[Tensor], *_args, **_kwargs) -> Tensor:
        if len(inputs) != 2:
            raise ValueError(
                "A `multiply` layer should be called on exactly 2 inputs"
            )
        a, b = inputs
        return tf.math.multiply(a, b)
###############################################################################
class Repeat(klayers.Layer):
    """Repeats the input n times. Similar to Keras' `RepeatVector`,
    except that it works on any Tensor.

    Input shape: 2D tensor of shape `(num_samples, features)`.
    Output shape: 3D tensor of shape `(num_samples, n, features)`.

    Args:
        n: Integer, repetition factor.
    """

    def __init__(self, n, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n
        if not isinstance(n, int):
            raise TypeError(f"Expected an integer value for `n`, got {type(n)}.")

    def call(self, inputs, *_args, **_kwargs):
        dims = inputs.shape.rank  # number of axes in Tensor
        assert dims >= 2
        inputs = tf.expand_dims(inputs, 1)

        # Construct array [1, n, 1, ..., 1] for tf.tile
        multiples = [1] * dims
        multiples.insert(1, self.n)
        return tf.tile(inputs, tf.constant(multiples))


class ReduceMean(klayers.Layer):
    """Reduce input augmented along `axis=1` by taking the mean."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        return tf.math.reduce_mean(inputs, axis=1, keepdims=False)

class Expand_dim(klayers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        return  tf.expand_dims(inputs, axis=0)
    
class Reshape(klayers.Layer):
    """Layer that reshapes tensor to the shape specified on init."""

    def __init__(self, shape: ShapeTuple, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._shape = shape

    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        return tf.reshape(inputs, self._shape)


class AugmentationToBatchAxis(klayers.Layer):
    """Move augmentation from axis=1 to batch axis."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        input_shape = ibackend.shape(inputs)
        output_shape = [-1] + input_shape[2:]  # type: ignore
        return tf.reshape(inputs, output_shape)


class AugmentationFromBatchAxis(klayers.Layer):
    """Move augmentation from batch axis to axis=1.

    Args:
        n: Factor of augmentation.
    """

    def __init__(self, n, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n
        if not isinstance(n, int):
            raise TypeError(f"Expected an integer value for `n`, got {type(n)}.")

    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        input_shape = ibackend.shape(inputs)
        output_shape = [-1, self.n] + input_shape[1:]
        return tf.reshape(inputs, output_shape)


class MultiplyWithLinspace(klayers.Layer):
    def __init__(self, start, end, *args, n=1, axis=-1, **kwargs):
        super().__init__(*args, **kwargs)
        self._start = start
        self._end = end
        self._n = n
        self._axis = axis

    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        linspace = self._start + (self._end - self._start) * (
            kbackend.arange(self._n, dtype=kbackend.floatx()) / self._n
        )

        # Make broadcastable.
        shape = np.ones(len(kbackend.int_shape(inputs)))
        shape[self._axis] = self._n
        linspace = kbackend.reshape(linspace, shape)
        return inputs * linspace


class AddGaussianNoise(klayers.Layer):
    "Add Gaussian noise to Tensor. Also applies to test phase."

    def __init__(self, *args, mean: float = 0.0, stddev: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean = mean
        self.stddev = stddev

    def call(self, inputs: Tensor, *_args, seed=None, **_kwargs) -> Tensor:
        noise = tf.random.normal(
            shape=tf.shape(inputs),
            mean=self.mean,
            stddev=self.stddev,
            dtype=inputs.dtype,
            seed=seed,
        )
        return tf.add(inputs, noise)


class ExtractConv2DPatches(klayers.Layer):
    def __init__(self, kernel_shape, depth, strides, rates, padding, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kernel_shape = kernel_shape
        self._depth = depth
        self._strides = strides
        self._rates = rates
        self._padding = padding

    def call(self, inputs, *_args, **_kwargs):
        return ibackend.extract_conv2d_patches(
            inputs, self._kernel_shape, self._strides, self._rates, self._padding
        )


class RunningMeans(klayers.Layer):
    """Layer used to keep track of a running mean."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stateful = True

    def build(self, input_shape: Sequence[ShapeTuple]) -> None:
        means_shape, counts_shape = input_shape

        self.means = self.add_weight(
            shape=means_shape, initializer="zeros", name="means", trainable=False
        )
        self.counts = self.add_weight(
            shape=counts_shape, initializer="zeros", name="counts", trainable=False
        )
        self.built = True

    def call(self, inputs: list[Tensor], *_args, **_kwargs) -> list[Tensor]:
        if len(inputs) != 2:
            raise ValueError(
                "A `RunningMeans` layer should be called on exactly 2 inputs"
            )
        means, counts = inputs
        new_counts = counts + self.counts

        # If new_means are not used for the model output,
        # the following part of the code will be executed after
        # self.counts is updated, therefore we cannot use it
        # hereafter.
        factor_new = ibackend.safe_divide(counts, new_counts, factor=1)
        factor_old = kbackend.ones_like(factor_new) - factor_new
        new_means = self.means * factor_old + means * factor_new

        # Update state.
        self.add_update(
            [
                kbackend.update(self.means, new_means),
                kbackend.update(self.counts, new_counts),
            ]
        )

        return [new_means, new_counts]


class Broadcast(klayers.Layer):
    def call(self, inputs: list[Tensor], *_args, **_kwargs) -> Tensor:
        if len(inputs) != 2:
            raise ValueError("A `Broadcast` layer should be called on exactly 2 inputs")
        target_shapped, x = inputs
        return target_shapped * 0 + x


class MaxNeuronSelection(klayers.Layer):
    """Applied to the last layer of a model, this reduces the output
    to the max neuron activation."""

    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        return tf.math.reduce_max(
            inputs, axis=-1, keepdims=False
        )  # max along batch axis


class MaxNeuronIndex(klayers.Layer):
    """Applied to the last layer of a model, this reduces the output
    to the index of the max activated neuron."""

    def call(self, inputs: Tensor, *_args, **_kwargs) -> Tensor:
        return tf.math.argmax(inputs, axis=-1)  # max along batch axis


class NeuronSelection(klayers.Layer):
    """Applied to the last layer of a model, this selects output neurons at given indices
    by wrapping `tf.gather`."""

    def call(self, inputs: list[Tensor], *_args, **_kwargs):
        if len(inputs) != 2:
            raise ValueError(
                "A `NeuronSelection` layer should be called on exactly 2 inputs"
            )
        X, index = inputs
        index_shape = ibackend.shape(index)
        if len(index_shape) != 2 or index_shape[1] != 2:
            raise ValueError(
                "Layer `NeuronSelection` expects index of shape (batch_size, 2),",
                f"got {index_shape}.",
            )
        return tf.gather_nd(X, index)



class MaxPool_forward(klayers.Layer):

    "Returns bn input with weighted beta"
    " expects inputs+[output,]."

    def __init__(self, n_inputs, **kwargs):
        self.n_inputs = n_inputs
        super(MaxPool_forward, self).__init__(**kwargs)

    def call(self, x):
        assert isinstance(x, (list, tuple))
        Xs, tmp_Ys = x[:self.n_inputs], x[self.n_inputs:]
        Ys, tmp_Ys = tmp_Ys[:1], tmp_Ys[1:]
        masked_Xs, reverse_state = tmp_Ys[:1], tmp_Ys[1:]

        reverse_state = reverse_state[0]

        grad = tf.gradients(Ys, Xs)

        y = tf.constant(0.0)

        mask_the_zeros = [tf.math.not_equal(x, y) for x in grad]

        Xs_prime = [tf.math.multiply(masked_Xs, tf.cast(mask_the_zeros, tf.float32))[0]]

        absolute_Ys = ibackend.apply(reverse_state["layer"], tf.math.abs(Xs_prime))[0]
        non_Ys = ibackend.apply(reverse_state["layer"], Xs_prime)[0]
        new_Ys = tf.where(tf.equal(non_Ys, 0), -absolute_Ys, non_Ys)

        # ratio = [tf.math.divide_no_nan(new_Ys[0], Ys[0])]

        return new_Ys


    def compute_output_shape(self, input_shapes):
        return input_shapes[:self.n_inputs]
