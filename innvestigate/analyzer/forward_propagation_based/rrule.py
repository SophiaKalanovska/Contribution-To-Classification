from __future__ import annotations
# import tensorflow.backend.backend as kbackend
# import tensorflow.backend.layers as klayers
#
# import innvestigate.backend as ibackend
# import innvestigate.backend.graph as igraph
# import innvestigate.layers as ilayers
import time
import random
import tensorflow.keras as keras
import tensorflow as tf
import innvestigate.backend as ibackend
import tensorflow.keras.backend as kbackend
import tensorflow.keras.layers as klayers
from innvestigate.backend.types import Layer, OptionalList, Tensor

###############################################################################
import numpy as np

from innvestigate import layers as ilayers
import innvestigate.backend as kutils
import innvestigate.backend.graph as igraph
from . import utils as rutils

# TODO: differentiate between LRP and DTD rules?
# DTD rules are special cases of LRP rules with additional assumptions
__all__ = [
    # dedicated treatment for special layers
    # general rules
    "ZRule",
    "EpsilonRule",
    "WSquareRule",
    "FlatRule",
    "AlphaBetaRule",
    "AlphaBetaIgnoreBiasRule",
    "Alpha2Beta1Rule",
    "Alpha2Beta1IgnoreBiasRule",
    "Alpha1Beta0Rule",
    "Alpha1Beta0IgnoreBiasRule",
    "AlphaBetaXRule",
    "AlphaBetaX1000Rule",
    "AlphaBetaX1010Rule",
    "AlphaBetaX1001Rule",
    "AlphaBetaX2m100Rule",
    "ZPlusRule",
    "ZPlusFastRule",
    "BoundedRule",
]


class ZRule(igraph.ReverseMappingBase):
    """
    Basic LRP decomposition rule (for layers with weight kernels),
    which considers the bias a constant input neuron.
    """

    def __init__(self, layer: Layer, _state, bias: bool = True) -> None:
        self._layer_wo_act = igraph.copy_layer_wo_activation(
            layer, keep_bias=bias, name_template="reversed_kernel_%s"
        )

    def apply(
        self,
        Xs: list[Tensor],
        _Ys: list[Tensor],
        Rs: list[Tensor],
        _reverse_state,
    ) -> list[Tensor]:

        # Get activations.
        Zs = ibackend.apply(self._layer_wo_act, Xs)
        # Divide incoming relevance by the activations.
        tmp = [ibackend.safe_divide(a, b) for a, b in zip(Rs, Zs)]
        # Propagate the relevance to input neurons
        # using the gradient.
        grads = ibackend.gradients(Xs, Zs, tmp)
        # Re-weight relevance with the input values.
        return [klayers.Multiply()([a, b]) for a, b in zip(Xs, grads)]


class EpsilonRule(igraph.ReverseMappingBase):
    """
    Similar to ZRule.
    The only difference is the addition of a numerical stabilizer term
    epsilon to the decomposition function's denominator.
    the sign of epsilon depends on the sign of the output activation
    0 is considered to be positive, ie sign(0) = 1
    """

    def __init__(self, layer: Layer, _state, epsilon=1e-7, bias: bool = True):
        self._epsilon = rutils.assert_lrp_epsilon_param(epsilon, self)
        self._layer_wo_act = igraph.copy_layer_wo_activation(
            layer, keep_bias=bias, name_template="reversed_kernel_%s"
        )

    def apply(
        self,
        Xs: list[Tensor],
        _Ys: list[Tensor],
        Rs: list[Tensor],
        _reverse_state: dict,
    ):
        # The epsilon rule aligns epsilon with the (extended) sign:
        # 0 is considered to be positive
        prepare_div = klayers.Lambda(
            lambda x: x
            + (kbackend.cast(kbackend.greater_equal(x, 0), kbackend.floatx()) * 2 - 1)
            * self._epsilon
        )

        # Get activations.
        Zs = ibackend.apply(self._layer_wo_act, Xs)

        # Divide incoming relevance by the activations.
        tmp = [a / prepare_div(b) for a, b in zip(Rs, Zs)]
        # Propagate the relevance to input neurons
        # using the gradient.
        grads = ibackend.gradients(Xs, Zs, tmp)
        # Re-weight relevance with the input values.
        return [klayers.Multiply()([a, b]) for a, b in zip(Xs, grads)]


class WSquareRule(igraph.ReverseMappingBase):
    """W**2 rule from Deep Taylor Decomposition"""

    def __init__(self, layer: Layer, _state, copy_weights=False) -> None:
        # W-square rule works with squared weights and no biases.
        if copy_weights:
            weights = layer.get_weights()
        else:
            weights = layer.weights
        if getattr(layer, "use_bias", False):
            weights = weights[:-1]
        weights = [x**2 for x in weights]

        self._layer_wo_act_b = igraph.copy_layer_wo_activation(
            layer, keep_bias=False, weights=weights, name_template="reversed_kernel_%s"
        )

    def apply(
        self,
        Xs: list[Tensor],
        Ys: list[Tensor],
        Rs: list[Tensor],
        _reverse_state: dict,
    ) -> list[Tensor]:
        # Create dummy forward path to take the derivative below.
        Ys = ibackend.apply(self._layer_wo_act_b, Xs)

        # Compute the sum of the weights.
        ones = ilayers.OnesLike()(Xs)
        Zs = [self._layer_wo_act_b(X) for X in ones]
        # Weight the incoming relevance.
        tmp = [ilayers.SafeDivide()([a, b]) for a, b in zip(Rs, Zs)]
        # Redistribute the relevances along the gradient.
        grads = ibackend.gradients(Xs, Ys, tmp)
        return grads


class FlatRule(WSquareRule):
    """Same as W**2 rule but sets all weights to ones."""

    def __init__(self, layer: Layer, _state, copy_weights: bool = False) -> None:
        # The flat rule works with weights equal to one and
        # no biases.
        if copy_weights:
            weights = layer.get_weights()
            if getattr(layer, "use_bias", False):
                weights = weights[:-1]
            weights = [np.ones_like(x) for x in weights]
        else:
            weights = layer.weights
            if getattr(layer, "use_bias", False):
                weights = weights[:-1]
            weights = [kbackend.ones_like(x) for x in weights]

        self._layer_wo_act_b = igraph.copy_layer_wo_activation(
            layer, keep_bias=False, weights=weights, name_template="reversed_kernel_%s"
        )





class AlphaBetaRule(igraph.ReverseMappingBase):
    """
    This decomposition rule handles the positive forward
    activations (x*w > 0) and negative forward activations
    (w * x < 0) independently, reducing the risk of zero
    divisions considerably. In fact, the only case where
    divisions by zero can happen is if there are either
    no positive or no negative parts to the activation
    at all.
    Corresponding parameterization of this rule implement
    methods such as Excitation Backpropagation with
    alpha=1, beta=0
    s.t.
    alpha - beta = 1 (after current param. scheme.)
    and
    alpha > 1
    beta > 0
    """

    def __init__(
        self,
        layer: Layer,
        _state,
        alpha=None,
        beta=None,
        bias: bool = True,
        copy_weights=False,
    ) -> None:
        alpha, beta = rutils.assert_infer_reveal_alpha_beta_param(alpha, beta, self)
        self._alpha = alpha
        self._beta = beta
        self.bias = None

        self.weights = layer.weights
        if layer.use_bias:
            self.bias = self.weights[-1]
            self.weights_no_bias = self.weights[:-1]
        else:
            self.weights_no_bias = self.weights

        self._layer_no_act = igraph.copy_layer_wo_activation(
            layer,
            keep_bias=True,
            weights=self.weights_no_bias,
            name_template="reversed_kernel_positive_%s"+ str(random.randint(0, 10000000)),
        )

        self._layer = igraph.copy_layer(
            layer,
            keep_bias=True,
            weights=self.weights_no_bias,
            name_template="reversed_kernel_positive_%s" + str(random.randint(0, 10000000)),
        )

    def apply(
        self,
        Xs: list[Tensor],
        _Ys: list[Tensor],
        Rs: list[Tensor],
        _reverse_state: dict,
    ):

        list_con = ilayers.Split(num_or_size_splits=_reverse_state["masks_size"], axis=0)(Rs)

        activator_relevances = ilayers.ApplyLayerToList([self._layer_no_act])(list_con)

        contribution_no_act_no_bias = [ilayers.Concat(axis=0)(activator_relevances)]

        mask_the_zeros = [ilayers.Not_Equal_Zero()(a) for a in activator_relevances]

        casted_mask_the_zeros_list = [ilayers.Cast_To_Float()(a) for a in mask_the_zeros]

        mask_the_non_zeros = ilayers.Equal_Zero()(activator_relevances[-1])

        mask_the_non_zeros = ilayers.Cast_To_Float()(mask_the_non_zeros)

        casted_mask_the_zeros_list = [ilayers.Add()([a, mask_the_non_zeros]) for a in casted_mask_the_zeros_list]

        scale_log = None

        if self.bias is not None:

            scaled_tensor = [ilayers.Multiply()([a, tf.constant(10.0)]) for a in activator_relevances]
            # Round to nearest whole number
            rounded_tensor = [ilayers.Ceil()(a) for a in scaled_tensor]
            # Scale back down by 10^2
            activator_relevances_prime = [ilayers.Divide_no_nan()([a, tf.constant(10.0)]) for a in rounded_tensor]
        

            net_plus_bias = ilayers.Add()([activator_relevances_prime[-1], self.bias])
            net_plus_bias_mean = ilayers.Reduce_mean()(net_plus_bias)
   
            

            ratio_net_plus_bias = ilayers.SafeDivide()([net_plus_bias, net_plus_bias_mean])
            absolut_net = ilayers.Absolut()([ratio_net_plus_bias])
            scaled_tensor = ilayers.Multiply()([absolut_net, tf.constant(1000.0)]) 
            # Round to nearest whole number
            rounded_tensor = ilayers.Ceil()(scaled_tensor) 
            absolut_net =  ilayers.Divide_no_nan()([rounded_tensor, tf.constant(1000.0)])
            # Scale back down by 10^2
            # net_plus_bias = ilayers.Add()([activator_relevances_prime[-1], self.bias])

            # absolut_net = ilayers.Absolut()([net_plus_bias])




            ratio = [ilayers.Divide_no_nan()([a, activator_relevances_prime[-1]]) for a in activator_relevances_prime]
            absolut_bias = [ilayers.Absolut()([a]) for a in ratio]

            scaled_tensor = [ilayers.Multiply()([a, tf.constant(1000.0)]) for a in absolut_bias]
            # Round to nearest whole number
            rounded_tensor = [ilayers.Ceil()(a) for a in scaled_tensor]
            # Scale back down by 10^2
            absolut_bias = [ilayers.Divide_no_nan()([a, tf.constant(1000.0)]) for a in rounded_tensor]

            # weighted_bias = [ilayers.Multiply()([self.bias, a]) for a in absolut_bias]


        

# ========================================================================================================

            log_of_ten_Ys = ilayers.Log_Of_Ten()(absolut_net)

            not_equal_Ys = ilayers.Not_Equal_Zero()(absolut_net)        

            log_of_ten_net = ilayers.Where()([not_equal_Ys, log_of_ten_Ys, tf.constant(0.0)])[0]

            # log_of_ten_net = ilayers.Floor()(log_of_ten_net)

            # log_of_ten_net_max = ilayers.Reduce_max()(log_of_ten_net)


            # min_abs_net = ilayers.Reduce_min()(log_of_ten_net)

            std_from_the_min_value = ilayers.Reduce_std_sparse()([log_of_ten_net, casted_mask_the_zeros_list[-1]])

 # ========================================================================================================

            log_of_ten = [ilayers.Log_Of_Ten()(a) for a in absolut_bias]

            not_equal = [ilayers.Not_Equal_Zero()([a]) for a in absolut_bias]

            log_of_ten = [ilayers.Where()([a, b, tf.constant(0.0)]) for a, b in zip(not_equal, log_of_ten)]

            # min_abs_bias = [ilayers.Reduce_min()(a) for a in log_of_ten]

            squeezed_log = [ilayers.Squeeze()(a) for a in log_of_ten]

            # squeezed_log = [ilayers.Floor()(a) for a in squeezed_log]

            # log_of_ten_max = [ilayers.Reduce_max()(a) for a in squeezed_log]
# 


            std_bias =  [ilayers.Reduce_std_sparse()([a, b]) for a, b in zip(squeezed_log, casted_mask_the_zeros_list)]
            

 # ========================================================================================================

            # scaling_factor = [ilayers.Substract()([log_of_ten_net_max, a]) for a in log_of_ten_max]

            mean_std_more_than_one = [ilayers.MoreThan()([a, std_from_the_min_value]) for a  in std_bias]
            # mean_std_more_than_one = [ilayers.MoreThan()([a, b]) for a,b   in zip(std_bias, std_Ys)]

            mean_bias = [ilayers.Reduce_mean_sparse()([a, b]) for a, b in zip(squeezed_log, casted_mask_the_zeros_list)]
            # mean_act = ilayers.Reduce_mean_sparse()([log_of_ten_net, casted_mask_the_zeros_list[-1]])

            ratio_norm = [ilayers.Substract()([a, b]) for a, b in zip(squeezed_log, mean_bias)]

            ratio_norm = [ilayers.Divide_no_nan()([a, b]) for a, b in zip(ratio_norm, std_bias)]
# 
            ratio_norm = [ilayers.Multiply()([a, std_from_the_min_value]) for a in ratio_norm]

            ratio_norm = [ilayers.Add()([a, b]) for a, b in zip(ratio_norm, mean_bias)]


            # ratio_norm = [ilayers.Multiply()([a, b]) for a, b in zip(ratio_norm, std_Ys)]

            new_bias = [ilayers.Where()([a, b, c]) for a, b, c in zip(mean_std_more_than_one, ratio_norm, squeezed_log)]


 # ========================================================================================================


            power = [ilayers.Power()(a) for a in new_bias]

            scaler = [ilayers.Where()([a, b, tf.constant(0.0)]) for a, b in zip(not_equal, power)]

            # scaler = [ilayers.Expand_dim()(a) for a in scaler]

            # neg_bias = [ilayers.LessThanZero()(a) for a in weighted_bias]

            # scaler = [ilayers.Where()([a, -b, b]) for a, b in zip(neg_bias, scaler)]

            scaler = [ilayers.Multiply()([self.bias, a]) for a in scaler]

            

            net_value = [ilayers.Add()([a, b]) for a, b in zip(activator_relevances, scaler)]


            net_value = [ilayers.Concat(axis=0)(net_value)]
            # scale_log = [ilayers.Concat(axis=0)(relevance)]

            # net_value = contribution_no_act_no_bias
        else:
            net_value = contribution_no_act_no_bias

        # if self._layer.get_config()["activation"] is not None:
        #     if not(self._layer.get_config()["activation"] == "relu"):
        #         layer = tf.keras.layers.Activation(self._layer.get_config()["activation"])
        #         net_value = ibackend.apply(layer, net_value)
        #     else:   
        #         net_value = net_value

        mask_the_zeros = ilayers.Not_Equal_Zero()(_Ys)

        casted_mask_the_zeros = [ilayers.Cast_To_Float()(mask_the_zeros)]

        net_value = [ilayers.Multiply()([a, b]) for a, b in zip(net_value, casted_mask_the_zeros)]

        if scale_log == None:
            scale_log = [ilayers.Divide_no_nan()([a, _Ys[-1]]) for a in net_value]
        return net_value, scale_log


class AlphaBetaIgnoreBiasRule(AlphaBetaRule):
    """Same as AlphaBetaRule but ignores biases."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, bias=False, **kwargs)


class Alpha2Beta1Rule(AlphaBetaRule):
    """AlphaBetaRule with alpha=2, beta=1"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, alpha=2, beta=1, bias=True, **kwargs)


class Alpha2Beta1IgnoreBiasRule(AlphaBetaRule):
    """AlphaBetaRule with alpha=2, beta=1 and ignores biases"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, alpha=2, beta=1, bias=False, **kwargs)


class Alpha1Beta0Rule(AlphaBetaRule):
    """AlphaBetaRule with alpha=1, beta=0"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, alpha=1, beta=0, bias=True, **kwargs)


class Alpha1Beta0IgnoreBiasRule(AlphaBetaRule):
    """AlphaBetaRule with alpha=1, beta=0 and ignores biases"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, alpha=1, beta=0, bias=False, **kwargs)


class AlphaBetaXRule(igraph.ReverseMappingBase):
    """
    AlphaBeta advanced as proposed by Alexander Binder.
    """

    def __init__(
        self,
        layer: Layer,
        _state,
        alpha: tuple[float, float] = (0.5, 0.5),
        beta: tuple[float, float] = (0.5, 0.5),
        bias: bool = True,
        copy_weights: bool = False,
    ) -> None:
        self._alpha = alpha
        self._beta = beta

        # prepare positive and negative weights for computing positive
        # and negative preactivations z in apply_accordingly.
        if copy_weights:
            weights = layer.get_weights()
            if not bias and getattr(layer, "use_bias", False):
                weights = weights[:-1]
            positive_weights = [x * (x > 0) for x in weights]
            negative_weights = [x * (x < 0) for x in weights]
        else:
            weights = layer.weights
            if not bias and getattr(layer, "use_bias", False):
                weights = weights[:-1]
            positive_weights = [x * ibackend.cast_to_floatx(x > 0) for x in weights]
            negative_weights = [x * ibackend.cast_to_floatx(x < 0) for x in weights]

        self._layer_wo_act_positive = igraph.copy_layer_wo_activation(
            layer,
            keep_bias=bias,
            weights=positive_weights,
            name_template="reversed_kernel_positive_%s",
        )
        self._layer_wo_act_negative = igraph.copy_layer_wo_activation(
            layer,
            keep_bias=bias,
            weights=negative_weights,
            name_template="reversed_kernel_negative_%s",
        )

    def apply(
        self,
        Xs: list[Tensor],
        _Ys: list[Tensor],
        Rs: list[Tensor],
        _reverse_state: dict,
    ):
        # this method is correct, but wasteful
        times_alpha0 = klayers.Lambda(lambda x: x * self._alpha[0])
        # times_alpha1 = klayers.Lambda(lambda x: x * self._alpha[1]) # unused
        times_beta0 = klayers.Lambda(lambda x: x * self._beta[0])
        times_beta1 = klayers.Lambda(lambda x: x * self._beta[1])
        keep_positives = klayers.Lambda(
            lambda x: x * kbackend.cast(kbackend.greater(x, 0), kbackend.floatx())
        )
        keep_negatives = klayers.Lambda(
            lambda x: x * kbackend.cast(kbackend.less(x, 0), kbackend.floatx())
        )

        def fn_tmp(layer: Layer, Xs: OptionalList[Tensor]):
            Zs = ibackend.apply(layer, Xs)
            # Divide incoming relevance by the activations.
            tmp = [ibackend.safe_divide(a, b) for a, b in zip(Rs, Zs)]
            # Propagate the relevance to the input neurons
            # using the gradient
            grads = ibackend.gradients(Xs, Zs, tmp)
            # Re-weight relevance with the input values.
            return [klayers.Multiply()([a, b]) for a, b in zip(Xs, grads)]

        # Distinguish postive and negative inputs.
        Xs_pos = ibackend.apply(keep_positives, Xs)
        Xs_neg = ibackend.apply(keep_negatives, Xs)

        # xpos*wpos
        r_pp = fn_tmp(self._layer_wo_act_positive, Xs_pos)
        # xneg*wneg
        r_nn = fn_tmp(self._layer_wo_act_negative, Xs_neg)
        # a0 * r_pp + a1 * r_nn
        r_pos = [
            klayers.Add()([times_alpha0(pp), times_beta1(nn)])
            for pp, nn in zip(r_pp, r_nn)
        ]

        # xpos*wneg
        r_pn = fn_tmp(self._layer_wo_act_negative, Xs_pos)
        # xneg*wpos
        r_np = fn_tmp(self._layer_wo_act_positive, Xs_neg)
        # b0 * r_pn + b1 * r_np
        r_neg = [
            klayers.Add()([times_beta0(pn), times_beta1(np)])
            for pn, np in zip(r_pn, r_np)
        ]

        return [klayers.Subtract()([a, b]) for a, b in zip(r_pos, r_neg)]


class AlphaBetaX1000Rule(AlphaBetaXRule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, alpha=(1, 0), beta=(0, 0), bias=True, **kwargs)


class AlphaBetaX1010Rule(AlphaBetaXRule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, alpha=(1, 0), beta=(0, -1), bias=True, **kwargs)


class AlphaBetaX1001Rule(AlphaBetaXRule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, alpha=(1, 1), beta=(0, 0), bias=True, **kwargs)


class AlphaBetaX2m100Rule(AlphaBetaXRule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, alpha=(2, 0), beta=(1, 0), bias=True, **kwargs)


class BoundedRule(igraph.ReverseMappingBase):
    """Z_B rule from the Deep Taylor Decomposition"""

    # TODO: this only works for relu networks, needs to be extended.
    # TODO: check
    def __init__(
        self, layer: Layer, _state, low=-1, high=1, copy_weights: bool = False
    ) -> None:
        self._low = low
        self._high = high

        # This rule works with three variants of the layer, all without biases.
        # One is the original form and two with only the positive or
        # negative weights.
        if copy_weights:
            weights = layer.get_weights()
            if getattr(layer, "use_bias", False):
                weights = weights[:-1]
            positive_weights = [x * (x > 0) for x in weights]
            negative_weights = [x * (x < 0) for x in weights]
        else:
            weights = layer.weights
            if getattr(layer, "use_bias", False):
                weights = weights[:-1]
            positive_weights = [x * ibackend.cast_to_floatx(x > 0) for x in weights]
            negative_weights = [x * ibackend.cast_to_floatx(x < 0) for x in weights]

        self._layer_wo_act = igraph.copy_layer_wo_activation(
            layer, keep_bias=False, name_template="reversed_kernel_%s"
        )
        self._layer_wo_act_positive = igraph.copy_layer_wo_activation(
            layer,
            keep_bias=False,
            weights=positive_weights,
            name_template="reversed_kernel_positive_%s",
        )
        self._layer_wo_act_negative = igraph.copy_layer_wo_activation(
            layer,
            keep_bias=False,
            weights=negative_weights,
            name_template="reversed_kernel_negative_%s",
        )

    # TODO: clean up this implementation and add more documentation
    def apply(self, Xs, _Ys, Rs, reverse_state: dict):
        to_low = klayers.Lambda(lambda x: x * 0 + self._low)
        to_high = klayers.Lambda(lambda x: x * 0 + self._high)

        low = [to_low(x) for x in Xs]
        high = [to_high(x) for x in Xs]

        # Get values for the division.
        A = ibackend.apply(self._layer_wo_act, Xs)
        B = ibackend.apply(self._layer_wo_act_positive, low)
        C = ibackend.apply(self._layer_wo_act_negative, high)
        Zs = [
            klayers.Subtract()([a, klayers.Add()([b, c])]) for a, b, c in zip(A, B, C)
        ]

        # Divide relevances with the value.
        tmp = [ibackend.safe_divide(a, b) for a, b in zip(Rs, Zs)]
        # Distribute along the gradient.
        grads_a = ibackend.gradients(Xs, A, tmp)
        grads_b = ibackend.gradients(low, B, tmp)
        grads_c = ibackend.gradients(high, C, tmp)

        tmp_a = [klayers.Multiply()([a, b]) for a, b in zip(Xs, grads_a)]
        tmp_b = [klayers.Multiply()([a, b]) for a, b in zip(low, grads_b)]
        tmp_c = [klayers.Multiply()([a, b]) for a, b in zip(high, grads_c)]

        ret = [
            klayers.Subtract()([a, klayers.Add()([b, c])])
            for a, b, c in zip(tmp_a, tmp_b, tmp_c)
        ]

        return ret


class ZPlusRule(Alpha1Beta0IgnoreBiasRule):
    """
    The ZPlus rule is a special case of the AlphaBetaRule
    for alpha=1, beta=0, which assumes inputs x >= 0
    and ignores the bias.
    CAUTION! Results differ from Alpha=1, Beta=0
    if inputs are not strictly >= 0
    """

    # TODO: assert that layer inputs are always >= 0


class ZPlusFastRule(igraph.ReverseMappingBase):
    """
    The ZPlus rule is a special case of the AlphaBetaRule
    for alpha=1, beta=0 and assumes inputs x >= 0.
    """

    def __init__(self, layer: Layer, _state, copy_weights=False):
        # The z-plus rule only works with positive weights and
        # no biases.
        # TODO: assert that layer inputs are always >= 0
        if copy_weights:
            weights = layer.get_weights()
            if getattr(layer, "use_bias", False):
                weights = weights[:-1]
            weights = [x * (x > 0) for x in weights]
        else:
            weights = layer.weights
            if getattr(layer, "use_bias", False):
                weights = weights[:-1]
            weights = [x * ibackend.cast_to_floatx(x > 0) for x in weights]

        self._layer_wo_act_b_positive = igraph.copy_layer_wo_activation(
            layer,
            keep_bias=False,
            weights=weights,
            name_template="reversed_kernel_positive_%s",
        )

    def apply(self, Xs, _Ys, Rs, reverse_state: dict):
        # TODO: assert all inputs are positive, instead of only keeping the positives.
        # keep_positives = klayers.Lambda(
        #     lambda x: x * kbackend.cast(kbackend.greater(x, 0), kbackend.floatx())
        # )
        # Xs = ibackend.apply(keep_positives, Xs)

        # Get activations.
        Zs = ibackend.apply(self._layer_wo_act_b_positive, Xs)
        # Divide incoming relevance by the activations.
        tmp = [ibackend.safe_divide(a, b) for a, b in zip(Rs, Zs)]
        # Propagate the relevance to input neurons
        # using the gradient.
        grads = ibackend.gradients(Xs, Zs, tmp)
        # Re-weight relevance with the input values.
        return [klayers.Multiply()([a, b]) for a, b in zip(Xs, grads)]
