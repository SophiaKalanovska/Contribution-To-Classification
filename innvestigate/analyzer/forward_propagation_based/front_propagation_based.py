from __future__ import annotations

import inspect
import random
import time
import keras.layers
import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf
import tensorflow.keras.backend as kbackend
import tensorflow.keras.layers as klayers

# from backend import graph as kgraph
import innvestigate.analyzer.forward_propagation_based.rrule as rrule
import innvestigate.analyzer.forward_propagation_based.utils as rutils
import innvestigate.backend as ibackend
import innvestigate.backend.checks as ichecks
import innvestigate.backend.graph as igraph
from innvestigate import layers as ilayers
from innvestigate.analyzer.reverse_base import ReverseAnalyzerBase
from innvestigate.backend.types import Layer, LayerCheck, OptionalList, Tensor



__all__ = [
    "REVEAL",
    "REVEALAlphaBeta",
    "REVEALAlpha2Beta1",
]

###############################################################################
BASELINE_LRPZ_LAYERS = (
    klayers.InputLayer,
    klayers.Conv1D,
    klayers.Conv2D,
    klayers.Conv2DTranspose,
    klayers.Conv3D,
    klayers.Conv3DTranspose,
    klayers.Cropping1D,
    klayers.Cropping2D,
    klayers.Cropping3D,
    klayers.SeparableConv1D,
    klayers.SeparableConv2D,
    klayers.UpSampling1D,
    klayers.UpSampling2D,
    klayers.UpSampling3D,
    klayers.ZeroPadding1D,
    klayers.ZeroPadding2D,
    klayers.ZeroPadding3D,
    klayers.Activation,
    klayers.ActivityRegularization,
    klayers.Dense,
    klayers.Dropout,
    klayers.Flatten,
    klayers.Lambda,
    klayers.Masking,
    klayers.Permute,
    klayers.RepeatVector,
    klayers.Reshape,
    klayers.SpatialDropout1D,
    klayers.SpatialDropout2D,
    klayers.SpatialDropout3D,
    klayers.LocallyConnected1D,
    klayers.LocallyConnected2D,
    klayers.Add,
    klayers.Concatenate,
    klayers.Dot,
    klayers.Maximum,
    klayers.Minimum,
    klayers.Subtract,
    klayers.AlphaDropout,
    klayers.GaussianDropout,
    klayers.GaussianNoise,
    klayers.BatchNormalization,
    klayers.GlobalMaxPooling1D,
    klayers.GlobalMaxPooling2D,
    klayers.GlobalMaxPooling3D,
    klayers.MaxPooling1D,
    klayers.MaxPooling2D,
    klayers.MaxPooling3D,
)

#
#

#
# ############################################
# Utility list enabling name mappings via string
REVEAL_RULES: dict = {
    "Z": rrule.ZRule,
    "Epsilon": rrule.EpsilonRule,
    "WSquare": rrule.WSquareRule,
    "Flat": rrule.FlatRule,
    "AlphaBeta": rrule.AlphaBetaRule,
    "AlphaBetaIgnoreBias": rrule.AlphaBetaIgnoreBiasRule,
    "Alpha2Beta1": rrule.Alpha2Beta1Rule,
    "Alpha2Beta1IgnoreBias": rrule.Alpha2Beta1IgnoreBiasRule,
    "Alpha1Beta0": rrule.Alpha1Beta0Rule,
    "Alpha1Beta0IgnoreBias": rrule.Alpha1Beta0IgnoreBiasRule,
    "ZPlus": rrule.ZPlusRule,
    "ZPlusFast": rrule.ZPlusFastRule,
    "Bounded": rrule.BoundedRule,
}


class EmbeddingRevealLayer(igraph.ReverseMappingBase):
    def __init__(self, _layer, _state):
        # TODO: implement rule support.
        pass

    def apply(self, _Xs, _Ys, Rs, _reverse_state: dict):
        # the embedding layer outputs for an (indexed) input a vector.
        # thus, in the relevance backward pass, the embedding layer receives
        # relevances Rs corresponding to those vectors.
        # Due to the 1:1 relationship between input index and output mapping vector,
        # the relevance backward pass can be realized by pooling relevances
        # over the vector axis.

        # relevances are given shaped [batch_size, sequence_length, embedding_dims]
        pool_relevance = klayers.Lambda(lambda x: kbackend.sum(x, axis=-1))
        return [pool_relevance(R) for R in Rs]


class BatchNormalizationRevealLayer(igraph.ReverseMappingBase):
    """Special BN handler that applies the Z-Rule"""

    def __init__(self, layer, _state):
        config = layer.get_config()

        self._axis = _state["layer"].axis
        self._beta = _state["layer"].beta
        self.gamma = _state["layer"].gamma
        self.eps = _state["layer"].epsilon

        self._center = _state["layer"].center

        self._mean = _state["layer"].moving_mean
        self._var = _state["layer"].moving_variance


    def apply(self, Xs, Ys, Rs, _reverse_state: dict):
#         mask_the_zeros = ilayers.Not_Equal_Zero()(Rs)
#         casted_mask_the_zeros = [ilayers.Cast_To_Float()(mask_the_zeros)]
#         casted_mask_the_zeros_list = ilayers.Split(num_or_size_splits=_reverse_state["masks_size"])(
#             casted_mask_the_zeros)
        
#         not_mask_the_zeros = ilayers.Equal_Zero()(Xs[0])
#         not_casted_mask_the_zeros =ilayers.Cast_To_Float()(not_mask_the_zeros)

#         casted_mask_the_zeros_list = [ilayers.Add()([a, not_casted_mask_the_zeros]) for a in casted_mask_the_zeros_list]
#         # casted_mask_the_zeros_list_float = [ilayers.Cast_To_Bool()(a) for a in casted_mask_the_zeros_list]



#         contribution_features = ilayers.Split(num_or_size_splits=_reverse_state["masks_size"])(Rs)
        

#         scaled_tensor = [ilayers.Multiply()([a, tf.constant(10.0)]) for a in contribution_features]
#         rounded_tensor = [ilayers.Ceil()(a) for a in scaled_tensor]
#         activator_relevances_prime = [ilayers.Divide_no_nan()([a, tf.constant(10.0)]) for a in rounded_tensor]


#         net_plus_bias = ilayers.Substract()([activator_relevances_prime[-1], self._mean])
#         # net_plus_bias = activator_relevances_prime[-1]
#         net_plus_bias_mean = ilayers.Reduce_mean()(net_plus_bias)


#         ratio_net_plus_bias = ilayers.SafeDivide()([net_plus_bias, net_plus_bias_mean])
#         absolut_net = ilayers.Absolut()([ratio_net_plus_bias])
#         scaled_tensor = ilayers.Multiply()([absolut_net, tf.constant(1000.0)]) 
#         rounded_tensor = ilayers.Ceil()(scaled_tensor) 
#         absolut_net =  ilayers.Divide_no_nan()([rounded_tensor, tf.constant(1000.0)])
        

#         ratio = [ilayers.Divide_no_nan()([a, activator_relevances_prime[-1]]) for a in activator_relevances_prime]
#         absolut_bias = [ilayers.Absolut()([a]) for a in ratio]

#         scaled_tensor = [ilayers.Multiply()([a, tf.constant(1000.0)]) for a in absolut_bias]
#         rounded_tensor = [ilayers.Ceil()(a) for a in scaled_tensor]
#         absolut_bias = [ilayers.Divide_no_nan()([a, tf.constant(1000.0)]) for a in rounded_tensor]
    
# # ========================================================================================================

#         log_of_ten_Ys = ilayers.Log_Of_Ten()(absolut_net)
#         not_equal_Ys = ilayers.Not_Equal_Zero()(absolut_net)        
#         log_of_ten_net = ilayers.Where()([not_equal_Ys, log_of_ten_Ys, tf.constant(0.0)])[0]
#         std_from_the_min_value = ilayers.Reduce_std_sparse()([log_of_ten_net, casted_mask_the_zeros_list[-1]])

# # ========================================================================================================

#         log_of_ten = [ilayers.Log_Of_Ten()(a) for a in absolut_bias]
#         not_equal = [ilayers.Not_Equal_Zero()([a]) for a in absolut_bias]
#         log_of_ten = [ilayers.Where()([a, b, tf.constant(0.0)]) for a, b in zip(not_equal, log_of_ten)]
#         squeezed_log = [ilayers.Squeeze()(a) for a in log_of_ten]
#         std_bias =  [ilayers.Reduce_std_sparse()([a, b]) for a, b in zip(squeezed_log, casted_mask_the_zeros_list)]
        

# # ========================================================================================================
#         mean_std_more_than_one = [ilayers.MoreThan()([a, std_from_the_min_value]) for a  in std_bias]
#         mean_bias = [ilayers.Reduce_mean_sparse()([a, b]) for a, b in zip(squeezed_log, casted_mask_the_zeros_list)]
#         ratio_norm = [ilayers.Substract()([a, b]) for a, b in zip(squeezed_log, mean_bias)]
#         ratio_norm = [ilayers.Divide_no_nan()([a, b]) for a, b in zip(ratio_norm, std_bias)]
#         ratio_norm = [ilayers.Multiply()([a, std_from_the_min_value]) for a in ratio_norm]
#         ratio_norm = [ilayers.Add()([a, b]) for a, b in zip(ratio_norm, mean_bias)]
#         new_bias = [ilayers.Where()([a, b, c]) for a, b, c in zip(mean_std_more_than_one, ratio_norm, squeezed_log)]


# # ========================================================================================================


#         power = [ilayers.Power()(a) for a in new_bias]
#         scaler = [ilayers.Where()([a, b, tf.constant(0.0)]) for a, b in zip(not_equal, power)]    
#         scaler = [ilayers.Where()([not_mask_the_zeros, b, tf.constant(1.0)]) for b in scaler]

#         weighted_mean = [ilayers.Multiply()([self._mean, a]) for a in scaler]

#         contribution = [ilayers.Substract()([a, b]) for a, b in zip(contribution_features, weighted_mean)]
#         # contribution = [ilayers.Substract()([a, self._mean]) for a in contribution_features]
#         std = ilayers.Squareroot_the_variance(eps = self.eps)(self._var)
#         contribution = [ilayers.Divide_no_nan()([a, std]) for a in contribution]
        

        # var_cont = [ilayers.Reduce_var_not_sparse(mean = self._mean)(a) for a in contribution_features]
        # print(var_cont[0].shape)
        # # mult_cont = [ilayers.Multiply()([contribution_features[-1], a]) for a in casted_mask_the_zeros_list]
        # # var_cont_prime = [ilayers.Reduce_var_sparse(mean = weighted_mean[-1])([a, b]) for a, b in zip(mult_cont, casted_mask_the_zeros_list)]

        # ratio = [ilayers.Divide_no_nan()([a, var_cont[-1]]) for a in var_cont]
        # absolut_net = [ilayers.Absolut()([a]) for a in ratio]
        # scaled_tensor = [ilayers.Multiply()([a, tf.constant(100.0)]) for a in absolut_net]
        # rounded_tensor = [ilayers.Ceil()(a) for a in scaled_tensor]
        # absolut_rat =  [ilayers.Divide_no_nan()([a, tf.constant(100.0)]) for a in rounded_tensor]

        # new_var = [ilayers.Multiply()([self._var, a]) for a in absolut_rat]

        # std = [ilayers.Squareroot_the_variance(eps = self.eps)(a) for a in new_var]

        # # contribution_features_scaled_down = [ilayers.Divide_no_nan()([a, std]) for a in contribution_features]
        # # mean_scaled =  ilayers.Divide_no_nan()([self._mean, std]) 

        # # contribution = [ilayers.Substract()([a, self._mean]) for a in contribution_features]
        # contribution = [ilayers.Divide_no_nan()([a, b]) for a, b in zip(contribution, std)]




        # if self._center:
        #     contribution = [ilayers.Add()([a, self._beta]) for a in contribution]
    #         scaled_tensor = [ilayers.Multiply()([a, tf.constant(10.0)]) for a in contribution]
    #         rounded_tensor = [ilayers.Ceil()(a) for a in scaled_tensor]
    #         activator_relevances_prime = [ilayers.Divide_no_nan()([a, tf.constant(10.0)]) for a in rounded_tensor]


    #         # contribution = ilayers.Add()([contribution[-1], self._beta])
    #         net_plus_bias = ilayers.Add()([activator_relevances_prime[-1], self._beta])
    #         net_plus_bias_mean = ilayers.Reduce_mean()(net_plus_bias)


    #         ratio_net_plus_bias = ilayers.SafeDivide()([net_plus_bias, net_plus_bias_mean])
    #         absolut_net = ilayers.Absolut()([ratio_net_plus_bias])
    #         scaled_tensor = ilayers.Multiply()([absolut_net, tf.constant(10.0)]) 
    #         rounded_tensor = ilayers.Ceil()(scaled_tensor) 
    #         absolut_net =  ilayers.Divide_no_nan()([rounded_tensor, tf.constant(10.0)])
            

    #         # ratio = [ilayers.SafeDivide()([a, activator_relevances_prime[-1]]) for a in activator_relevances_prime]
    #         # absolut_bias = [ilayers.Absolut()([a]) for a in ratio]

    #         # scaled_tensor = [ilayers.Multiply()([a, tf.constant(100.0)]) for a in absolut_bias]
    #         # rounded_tensor = [ilayers.Ceil()(a) for a in scaled_tensor]
    #         # absolut_bias = [ilayers.Divide_no_nan()([a, tf.constant(100.0)]) for a in rounded_tensor]
        
    # # ========================================================================================================

    #         log_of_ten_Ys = ilayers.Log_Of_Ten()(absolut_net)
    #         not_equal_Ys = ilayers.Not_Equal_Zero()(absolut_net)        
    #         log_of_ten_net = ilayers.Where()([not_equal_Ys, log_of_ten_Ys, tf.constant(0.0)])[0]
    #         std_from_the_min_value = ilayers.Reduce_std_sparse()([log_of_ten_net, casted_mask_the_zeros_list[-1]])

    # # # ========================================================================================================

    #         # log_of_ten = [ilayers.Log_Of_Ten()(a) for a in absolut_bias]
    #         # not_equal = [ilayers.Not_Equal_Zero()([a]) for a in absolut_bias]
    #         # log_of_ten = [ilayers.Where()([a, b, tf.constant(0.0)]) for a, b in zip(not_equal, log_of_ten)]
    #         # squeezed_log = [ilayers.Squeeze()(a) for a in log_of_ten]
    #         # std_bias =  [ilayers.Reduce_std_sparse()([a, b]) for a, b in zip(squeezed_log, casted_mask_the_zeros_list)]
            

    # # # ========================================================================================================
    #         mean_std_more_than_one = [ilayers.MoreThan()([a, std_from_the_min_value]) for a  in std_bias]
    #         mean_bias = [ilayers.Reduce_mean_sparse()([a, b]) for a, b in zip(squeezed_log, casted_mask_the_zeros_list)]
    #         ratio_norm = [ilayers.Substract()([a, b]) for a, b in zip(squeezed_log, mean_bias)]
    #         ratio_norm = [ilayers.Divide_no_nan()([a, b]) for a, b in zip(ratio_norm, std_bias)]
    #         ratio_norm = [ilayers.Multiply()([a, std_from_the_min_value]) for a in ratio_norm]
    #         ratio_norm = [ilayers.Add()([a, b]) for a, b in zip(ratio_norm, mean_bias)]
    #         new_bias = [ilayers.Where()([a, b, c]) for a, b, c in zip(mean_std_more_than_one, ratio_norm, squeezed_log)]


    # # # ========================================================================================================


    #         power = [ilayers.Power()(a) for a in new_bias]
    #         scaler = [ilayers.Where()([a, b, tf.constant(0.0)]) for a, b in zip(not_equal, power)]   
             

            # weighted_beta = [ilayers.Multiply()([self._beta, a]) for a in scaler]

            # contribution = [ilayers.Add()([a, b]) for a, b in zip(contribution, weighted_beta)]

            
        # contribution = [ilayers.Multiply()([a, b]) for a, b in zip(contribution, casted_mask_the_zeros_list)]

        # ratio = [ilayers.Divide_no_nan()([Rs, Ys[0]]) ]
        # contribution = [ilayers.Concat()(contribution)]
        # ratio = [ilayers.Concat()(ratio)]

        return Rs, Rs


class AddRevealLayer(igraph.ReverseMappingBase):
    """Special Add layer handler that applies the Z-Rule"""

    def __init__(self, layer, _state):
        self._layer_wo_act = igraph.copy_layer_wo_activation(
            layer, name_template="reversed_kernel_%s"
        )

        # TODO: implement rule support.
        # super().__init__(layer, state)

    def apply(self, Xs, _Ys, Rs, _reverse_state: dict):
        # The outputs of the pooling operation at each location
        # is the sum of its inputs.
        # The forward message must be known in this case,
        # and are the inputs for each pooling thing.
        # The gradient is 1 for each output-to-input connection,
        # which corresponds to the "weights" of the layer.
        # It should thus be sufficient to reweight the relevances
        # and do a gradient_wrt
            # Get activations.
        contribution = [_reverse_state["layer"](Rs)]
        ratio = [ilayers.Divide_no_nan()([a, b]) for a, b in zip(contribution, _Ys)]
        return contribution, ratio


class AveragePoolingRevealLayer(igraph.ReverseMappingBase):
    """Special AveragePooling handler that applies the Z-Rule"""

    def __init__(self, layer, _state):
        self.layer = layer

        # TODO: implement rule support.
        # super().__init__(layer, state)

    def apply(self, Xs, _Ys, Rs, reverse_state: dict):

        list_con = ilayers.Split(num_or_size_splits=reverse_state["masks_size"], axis=0)(Rs)

        activator_relevances = ilayers.ApplyLayerToList([self.layer])(list_con)

        contribution = [ilayers.Concat(axis=0)(activator_relevances)]

        ratio = [ilayers.Divide_no_nan()([a, b]) for a, b in zip(contribution, _Ys)]
        return contribution, ratio


###############################################################################
# ANALYZER CLASSES AND PRESETS ################################################
###############################################################################
class REVEAL(ReverseAnalyzerBase):
    """
    Base class for Reveal-based model analyzers


    :param model: A Keras model.

    :param rule: A rule can be a  string or a Rule object, lists thereof or
      a list of conditions [(Condition, Rule), ... ]
      gradient.

    :param input_layer_rule: either a Rule object, atuple of (low, high)
      the min/max pixel values of the inputs
    :param bn_layer_rule: either a Rule object or None.
      None means dedicated BN rule will be applied.
    """

    def __init__(
        self,
        model,
        *args,
        rule=None,
        input_layer_rule=None,
        until_layer_idx=None,
        until_layer_rule=None,
        bn_layer_rule=None,
        bn_layer_fuse_mode: str = "one_linear",
        **kwargs,
    ):
        super().__init__(model, *args, **kwargs)

        self._input_layer_rule = input_layer_rule
        self._until_layer_rule = until_layer_rule
        self._until_layer_idx = until_layer_idx
        self._bn_layer_rule = bn_layer_rule
        self._bn_layer_fuse_mode = bn_layer_fuse_mode

        if self._bn_layer_fuse_mode not in ["one_linear", "two_linear"]:
            raise ValueError(f"Unknown _bn_layer_fuse_mode {self._bn_layer_fuse_mode}")

        self._add_model_softmax_check()
        self._add_model_check(
            lambda layer: not ichecks.is_convnet_layer(layer),
            "LRP is only tested for convolutional neural networks.",
            check_type="warning",
        )

        # TODO: refactor rule type checking into separate function
        # check if rule was given explicitly.
        # rule can be a string, a list (of strings) or
        # a list of conditions [(Condition, Rule), ... ] for each layer.
        if rule is None:
            raise ValueError("Need LRP rule(s).")

        if isinstance(rule, list):
            self._rule = list(rule)
        else:
            self._rule = rule

        if isinstance(rule, str) or (
            inspect.isclass(rule) and issubclass(rule, igraph.ReverseMappingBase)
        ):  # NOTE: All LRP rules inherit from igraph.ReverseMappingBase
            # the given rule is a single string or single rule implementing cla ss
            use_conditions = True
            rules = [(lambda _: True, rule)]

        elif not isinstance(rule[0], tuple):
            # rule list of rule strings or classes
            use_conditions = False
            rules = list(rule)
        else:
            # rule is list of conditioned rules
            use_conditions = True
            rules = rule

        # apply rule to first self._until_layer_idx layers
        if self._until_layer_rule is not None and self._until_layer_idx is not None:
            for i in range(self._until_layer_idx + 1):
                is_at_idx: LayerCheck = lambda layer: ichecks.is_layer_at_idx(layer, i)
                rules.insert(0, (is_at_idx, self._until_layer_rule))

        # create a BoundedRule for input layer handling from given tuple
        if self._input_layer_rule is not None:
            input_layer_rule = self._input_layer_rule
            if isinstance(input_layer_rule, tuple):
                low, high = input_layer_rule

                class BoundedProxyRule(rrule.BoundedRule):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, low=low, high=high, **kwargs)

                input_layer_rule = BoundedProxyRule

            if use_conditions is True:
                rules.insert(0, (ichecks.is_input_layer, input_layer_rule))
            else:
                rules.insert(0, input_layer_rule)

        self._rules_use_conditions = use_conditions
        self._rules = rules

    def create_rule_mapping(self, layer: Layer, reverse_state: dict):
        if self._rules_use_conditions is True:
            for condition, rule in self._rules:
                if condition(layer):
                    rule_class = rule
                    break
        else:
            rule_class = self._rules.pop()

        if rule_class is None:
            raise Exception(f"No rule applies to layer {layer}")

        if isinstance(rule_class, str):
            rule_class = REVEAL_RULES[rule_class]
        rule = rule_class(layer, reverse_state)

        return rule.apply

    def _create_analysis(self, *args, **kwargs):
        ###################################################################
        # Functionality responible for backwards rule selection below  ####
        ###################################################################

        # default backward hook
        self._add_conditional_reverse_mapping(
            ichecks.contains_kernel,
            self.create_rule_mapping,
            name="reveal_layer_with_kernel_mapping",
        )

        # specialized backward hooks.
        # TODO: add ReverseLayer class handling layers without kernel: Add and AvgPool
        bn_layer_rule = self._bn_layer_rule

        if bn_layer_rule is None:
            # TODO (alber): get rid of this option!
            # alternatively a default rule should be applied.
            bn_mapping = BatchNormalizationRevealLayer
        else:
            if isinstance(bn_layer_rule, str):
                bn_layer_rule = REVEAL_RULES[bn_layer_rule]

            bn_mapping = igraph.apply_mapping_to_fused_bn_layer(
                bn_layer_rule,
                fuse_mode=self._bn_layer_fuse_mode,
            )
        self._add_conditional_reverse_mapping(
            ichecks.is_batch_normalization_layer,
            bn_mapping,
            name="reveal_batch_norm_mapping",
        )
        self._add_conditional_reverse_mapping(
            ichecks.is_average_pooling,
            AveragePoolingRevealLayer,
            name="reveal_average_pooling_mapping",
        )
        self._add_conditional_reverse_mapping(
            ichecks.is_add_layer,
            AddRevealLayer,
            name="reveal_add_layer_mapping",
        )
        self._add_conditional_reverse_mapping(
            ichecks.is_embedding_layer,
            EmbeddingRevealLayer,
            name="reveal_embedding_mapping",
        )

        # FINALIZED constructor.
        return super()._create_analysis(*args, **dict(**kwargs, forward_contibution=True, random_masks=self.masks, index = self.index))

    def _default_reverse_mapping(
        self,
        Xs: OptionalList[Tensor],
        Ys: OptionalList[Tensor],
        masked_Xs: OptionalList[Tensor],
        reverse_state: dict,
    ):
        # default_return_layers = [klayers.Activation]# TODO extend
        if (
            len(Xs) == len(Ys)
            and isinstance(reverse_state["layer"], (klayers.Activation,))
            and all(
                kbackend.int_shape(x) == kbackend.int_shape(y) for x, y in zip(Xs, Ys)
            )
        ):
            # if not (reverse_state["layer"].get_config()["activation"] == "relu"):
            # layer = tf.keras.layers.Activation(self._layer.get_config()["activation"])
            # new_Ys = ibackend.apply(layer, masked_Xs)

            mask_the_zeros = ilayers.Not_Equal_Zero()(Ys)

            casted_mask_the_zeros = [ilayers.Cast_To_Float()(mask_the_zeros)]

            new_Ys = [ilayers.Multiply()([a, b]) for a, b in zip(masked_Xs, casted_mask_the_zeros)]

            ratio = [ilayers.Divide_no_nan()([a, b]) for a, b in zip(new_Ys, Ys)]
            # Expect Xs and Ys to have the same shapes.
            # There is not mixing of relevances as there is kernel,
            # therefore we pass them as they are.
            return new_Ys, ratio
        elif (isinstance(reverse_state["layer"], keras.layers.MaxPooling1D)
            or isinstance(reverse_state["layer"], keras.layers.MaxPooling2D)
            or isinstance(reverse_state["layer"], keras.layers.MaxPooling3D)
            or isinstance(reverse_state["layer"], keras.layers.Softmax)
            or isinstance(reverse_state["layer"], keras.layers.GlobalMaxPooling1D)
            or isinstance(reverse_state["layer"], keras.layers.GlobalMaxPooling2D)
            or isinstance(reverse_state["layer"], keras.layers.GlobalMaxPooling3D)
            ):

            # contribution = ilayers.MaxPool_forward(len(Xs))(Xs + Ys + masked_Xs + [reverse_state])

            # grad = tf.gradients(Ys, Xs)
            grad = ilayers.Gradient()([Ys, Xs])

            mask_the_zeros = ilayers.Not_Equal_Zero()(grad)

            casted_mask_the_zeros = [ilayers.Cast_To_Float()(mask_the_zeros)]

            Xs_prime = [ilayers.Multiply()([a, b]) for a, b in zip(masked_Xs, casted_mask_the_zeros)]
            list_Xs_prime = ilayers.Split(num_or_size_splits=reverse_state["masks_size"], axis=0)(Xs_prime)

            abs_Xs = [ilayers.Absolut()(Xs_prime)]
            list_abs_Xs_prime = ilayers.Split(num_or_size_splits=reverse_state["masks_size"], axis=0)(abs_Xs)

            absolute_Ys = ilayers.ApplyLayerToList([reverse_state["layer"]])(list_abs_Xs_prime)
            non_Ys = ilayers.ApplyLayerToList([reverse_state["layer"]])(list_Xs_prime)

            zero_the_non_zeros_non_Ys = [ilayers.Equal_Zero()(a) for a in non_Ys]

            contribution = [ilayers.Where()([a, -b, c]) for a, b, c in zip(zero_the_non_zeros_non_Ys, absolute_Ys, non_Ys)]
            contribution = [ilayers.Concat(axis=0)(contribution)]

            # list_con = ilayers.Split(num_or_size_splits=reverse_state["masks_size"], axis=0)(masked_Xs)

            # activator_relevances = ilayers.ApplyLayerToList([reverse_state["layer"]])(list_con)

            # contribution_no_act_no_bias = [ilayers.Concat(axis=0)(activator_relevances)]


            # mask_the_zeros = ilayers.Not_Equal_Zero()(Ys)
            #
            # casted_mask_the_zeros = [ilayers.Cast_To_Float()(mask_the_zeros)]
            #
            # contribution = [ilayers.Multiply()([a, b]) for a, b in zip(contribution, casted_mask_the_zeros)]

            ratio = [ilayers.Divide_no_nan()([a, b]) for a, b in zip(contribution, Ys)]

            return contribution, ratio

        else:
            # if isinstance(reverse_state["layer"], keras.layers.Concatenate):
            #     list_con = [ilayers.Split(num_or_size_splits=reverse_state["masks_size"], axis=0)([x]) for x in masked_Xs]
            #     list_con_t = list(map(list, zip(*list_con)))
            #     # list_con = np.array(list_con).T.tolist()
            #     activator_relevances = ilayers.ApplyLayerToList([reverse_state["layer"]])(list_con_t)
            #     contribution = [ilayers.Concat(axis=0)(activator_relevances)]
            # else:
            #     print("hey")

            # mask_the_zeros = ilayers.Not_Equal_Zero()(Ys)
            #
            # casted_mask_the_zeros = [ilayers.Cast_To_Float()(mask_the_zeros)]
            #
            # contribution = [ilayers.Multiply()([a, b]) for a, b in zip(contribution, casted_mask_the_zeros)]
            if len(Xs) > 1:
                contribution = [reverse_state["layer"](masked_Xs)]
                ratio = [ilayers.Divide_no_nan()([a, b]) for a, b in zip(contribution, Ys)]
            else: 
                list_con = ilayers.Split(num_or_size_splits=reverse_state["masks_size"], axis=0)(masked_Xs)
                activator_relevances = [reverse_state["layer"](a) for a in list_con]

                contribution = [ilayers.Concat(axis=0)(activator_relevances)]
                ratio = [ilayers.Divide_no_nan()([a, b]) for a, b in zip(contribution, Ys)]
            return contribution, ratio

    ######################################
    # End of Rule Selection Business. ####
    ######################################

    def _get_state(self):
        state = super()._get_state()
        state.update({"rule": self._rule})
        state.update({"input_layer_rule": self._input_layer_rule})
        state.update({"bn_layer_rule": self._bn_layer_rule})
        state.update({"bn_layer_fuse_mode": self._bn_layer_fuse_mode})
        return state

    @classmethod
    def _state_to_kwargs(cls, state):
        rule = state.pop("rule")
        input_layer_rule = state.pop("input_layer_rule")
        bn_layer_rule = state.pop("bn_layer_rule")
        bn_layer_fuse_mode = state.pop("bn_layer_fuse_mode")
        # call super after popping class-specific states:
        kwargs = super()._state_to_kwargs(state)

        kwargs.update(
            {
                "rule": rule,
                "input_layer_rule": input_layer_rule,
                "bn_layer_rule": bn_layer_rule,
                "bn_layer_fuse_mode": bn_layer_fuse_mode,
            }
        )
        return kwargs

class REVEALAlphaBeta(REVEAL):
    """Base class for LRP AlphaBeta"""

    def __init__(self, model, *args, alpha=None, beta=None, bias=True, **kwargs):
        alpha, beta = rutils.assert_infer_reveal_alpha_beta_param(alpha, beta, self)
        self._alpha = alpha
        self._beta = beta
        self._bias = bias
        self.random_masks = kwargs.get("random_masks")

        class AlphaBetaProxyRule(rrule.AlphaBetaRule):
            """
            Dummy class inheriting from AlphaBetaRule
            for the purpose of passing along the chosen parameters from
            the LRP analyzer class to the decopmosition rules.
            """

            def __init__(self, *args, **kwargs):
                super().__init__(*args, alpha=alpha, beta=beta, bias=bias, **kwargs)

        super().__init__(
            model,
            *args,
            rule=AlphaBetaProxyRule,
            bn_layer_rule=BatchNormalizationRevealLayer,
            **kwargs,
        )
        self._do_model_checks()

    def _get_state(self):
        state = super()._get_state()
        del state["rule"]
        state.update({"alpha": self._alpha})
        state.update({"beta": self._beta})
        state.update({"bias": self._bias})
        return state

    @classmethod
    def _state_to_kwargs(cls, state):
        alpha = state.pop("alpha")
        beta = state.pop("beta")
        bias = state.pop("bias")
        state["rule"] = None
        # call super after popping class-specific states:
        kwargs = super()._state_to_kwargs(state)

        del kwargs["rule"]
        del kwargs["bn_layer_rule"]
        kwargs.update({"alpha": alpha, "beta": beta, "bias": bias})
        return kwargs

class _REVEALAlphaBetaFixedParams(REVEALAlphaBeta):
    @classmethod
    def _state_to_kwargs(cls, state):
        # call super after popping class-specific states:
        kwargs = super()._state_to_kwargs(state)

        del kwargs["alpha"]
        del kwargs["beta"]
        del kwargs["bias"]
        return kwargs

class REVEALAlpha2Beta1(_REVEALAlphaBetaFixedParams):
    """LRP-analyzer that uses the LRP-alpha-beta rule with a=2,b=1"""

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, alpha=2, beta=1, bias=True, **kwargs)
        self._do_model_checks()