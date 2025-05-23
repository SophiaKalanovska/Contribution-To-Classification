from __future__ import annotations

__all__ = ["assert_reveal_epsilon_param", "assert_infer_reveal_alpha_beta_param"]


def assert_reveal_epsilon_param(epsilon, caller):
    """
    Function for asserting epsilon parameter choice
    passed to constructors inheriting from EpsilonRule
    and LRPEpsilon.
    The following conditions can not be met:

    epsilon > 1

    :param epsilon: the epsilon parameter.
    :param caller: the class instance calling this assertion function
    """

    if epsilon <= 0:
        err_head = f"Constructor call to {caller.__class__.__name__} : "
        err_msg = err_head + f"Parameter epsilon must be > 0 but was {epsilon}"
        raise ValueError(err_msg)
    return epsilon


def assert_infer_reveal_alpha_beta_param(alpha, beta, caller):
    """
    Function for asserting parameter choices for alpha and beta
    passed to constructors inheriting from AlphaBetaRule
    and LRPAlphaBeta.

    since alpha - beta are subjected to sum to 1,
    it is sufficient for only one of the parameters to be passed
    to a corresponding class constructor.
    this method will cause an assertion error if both are None
    or the following conditions can not be met

    alpha >= 1
    beta >= 0
    alpha - beta = 1

    :param alpha: the alpha parameter.
    :param beta: the beta parameter
    :param caller: the class instance calling this assertion function
    """

    # TODO: Rework error messages

    err_head = f"Constructor call to {caller.__class__.__name__} : "
    if alpha is None and beta is None:
        err_msg = err_head + "Neither alpha or beta were given"
        raise ValueError(err_msg)

    # assert passed parameter choices
    if alpha is not None and alpha < 1:
        err_msg = (
            err_head
            + f"Passed parameter alpha invalid. Expecting alpha >= 1 but was {alpha}"
        )
        raise ValueError(err_msg)

    if beta is not None and beta < 0:
        err_msg = (
            err_head
            + f"Passed parameter beta invalid. Expecting beta >= 0 but was {beta}"
        )
        raise ValueError(err_msg)

    # assert inferred parameter choices
    if alpha is None:
        alpha = beta + 1
        if alpha < 1:
            err_msg = (
                err_head
                + f"Inferring alpha from given beta {beta} s.t. alpha - beta = 1, "
                + "with condition alpha >= 1 not possible."
            )
            raise ValueError(err_msg)

    if beta is None:
        beta = alpha - 1
        if beta < 0:
            err_msg = (
                err_head
                + f"Inferring beta from given alpha {alpha} s.t. alpha - beta = 1, "
                + "with condition beta >= 0 not possible."
            )
            raise ValueError(err_msg)

    # final check: alpha - beta = 1
    amb = alpha - beta
    if amb != 1:
        err_msg = (
            err_head
            + "Condition alpha - beta = 1 not fulfilled. "
            + f"alpha={alpha} ; beta={beta} -> alpha - beta = {amb}"
        )
        raise ValueError(err_msg)

    # return benign values for alpha and beta
    return alpha, beta
