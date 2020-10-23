#!/usr/bin/env python3
import typing
from enum import Enum
from inspect import signature
from typing import Any, Callable, Dict, List, Tuple, Union, cast, overload

import numpy as np
import torch
from torch import Tensor, device
from torch.nn import Module

from .._utils.typing import (
    BaselineType,
    Literal,
    TargetType,
    TupleOrTensorOrBoolGeneric,
)


class ExpansionTypes(Enum):
    repeat = 1
    repeat_interleave = 2


def safe_div(
    denom: Tensor, quotient: Union[Tensor, float], default_value: Tensor
) -> Tensor:
    r"""
        A simple utility function to perform `denom / quotient`
        if the statement is undefined => result will be `default_value`
    """
    if isinstance(quotient, float):
        return denom / quotient if quotient != 0.0 else default_value

    # if quotient is a tensor
    return denom / torch.where(quotient != 0.0, quotient, default_value)


@typing.overload
def _is_tuple(inputs: Tensor) -> Literal[False]:
    ...


@typing.overload
def _is_tuple(inputs: Tuple[Tensor, ...]) -> Literal[True]:
    ...


def _is_tuple(inputs: Union[Tensor, Tuple[Tensor, ...]]) -> bool:
    return isinstance(inputs, tuple)


def _validate_target(num_samples: int, target: TargetType) -> None:
    if isinstance(target, list) or (
        isinstance(target, torch.Tensor) and torch.numel(target) > 1
    ):
        assert num_samples == len(target), (
            "The number of samples provied in the"
            "input {} does not match with the number of targets. {}".format(
                num_samples, len(target)
            )
        )


def _validate_input(
    inputs: Tuple[Tensor, ...],
    baselines: Tuple[Union[Tensor, int, float], ...],
    draw_baseline_from_distrib: bool = False,
) -> None:
    assert len(inputs) == len(baselines), (
        "Input and baseline must have the same "
        "dimensions, baseline has {} features whereas input has {}.".format(
            len(baselines), len(inputs)
        )
    )

    for input, baseline in zip(inputs, baselines):
        if draw_baseline_from_distrib:
            assert (
                isinstance(baseline, (int, float))
                or input.shape[1:] == baseline.shape[1:]
            ), (
                "The samples in input and baseline batches must have"
                " the same shape or the baseline corresponding to the"
                " input tensor must be a scalar."
                " Found baseline: {} and input: {} ".format(baseline, input)
            )
        else:
            assert (
                isinstance(baseline, (int, float))
                or input.shape == baseline.shape
                or baseline.shape[0] == 1
            ), (
                "Baseline can be provided as a tensor for just one input and"
                " broadcasted to the batch or input and baseline must have the"
                " same shape or the baseline corresponding to each input tensor"
                " must be a scalar. Found baseline: {} and input: {}".format(
                    baseline, input
                )
            )


def _zeros(inputs: Tuple[Tensor, ...]) -> Tuple[int, ...]:
    r"""
    Takes a tuple of tensors as input and returns a tuple that has the same
    length as `inputs` with each element as the integer 0.
    """
    return tuple(0 for input in inputs)


def _format_baseline(
    baselines: BaselineType, inputs: Tuple[Tensor, ...]
) -> Tuple[Union[Tensor, int, float], ...]:
    if baselines is None:
        return _zeros(inputs)

    if not isinstance(baselines, tuple):
        baselines = (baselines,)

    for baseline in baselines:
        assert isinstance(
            baseline, (torch.Tensor, int, float)
        ), "baseline input argument must be either a torch.Tensor or a number \
            however {} detected".format(
            type(baseline)
        )

    return baselines


@overload
def _format_tensor_into_tuples(inputs: None) -> None:
    ...


@overload
def _format_tensor_into_tuples(
    inputs: Union[Tensor, Tuple[Tensor, ...]]
) -> Tuple[Tensor, ...]:
    ...


def _format_tensor_into_tuples(
    inputs: Union[None, Tensor, Tuple[Tensor, ...]]
) -> Union[None, Tuple[Tensor, ...]]:
    if inputs is None:
        return None
    if not isinstance(inputs, tuple):
        assert isinstance(
            inputs, torch.Tensor
        ), "`inputs` must have type " "torch.Tensor but {} found: ".format(type(inputs))
        inputs = (inputs,)
    return inputs


def _format_input(inputs: Union[Tensor, Tuple[Tensor, ...]]) -> Tuple[Tensor, ...]:
    return _format_tensor_into_tuples(inputs)


@overload
def _format_additional_forward_args(additional_forward_args: None) -> None:
    ...


@overload
def _format_additional_forward_args(
    additional_forward_args: Union[Tensor, Tuple]
) -> Tuple:
    ...


@overload
def _format_additional_forward_args(additional_forward_args: Any) -> Union[None, Tuple]:
    ...


def _format_additional_forward_args(additional_forward_args: Any) -> Union[None, Tuple]:
    if additional_forward_args is not None and not isinstance(
        additional_forward_args, tuple
    ):
        additional_forward_args = (additional_forward_args,)
    return additional_forward_args


def _expand_additional_forward_args(
    additional_forward_args: Any,
    n_steps: int,
    expansion_type: ExpansionTypes = ExpansionTypes.repeat,
) -> Union[None, Tuple]:
    def _expand_tensor_forward_arg(
        additional_forward_arg: Tensor,
        n_steps: int,
        expansion_type: ExpansionTypes = ExpansionTypes.repeat,
    ) -> Tensor:
        if len(additional_forward_arg.size()) == 0:
            return additional_forward_arg
        if expansion_type == ExpansionTypes.repeat:
            return torch.cat([additional_forward_arg] * n_steps, dim=0)
        elif expansion_type == ExpansionTypes.repeat_interleave:
            return additional_forward_arg.repeat_interleave(n_steps, dim=0)
        else:
            raise NotImplementedError(
                "Currently only `repeat` and `repeat_interleave`"
                " expansion_types are supported"
            )

    if additional_forward_args is None:
        return None

    return tuple(
        _expand_tensor_forward_arg(additional_forward_arg, n_steps, expansion_type)
        if isinstance(additional_forward_arg, torch.Tensor)
        else additional_forward_arg
        for additional_forward_arg in additional_forward_args
    )


def _expand_target(
    target: TargetType,
    n_steps: int,
    expansion_type: ExpansionTypes = ExpansionTypes.repeat,
) -> TargetType:
    if isinstance(target, list):
        if expansion_type == ExpansionTypes.repeat:
            return target * n_steps
        elif expansion_type == ExpansionTypes.repeat_interleave:
            expanded_target = []
            for i in target:
                expanded_target.extend([i] * n_steps)
            return cast(Union[List[Tuple[int, ...]], List[int]], expanded_target)
        else:
            raise NotImplementedError(
                "Currently only `repeat` and `repeat_interleave`"
                " expansion_types are supported"
            )

    elif isinstance(target, torch.Tensor) and torch.numel(target) > 1:
        if expansion_type == ExpansionTypes.repeat:
            return torch.cat([target] * n_steps, dim=0)
        elif expansion_type == ExpansionTypes.repeat_interleave:
            return target.repeat_interleave(n_steps, dim=0)
        else:
            raise NotImplementedError(
                "Currently only `repeat` and `repeat_interleave`"
                " expansion_types are supported"
            )

    return target


def _expand_and_update_baselines(
    inputs: Tuple[Tensor, ...],
    n_samples: int,
    kwargs: dict,
    draw_baseline_from_distrib: bool = False,
):
    def get_random_baseline_indices(bsz, baseline):
        num_ref_samples = baseline.shape[0]
        return np.random.choice(num_ref_samples, n_samples * bsz).tolist()

    # expand baselines to match the sizes of input
    if "baselines" not in kwargs:
        return

    baselines = kwargs["baselines"]
    baselines = _format_baseline(baselines, inputs)
    _validate_input(
        inputs, baselines, draw_baseline_from_distrib=draw_baseline_from_distrib
    )

    if draw_baseline_from_distrib:
        bsz = inputs[0].shape[0]
        baselines = tuple(
            baseline[get_random_baseline_indices(bsz, baseline)]
            if isinstance(baseline, torch.Tensor)
            else baseline
            for baseline in baselines
        )
    else:
        baselines = tuple(
            baseline.repeat_interleave(n_samples, dim=0)
            if isinstance(baseline, torch.Tensor)
            and baseline.shape[0] == input.shape[0]
            and baseline.shape[0] > 1
            else baseline
            for input, baseline in zip(inputs, baselines)
        )
    # update kwargs with expanded baseline
    kwargs["baselines"] = baselines


def _expand_and_update_additional_forward_args(n_samples: int, kwargs: dict):
    if "additional_forward_args" not in kwargs:
        return
    additional_forward_args = kwargs["additional_forward_args"]
    additional_forward_args = _format_additional_forward_args(additional_forward_args)
    if additional_forward_args is None:
        return
    additional_forward_args = _expand_additional_forward_args(
        additional_forward_args,
        n_samples,
        expansion_type=ExpansionTypes.repeat_interleave,
    )
    # update kwargs with expanded baseline
    kwargs["additional_forward_args"] = additional_forward_args


def _expand_and_update_target(n_samples: int, kwargs: dict):
    if "target" not in kwargs:
        return
    target = kwargs["target"]
    target = _expand_target(
        target, n_samples, expansion_type=ExpansionTypes.repeat_interleave
    )
    # update kwargs with expanded baseline
    kwargs["target"] = target


@typing.overload
def _format_output(
    is_inputs_tuple: Literal[True], output: Tuple[Tensor, ...]
) -> Tuple[Tensor, ...]:
    ...


@typing.overload
def _format_output(
    is_inputs_tuple: Literal[False], output: Tuple[Tensor, ...]
) -> Tensor:
    ...


@typing.overload
def _format_output(
    is_inputs_tuple: bool, output: Tuple[Tensor, ...]
) -> Union[Tensor, Tuple[Tensor, ...]]:
    ...


def _format_output(
    is_inputs_tuple: bool, output: Tuple[Tensor, ...]
) -> Union[Tensor, Tuple[Tensor, ...]]:
    r"""
    In case input is a tensor and the output is returned in form of a
    tuple we take the first element of the output's tuple to match the
    same shape signatues of the inputs
    """
    assert isinstance(output, tuple), "Output must be in shape of a tuple"
    assert is_inputs_tuple or len(output) == 1, (
        "The input is a single tensor however the output isn't."
        "The number of output tensors is: {}".format(len(output))
    )
    return output if is_inputs_tuple else output[0]


def _run_forward(
    forward_func: Callable,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    target: TargetType = None,
    additional_forward_args: Any = None,
    application = None
) -> Tensor:
    forward_func_args = signature(forward_func).parameters
    if len(forward_func_args) == 0:
        output = forward_func()
        return output if target is None else _select_targets(output, target)

    # make everything a tuple so that it is easy to unpack without
    # using if-statements
    inputs = _format_input(inputs)
    additional_forward_args = _format_additional_forward_args(additional_forward_args)

    print('captum._utils.common: inputs', inputs)

    output = forward_func(
        *(*inputs, *additional_forward_args)
        if additional_forward_args is not None
        else inputs
    )

    try:
        print('captum._utlis.common: You should not be here!')
        ret = _select_targets(output, target)
    except:
        print('captum._utils.common, output[0]:', output[0])
        ret = _select_targets(output[0], target)

    # Below line won't work currently as this is being called in multiple places
    # ret = _select_targets(output[0], target) if application == "text" else _select_targets(output, target)
    return ret  # Should be `output[0]` for text; `output` for others.


def _select_targets(output: Tensor, target: TargetType) -> Tensor:
    if target is None:
        return output

    num_examples = output.shape[0]
    dims = len(output.shape)
    device = output.device
    if isinstance(target, (int, tuple)):
        return _verify_select_column(output, target)
    elif isinstance(target, torch.Tensor):
        if torch.numel(target) == 1 and isinstance(target.item(), int):
            return _verify_select_column(output, cast(int, target.item()))
        elif len(target.shape) == 1 and torch.numel(target) == num_examples:
            assert dims == 2, "Output must be 2D to select tensor of targets."
            return torch.gather(output, 1, target.reshape(len(output), 1))
        else:
            raise AssertionError(
                "Tensor target dimension %r is not valid. %r"
                % (target.shape, output.shape)
            )
    elif isinstance(target, list):
        assert len(target) == num_examples, "Target list length does not match output!"
        if isinstance(target[0], int):
            assert dims == 2, "Output must be 2D to select tensor of targets."
            return torch.gather(
                output, 1, torch.tensor(target, device=device).reshape(len(output), 1)
            )
        elif isinstance(target[0], tuple):
            return torch.stack(
                [
                    output[(i,) + cast(Tuple, targ_elem)]
                    for i, targ_elem in enumerate(target)
                ]
            )
        else:
            raise AssertionError("Target element type in list is not valid.")
    else:
        raise AssertionError("Target type %r is not valid." % target)


def _verify_select_column(
    output: Tensor, target: Union[int, Tuple[int, ...]]
) -> Tensor:
    target = cast(Tuple[int, ...], (target,) if isinstance(target, int) else target)
    assert (
        len(target) <= len(output.shape) - 1
    ), "Cannot choose target column with output shape %r." % (output.shape,)
    return output[(slice(None), *target)]


def _extract_device(
    module: Module,
    hook_inputs: Union[None, Tensor, Tuple[Tensor, ...]],
    hook_outputs: Union[None, Tensor, Tuple[Tensor, ...]],
) -> device:
    params = list(module.parameters())
    if (
        (hook_inputs is None or len(hook_inputs) == 0)
        and (hook_outputs is None or len(hook_outputs) == 0)
        and len(params) == 0
    ):
        raise RuntimeError(
            """Unable to extract device information for the module
            {}. Both inputs and outputs to the forward hook and
            `module.parameters()` are empty.
            The reason that the inputs to the forward hook are empty
            could be due to the fact that the arguments to that
            module {} are all named and are passed as named
            variables to its forward function.
            """.format(
                module, module
            )
        )
    if hook_inputs is not None and len(hook_inputs) > 0:
        return hook_inputs[0].device
    if hook_outputs is not None and len(hook_outputs) > 0:
        return hook_outputs[0].device

    return params[0].device


def _reduce_list(
    val_list: List[TupleOrTensorOrBoolGeneric],
    red_func: Callable[[List], Any] = torch.cat,
) -> TupleOrTensorOrBoolGeneric:
    """
    Applies reduction function to given list. If each element in the list is
    a Tensor, applies reduction function to all elements of the list, and returns
    the output Tensor / value. If each element is a boolean, apply any method (or).
    If each element is a tuple, applies reduction
    function to corresponding elements of each tuple in the list, and returns
    tuple of reduction function outputs with length matching the length of tuple
    val_list[0]. It is assumed that all tuples in the list have the same length
    and red_func can be applied to all elements in each corresponding position.
    """
    if isinstance(val_list[0], torch.Tensor):
        return red_func(val_list)
    elif isinstance(val_list[0], bool):
        return any(val_list)
    elif isinstance(val_list[0], tuple):
        final_out = []
        for i in range(len(val_list[0])):
            final_out.append(
                _reduce_list([val_elem[i] for val_elem in val_list], red_func)
            )
    else:
        raise AssertionError(
            "Elements to be reduced can only be"
            "either Tensors or tuples containing Tensors."
        )
    return tuple(final_out)


def _sort_key_list(
    keys: List[device], device_ids: Union[None, List[int]] = None
) -> List[device]:
    """
    Sorts list of torch devices (keys) by given index list, device_ids. If keys
    contains only one device, then the list is returned unchanged. If keys
    contains a device for which the id is not contained in device_ids, then
    an error is returned. This method is used to identify the order of DataParallel
    batched devices, given the device ID ordering.
    """
    if len(keys) == 1:
        return keys
    id_dict: Dict[int, device] = {}
    assert device_ids is not None, "Device IDs must be provided with multiple devices."
    for key in keys:
        if key.index in id_dict:
            raise AssertionError("Duplicate CUDA Device ID identified in device list.")
        id_dict[key.index] = key

    out_list = [
        id_dict[device_id]
        for device_id in filter(lambda device_id: device_id in id_dict, device_ids)
    ]

    assert len(out_list) == len(keys), "Given Device ID List does not match"
    "devices with computed tensors."

    return out_list
