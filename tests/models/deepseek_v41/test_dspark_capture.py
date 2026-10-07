# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn


def test_target_forward_capture_keeps_target_output_identical():
    from vllm.models.deepseek_v41.common import contracts as C
    from vllm.models.deepseek_v41.sm70.model import DeepseekV41Model
    from vllm.sequence import IntermediateTensors

    class Layer(nn.Module):
        def __init__(self, layer_id: int) -> None:
            super().__init__()
            self.layer_id = layer_id

        def forward(self, h, pre, positions):
            return (h.float() + self.layer_id).bfloat16(), pre

    model = object.__new__(DeepseekV41Model)
    nn.Module.__init__(model)
    model.pp_is_first, model.pp_is_last = False, True
    model.start_layer, model.end_layer = 0, 3
    model.mirror, model.hc_fused = None, False
    model.layers = nn.ModuleList([Layer(i) for i in (37, 38, 39)])
    model.norm = nn.Module()
    model.norm.weight = nn.Parameter(torch.ones(5120))
    h = (torch.arange(2 * 4 * 5120).reshape(2, 4, 5120) % 64).bfloat16()
    pre = torch.ones(2, 4) / 4
    inputs = IntermediateTensors({C.PP_KEY_HIDDEN: h, C.PP_KEY_PRE_MIX: pre})
    baseline = model(None, torch.arange(2), inputs)
    model.dspark_aux_layers = (37, 38, 39)
    output, aux = model(None, torch.arange(2), inputs)
    assert torch.equal(output, baseline)
    assert len(aux) == 3
    for index, captured in enumerate(aux):
        expected = h
        for layer_id in range(37, 37 + index):
            expected = (expected.float() + layer_id).bfloat16()
        assert torch.equal(captured, expected.mean(1).float())


def test_runner_aux_ids_bind_to_last_stage_block_inputs():
    from types import SimpleNamespace

    from vllm.model_executor.models.interfaces import supports_eagle3
    from vllm.models.deepseek_v41.sm70.model import DeepseekV41ForCausalLM
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    runner = object.__new__(GPUModelRunner)
    runner.speculative_config = SimpleNamespace(
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(dspark_target_layer_ids=[37, 38, 39])
        )
    )
    aux = runner._get_eagle3_aux_layers_from_config()
    assert aux == (38, 39, 40)
    target = object.__new__(DeepseekV41ForCausalLM)
    nn.Module.__init__(target)
    target.model = nn.Module()
    target.model.pp_is_last = True
    target.model.owns_layer = lambda i: 28 <= i < 40
    assert supports_eagle3(target)
    target.set_aux_hidden_state_layers(aux)
    assert target.model.dspark_aux_layers == (37, 38, 39)
    target.model.pp_is_last = False
    target.set_aux_hidden_state_layers(aux)
    assert target.model.dspark_aux_layers == ()
