"""Policy network for discrete action prediction."""
from typing import Union, Dict, Tuple, Sequence, List

import torch
from maze.perception.blocks.general.action_masking import ActionMaskingBlock
from maze.perception.blocks.output.linear import LinearOutputBlock
from maze.perception.weight_init import make_module_init_normc
from torch import nn

from maze_action_set_es.models.unitary.base import BaseNet


class PolicyNet(BaseNet):
    """Policy network for discrete action prediction.

    :param obs_shapes: The observation shape.
    :param action_logits_shapes: The shapes of all actions as a dict structure.
    :param non_lin: The nonlinear activation to be used.
    :param hidden_units: The number of hidden units.
    """

    def __init__(self,
                 obs_shapes: Dict[str, Sequence[int]],
                 action_logits_shapes: Dict[str, Tuple[int]],
                 non_lin: Union[str, type(nn.Module)],
                 hidden_units: List[int]):
        super().__init__(obs_shapes, non_lin, hidden_units)

        # add link prediction path
        out_key = 'action_logits' if 'action_mask' in self.obs_shapes else 'action'
        self.perception_dict[out_key] = LinearOutputBlock(
            in_keys='hidden_out', in_shapes=self.perception_dict['hidden_out'].out_shapes(),
            out_keys=out_key, output_units=action_logits_shapes['action'][0]
        )

        if 'action_mask' in self.obs_shapes:
            in_shapes = self.perception_dict['action_logits'].out_shapes() + [self.obs_shapes["action_mask"]]
            self.perception_dict['action'] = ActionMaskingBlock(
                in_keys=['action_logits', "action_mask"], out_keys='action',
                in_shapes=in_shapes, num_actors=1, num_of_actor_actions=None
            )

        # Set up inference block
        self.perception_net = self.build_inference_block(list(action_logits_shapes.keys()))

        self.perception_net.apply(make_module_init_normc(1.0))
        self.perception_dict[out_key].apply(make_module_init_normc(0.01))

    def forward(self, tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute forward pass through the network.

        :param tensor_dict: input dict.
        :return: the computed output of the network.
        """
        return self.perception_net(tensor_dict)
