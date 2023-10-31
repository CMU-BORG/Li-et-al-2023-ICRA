# Copyright 2020-2021 Mathias Lechner
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import numpy as np


class CTRNNCell(nn.Module):
    def __init__(
        self,
        wiring,
        in_features=None,
        input_mapping="affine",
        output_mapping="affine",
        ode_unfolds=6,
        epsilon=1e-8,
    ):
        super(CTRNNCell, self).__init__()
        if in_features is not None:
            wiring.build((None, in_features))
        if not wiring.is_built():
            raise ValueError(
                "Wiring error! Unknown number of input features. Please pass the parameter 'in_features' or call the 'wiring.build()'."
            )
        self._init_ranges = {
            "tau": (0.001, 2.0),
            #"tau": (0.0, 0.0),
            "b": (-0.2, 0.2),
            "sigma": (3, 8),
            "mu": (0.3, 0.8),
            "sensory_sigma": (3, 8),
            "sensory_mu": (0.3, 0.8),
        }
        self._wiring = wiring
        self._input_mapping = input_mapping
        self._output_mapping = output_mapping
        self._ode_unfolds = ode_unfolds
        self._epsilon = epsilon
        self._allocate_parameters()

    @property
    def state_size(self):
        return self._wiring.units

    @property
    def sensory_size(self):
        return self._wiring.input_dim

    @property
    def motor_size(self):
        return self._wiring.output_dim

    @property
    def output_size(self):
        return self.motor_size

    @property
    def synapse_count(self):
        return np.sum(np.abs(self._wiring.adjacency_matrix))

    @property
    def sensory_synapse_count(self):
        return np.sum(np.abs(self._wiring.adjacency_matrix))

    def add_weight(self, name, init_value):
        param = torch.nn.Parameter(init_value)
        self.register_parameter(name, param)
        return param

    def add_constant(self, name, init_value):
        param = init_value.requires_grad_(False)
        return param

    def _get_init_value(self, shape, param_name):
        minval, maxval = self._init_ranges[param_name]
        if minval == maxval:
            return torch.ones(shape) * minval
        else:
            return torch.rand(*shape) * (maxval - minval) + minval

    def _allocate_parameters(self):
        print("alloc!")
        self._params = {}
        self._params["tau"] = self.add_weight(
            name="tau", init_value=self._get_init_value((self.state_size,), "tau")
        )
        self._params["b"] = self.add_weight(
            name="b", init_value=self._get_init_value((self.state_size,), "b")
        )
        self._params["sigma"] = self.add_weight(
            name="sigma",
            init_value=self._get_init_value(
                (self.state_size, self.state_size), "sigma"
            ),
        )
        self._params["mu"] = self.add_weight(
            name="mu",
            init_value=self._get_init_value((self.state_size, self.state_size), "mu"),
        )
        self._params["erev"] = self.add_weight(
            name="erev",
            init_value=torch.Tensor(self._wiring.erev_initializer()),
        )
        self._params["sensory_sigma"] = self.add_weight(
            name="sensory_sigma",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size), "sensory_sigma"
            ),
        )
        self._params["sensory_mu"] = self.add_weight(
            name="sensory_mu",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size), "sensory_mu"
            ),
        )
        self._params["sensory_erev"] = self.add_weight(
            name="sensory_erev",
            init_value=torch.Tensor(self._wiring.sensory_erev_initializer()),
        )

        self._params["sparsity_mask"] = torch.Tensor(
            np.abs(self._wiring.adjacency_matrix)
        )
        self._params["sensory_sparsity_mask"] = torch.Tensor(
            np.abs(self._wiring.sensory_adjacency_matrix)
        )

        if self._input_mapping in ["affine", "linear"]:
            self._params["input_w"] = self.add_weight(
                name="input_w",
                init_value=torch.ones((self.sensory_size,)),
            )
        if self._input_mapping == "affine":
            self._params["input_b"] = self.add_weight(
                name="input_b",
                init_value=torch.zeros((self.sensory_size,)),
            )

        if self._output_mapping in ["affine", "linear"]:
            self._params["output_w"] = self.add_weight(
                name="output_w",
                init_value=torch.ones((self.motor_size,)),
            )
        if self._output_mapping == "affine":
            self._params["output_b"] = self.add_weight(
                name="output_b",
                init_value=torch.zeros((self.motor_size,)),
            )

    def _sigmoid(self, v_pre, mu, sigma):
        v_pre = torch.unsqueeze(v_pre, -1)  # For broadcasting
        mues = v_pre - mu
        x = sigma * mues
        return torch.sigmoid(x)
        #v_pre = torch.unsqueeze(v_pre, -1)  # For broadcasting
        #num = v_pre - mu + sigma
        #den = 2*sigma
        #output = num/den
        #upthre = torch.ones_like(output)
        #lowthre = torch.zeros_like(output)
        #return torch.min(torch.max(output,lowthre),upthre)

    def _ode_solver(self, inputs, state, elapsed_time):
        v_pre = state

        # time constant term is loop invariant
        delta = elapsed_time / self._ode_unfolds
        q = delta / (self._params["tau"] + delta)
        ft = (1-q)*v_pre
        # We can pre-compute the effects of the sensory neurons here
        sensory_matrix = self._sigmoid(
            inputs, self._params["sensory_mu"], self._params["sensory_sigma"]
        )

        sensory_rev_activation = self._params["sensory_erev"] * sensory_matrix
        sensory_rev_activation *= self._params["sensory_sparsity_mask"]

        # Reduce over dimension 1 (=source sensory neurons)
        w_sensory = torch.sum(sensory_rev_activation, dim=1)

        # Unfold the multiply ODE multiple times into one RNN step
        for t in range(self._ode_unfolds):
            state_matrix = self._sigmoid(
                v_pre, self._params["mu"], self._params["sigma"]
            )

            rev_activation = self._params["erev"]*state_matrix
            rev_activation *= self._params["sparsity_mask"]

            # Reduce over dimension 1 (=source neurons)
            sum_input = torch.sum(rev_activation, dim=1) + w_sensory + self._params["b"]

        return ft + q*sum_input

    def _map_inputs(self, inputs):
        if self._input_mapping in ["affine", "linear"]:
            inputs = inputs * self._params["input_w"]
        if self._input_mapping == "affine":
            inputs = inputs + self._params["input_b"]
        return inputs

    def _map_outputs(self, state):
        output = state
        if self.motor_size < self.state_size:
            output = output[:, 0 : self.motor_size]  # slice

        if self._output_mapping in ["affine", "linear"]:
            output = output * self._params["output_w"]
        if self._output_mapping == "affine":
            output = output + self._params["output_b"]
        return output

    def _clip(self, w):
        return torch.nn.ReLU()(w)

    def apply_weight_constraints(self):
        self._params["tau"].data = self._clip(self._params["tau"].data)

    def forward(self, inputs, states):
        # Regularly sampled mode (elapsed time = 1 second)
        elapsed_time = 1.0
        inputs = self._map_inputs(inputs)

        next_state = self._ode_solver(inputs, states, elapsed_time)

        outputs = self._map_outputs(next_state)

        return outputs, next_state