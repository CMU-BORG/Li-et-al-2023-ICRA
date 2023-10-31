import numpy as np
import torch
import torch.nn as nn
from . import torchSNS as tSNS
from .torchSNS.torch import SNSCell

#########################################################

THETA_MAX = np.array([0.2, 0.2, -1, 0.5, 0.5])
THETA_MIN = np.array([-0.2, -0.2, 0, -0.5, -0.5])
F_MAX = np.array([300])
F_MIN = np.array([0])
SENSORY_LAYER_1_INPUT_SIZE = 9
SENSORY_LAYER_1_SIZE = 12
SENSORY_LAYER_2_INPUT_SIZE = 13
SENSORY_LAYER_2_SIZE = 6
COMMAND_LAYER_INPUT_SIZE = 6
COMMAND_LAYER_SIZE = 8
INTER_LAYER_1_INPUT_SIZE = 8
INTER_LAYER_1_SIZE = 4
INTER_LAYER_2_INPUT_SIZE = 12
INTER_LAYER_2_SIZE = 10
MOTOR_LAYER_INPUT_SIZE = 12
MOTOR_LAYER_SIZE = 5
R = 20

#########################################################


def layer_initialization(layer, theta_min_in, theta_max_in, theta_min_out, theta_max_out, tau, b, sensory_erev, sensory_w, R):
    # Parameter initialization
    in_features = layer._wiring.input_dim
    out_features = layer._wiring.output_dim
    neuron_num = layer._wiring.units
    with torch.no_grad():
        layer._params["tau"].requires_grad = False
        layer._params["b"].requires_grad = False
        layer._params["input_w"].requires_grad = False
        layer._params["input_b"].requires_grad = False
        layer._params["output_w"].requires_grad = False
        layer._params["output_b"].requires_grad = False
        layer._params["sensory_mu"].requires_grad = False
        layer._params["sensory_sigma"].requires_grad = False
        layer._params["mu"].requires_grad = False
        layer._params["sigma"].requires_grad = False
        layer._params["sensory_erev"].requires_grad = False
        layer._params["sensory_w"].requires_grad = False
        layer._params["erev"].requires_grad = False
        layer._params["w"].requires_grad = False

        layer._params["tau"].data = tau
        layer._params["b"].data = b
        layer._params["input_w"].data = torch.Tensor(
            R / (theta_max_in - theta_min_in))
        layer._params["input_b"].data = torch.Tensor(
            R * theta_min_in / (theta_min_in - theta_max_in))
        layer._params["output_w"].data = torch.Tensor(
            (theta_max_out - theta_min_out) / R)
        layer._params["output_b"].data = torch.Tensor(theta_min_out)
        layer._params["sensory_mu"].data = R/2 * \
            torch.ones((in_features, neuron_num))
        layer._params["sensory_sigma"].data = R / \
            2 * torch.ones((in_features, neuron_num))
        layer._params["mu"].data = R/2 * torch.ones((neuron_num, neuron_num))
        layer._params["sigma"].data = R/2 * \
            torch.ones((neuron_num, neuron_num))
        layer._params["sensory_erev"].data = sensory_erev
        layer._params["sensory_w"].data = sensory_w
        layer._params["erev"].data = torch.zeros((neuron_num, neuron_num))
        layer._params["w"].data = torch.zeros((neuron_num, neuron_num))


def SNS_layer(layer_input_size, layer_size, sparsity_mask, tau=None, theta_min_in=None, theta_max_in=None, theta_min_out=None, theta_max_out=None, R=20):
    # Create a SNS layer with layer_input_size input neurons and layer_size neurons. sparsity_mask reflects sparsity in the input synaptic connections.
    if tau is None:
        tau = torch.zeros(layer_size)
    if theta_min_in is None:
        theta_min_in = torch.zeros(layer_input_size)
    if theta_max_in is None:
        theta_max_in = R*torch.ones(layer_input_size)
    if theta_min_out is None:
        theta_min_out = torch.zeros(layer_size)
    if theta_max_out is None:
        theta_max_out = R*torch.ones(layer_size)
    wiring = tSNS.wirings.FullyConnected(
        layer_size, layer_size, self_connections=False, erev_init_seed=np.random.randint(0, 10000))
    layer = SNSCell(wiring, layer_input_size)
    config = layer._wiring.get_config()
    config["adjacency_matrix"] = np.zeros([layer_size, layer_size])
    config["sensory_adjacency_matrix"] = sparsity_mask * \
        config["sensory_adjacency_matrix"]
    new_wiring = tSNS.wirings.Wiring.from_config(config)
    layer = SNSCell(new_wiring, layer_input_size,
                    ode_unfolds=1, elapsed_time=1/240)
    b = torch.Tensor(R * theta_min_out / (theta_min_out - theta_max_out))
    sensory_erev = (R - b).reshape(1, -
                                   1).repeat(layer_input_size, 1) * torch.Tensor(sparsity_mask)
    sensory_w = torch.zeros((layer_input_size, layer_size))
    layer_initialization(layer, theta_min_in=theta_min_in, theta_max_in=theta_max_in, theta_min_out=theta_min_out,
                         theta_max_out=theta_max_out, tau=tau, b=b, sensory_erev=sensory_erev, sensory_w=sensory_w, R=R)

    return layer


# control and perception network
class SNS_Control(nn.Module):
    def __init__(
        self,
        inter_layer_1,
        inter_layer_2,
        motor_layer
    ):
        super(SNS_Control, self).__init__()
        self._inter_layer_1 = inter_layer_1
        self._inter_layer_2 = inter_layer_2
        self._motor_layer = motor_layer
        self._inter_layer_1_state = torch.zeros(
            (1, self._inter_layer_1.state_size))
        self._inter_layer_2_state = torch.zeros(
            (1, self._inter_layer_2.state_size))
        self._motor_layer_state = torch.Tensor([R/2, R/2, 0, R/2, R/2])

    def forward(self, object_position, target_position, input):
        _, self._inter_layer_1_state = self._inter_layer_1.forward(
            input, self._inter_layer_1_state)
        _, self._inter_layer_2_state = self._inter_layer_2.forward(torch.cat((object_position.repeat_interleave(2, dim=1)[
                                                                   :, :-1], target_position.repeat_interleave(2, dim=1)[:, :-1], self._inter_layer_1_state[:, :2]), dim=1), self._inter_layer_2_state)
        output, self._motor_layer_state = self._motor_layer.forward(torch.cat(
            (self._inter_layer_2_state, self._inter_layer_1_state[:, -2:]), dim=1), self._motor_layer_state)

        return output.squeeze(dim=0)

    def reset(self):
        self._inter_layer_1_state = torch.zeros(
            (1, self._inter_layer_1.state_size))
        self._inter_layer_2_state = torch.zeros(
            (1, self._inter_layer_2.state_size))
        self._motor_layer_state = torch.zeros(
            (1, self._motor_layer.state_size))


class SNS_Perception(nn.Module):
    def __init__(
        self,
        sensory_layer_1,
        sensory_layer_2,
        command_layer
    ):
        super(SNS_Perception, self).__init__()
        self._sensory_layer_1 = sensory_layer_1
        self._sensory_layer_2 = sensory_layer_2
        self._command_layer = command_layer
        self._sensory_layer_1_state = torch.zeros(
            (1, self._sensory_layer_1.state_size))
        self._sensory_layer_2_state = torch.zeros(
            (1, self._sensory_layer_2.state_size))
        self._command_layer_state = torch.zeros(
            (1, self._command_layer.state_size))

    def forward(self, gripper_position, object_position, target_position, force):
        position_input = torch.cat(
            (gripper_position, object_position, target_position), dim=1)
        _, self._sensory_layer_1_state = self._sensory_layer_1.forward(
            position_input, self._sensory_layer_1_state)
        sensory_layer_2_input = torch.cat(
            (self._sensory_layer_1_state, force), dim=1)
        _, self._sensory_layer_2_state = self._sensory_layer_2.forward(
            sensory_layer_2_input, self._sensory_layer_2_state)
        output, self._command_layer_state = self._command_layer.forward(
            self._sensory_layer_2_state, self._command_layer_state)

        return output

    def reset(self):
        self._sensory_layer_1_state = torch.zeros(
            (1, self._sensory_layer_1.state_size))
        self._sensory_layer_2_state = torch.zeros(
            (1, self._sensory_layer_2.state_size))
        self._command_layer_state = torch.zeros(
            (1, self._command_layer.state_size))


# sensory_layer_1
sparsity_mask = np.zeros(
    [SENSORY_LAYER_1_INPUT_SIZE, SENSORY_LAYER_1_SIZE], dtype=np.int32)
for i in range(3):
    sparsity_mask[i, i] = 1
    sparsity_mask[i + 3, i] = -1
for i in range(3, 6):
    sparsity_mask[i - 3, i] = -1
    sparsity_mask[i, i] = 1
for i in range(6, 9):
    sparsity_mask[i - 6, i] = 1
    sparsity_mask[i, i] = -1
for i in range(9, 12):
    sparsity_mask[i - 9, i] = -1
    sparsity_mask[i - 3, i] = 1
theta_max_in = np.tile(THETA_MAX[:3], 3)
theta_min_in = np.tile(THETA_MIN[:3], 3)
sensory_layer_1 = SNS_layer(layer_input_size=SENSORY_LAYER_1_INPUT_SIZE, layer_size=SENSORY_LAYER_1_SIZE,
                            sparsity_mask=sparsity_mask, theta_min_in=theta_min_in, theta_max_in=theta_max_in, R=R)


# sensory_layer_2
sparsity_mask = np.zeros(
    [SENSORY_LAYER_2_INPUT_SIZE, SENSORY_LAYER_2_SIZE], dtype=np.int32)
sparsity_mask[0:6, 0:2] = 1
sparsity_mask[6:12, 2:4] = 1
sparsity_mask[12, 4:6] = 1
theta_max_in = torch.Tensor(np.concatenate(
    (R * np.ones(SENSORY_LAYER_1_SIZE), F_MAX)))
sensory_layer_2 = SNS_layer(layer_input_size=SENSORY_LAYER_2_INPUT_SIZE,
                            layer_size=SENSORY_LAYER_2_SIZE, sparsity_mask=sparsity_mask, theta_max_in=theta_max_in, R=R)
sensory_layer_2.load_state_dict(
    torch.load("controller\\sensory_layer_2_param"))


# command_layer
# neuron 0 move to the pregrasp position
# neuron 1 move to the grasp position
# neuron 2 grasp
# neuron 3 move to the postgrasp position
# neuron 4 move to the prerelease position
# neuron 5 move to the release position
# neuron 6 release
# neuron 7 move to the postrelease position
sparsity_mask = np.zeros(
    [COMMAND_LAYER_INPUT_SIZE, COMMAND_LAYER_SIZE], dtype=np.int32)
# far away from the object position + far away from the target position + no force = move to the pregrasp position
sparsity_mask[[0, 2, 5], 0] = [1, 1, -1]
# not far away from the object position + not very close to the object + no force = move to the grasp position
sparsity_mask[[0, 1, 5], 1] = [-1, 1, -1]
# very close to the object position + no force = grasp
sparsity_mask[[1, 5], 2] = [-1, -1]
# not far away from the object position + force = move to the postgrasp position
sparsity_mask[[0, 5], 3] = [-1, 1]
# far away from the object position + far away from the target position + force = move to the prerelease position
sparsity_mask[[0, 2, 5], 4] = [1, 1, 1]
# not far away from the target position + not very close to the target position + force = move to the release position
sparsity_mask[[2, 3, 5], 5] = [-1, 1, 1]
# very close to the target position + force = release
sparsity_mask[[3, 5], 6] = [-1, 1]
# far away from the target position + no force = move to the postrelease position
sparsity_mask[[2, 5], 7] = [-1, -1]
command_layer = SNS_layer(layer_input_size=COMMAND_LAYER_INPUT_SIZE,
                          layer_size=COMMAND_LAYER_SIZE, sparsity_mask=sparsity_mask, R=R)
command_layer.load_state_dict(torch.load("controller\\command_layer_param"))
command_layer._params["b"].data[0] = -R
command_layer._params["b"].data[2] = R
command_layer._params["b"].data[4] = -2 * R
command_layer._params["b"].data[5] = -R
command_layer._params["b"].data[7] = R


# inter_layer_1
# neuron 0 move to the object position
# neuron 1 move to the target position
# neuron 2 lift the gripper up
# neuron 3 open/close the gripper
sparsity_mask = np.zeros(
    [INTER_LAYER_1_INPUT_SIZE, INTER_LAYER_1_SIZE], dtype=np.int32)
# move to the pregrasp position = move to the object position + lift the gripper up + open the gripper
sparsity_mask[0, [0, 2, 3]] = [1, 1/2, 1]
# move to the grasp position = move to the object position + open the gripper
sparsity_mask[1, [0, 3]] = 1
# grasp the object = move to the object position + close the gripper
sparsity_mask[2, [0, 3]] = [1, 0]
# move to the postgrasp position = move to the object position + lift the gripper up + close the gripper
sparsity_mask[3, [0, 2, 3]] = [1, 1, 0]
# move to the prerelease position = move to the target position + lift the gripper up + close the gripper
sparsity_mask[4, [1, 2, 3]] = [1, 1/2, 0]
# move to the release position = move to the target position + close the gripper
sparsity_mask[5, [1, 3]] = [1, 0]
# release = move to the target position + open the gripper
sparsity_mask[6, [1, 3]] = 1
# move to the postrelease position = move to the target position + lift the gripper up + open the gripper
sparsity_mask[7, [1, 2, 3]] = 1
inter_layer_1 = SNS_layer(layer_input_size=INTER_LAYER_1_INPUT_SIZE,
                          layer_size=INTER_LAYER_1_SIZE, sparsity_mask=sparsity_mask, R=R)


# inter_layer_2
# neuron 0/1 object_x(+-)
# neuron 2/3 target_x(+-)
# neuron 4/5 object_y(+-)
# neuron 6/7 target_y(+-)
# neuron 8 object_z
# neuron 9 target_z
sparsity_mask = np.zeros(
    [INTER_LAYER_2_INPUT_SIZE, INTER_LAYER_2_SIZE], dtype=np.int32)
for i in range(2):
    # positive object_x feedback = object_x(+), positive target_x feedback = target_x(+)
    sparsity_mask[2 * i, 4 * i] = 1
    # negative object_x feedback = object_x(-), negative target_x feedback = target_x(-)
    sparsity_mask[2 * i + 1, 4 * i + 1] = 1
    # positive object_y feedback = object_y(+), # positive target_y feedback = target_y(+)
    sparsity_mask[2 * i + 5, 4 * i + 2] = 1
    # negative object_y feedback = object_y(-), # negative target_y feedback = target_y(-)
    sparsity_mask[2 * i + 5 + 1, 4 * i + 2 + 1] = 1
    # move to the object position = target_x(+-), target_y(+-) inhibited
    sparsity_mask[10, [4 * i + 2, 4 * i + 2 + 1]] = -1
    # move to the target position = object_x(+-), object_y(+-) inhibited
    sparsity_mask[11, [4 * i, 4 * i + 1]] = -1
sparsity_mask[4, 8] = 1  # object_z feedback = object_z
sparsity_mask[9, 9] = 1  # target_z feedback = target_z
# move to the grasp position = target_z inhibited
sparsity_mask[10, 9] = -1
# move to the target position = object_z inhibited
sparsity_mask[11, 8] = -1
theta_max_in = np.concatenate((np.stack((THETA_MAX[0:3], THETA_MIN[0:3])).transpose().reshape(
    1, -1).squeeze()[:-1], np.stack((THETA_MAX[0:3], THETA_MIN[0:3])).transpose().reshape(1, -1).squeeze()[:-1], [R, R]))
theta_max_in = torch.Tensor(theta_max_in)
inter_layer_2 = SNS_layer(layer_input_size=INTER_LAYER_2_INPUT_SIZE, layer_size=INTER_LAYER_2_SIZE,
                          sparsity_mask=sparsity_mask, theta_max_in=theta_max_in, R=R)


# motor_layer
# neuron 0 x joint command
# neuron 1 y joint command
# neuron 2 z joint command
# neuron 3 left claw joint command
# neuron 4 right claw joint command
sparsity_mask = np.zeros(
    [MOTOR_LAYER_INPUT_SIZE, MOTOR_LAYER_SIZE], dtype=np.int32)
for i in range(2):  # object_x = x joint command, object_y = y joint command/target_x = x joint command, target_y = y joint command
    sparsity_mask[[4 * i, 4 * i + 2], i] = 1
    sparsity_mask[[4 * i + 1, 4 * i + 3], i] = -1
# object_z = z joint command/target_z = z joint command
sparsity_mask[8:10, 2] = 1
sparsity_mask[10, 2] = -1  # lift the gripper up = decrease z joint command
# open the gripper = negative left claw joint command + positive right claw joint command
sparsity_mask[11, [3, 4]] = [-1, 1]
theta_max_out = torch.Tensor(THETA_MAX)
theta_min_out = torch.Tensor(THETA_MIN)
tau = torch.Tensor([0.1, 0.3, 0.1, 0, 0])
motor_layer = SNS_layer(layer_input_size=MOTOR_LAYER_INPUT_SIZE, layer_size=MOTOR_LAYER_SIZE,
                        sparsity_mask=sparsity_mask, tau=tau, theta_min_out=theta_min_out, theta_max_out=theta_max_out, R=R)
motor_layer._params["sensory_erev"].data[10, 2] = -R / 10


perceptor = SNS_Perception(sensory_layer_1, sensory_layer_2, command_layer)
controller = SNS_Control(inter_layer_1, inter_layer_2, motor_layer)
perceptor.eval()
controller.eval()
