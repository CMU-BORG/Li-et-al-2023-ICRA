import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
from .SNS_layer import SNS_layer, SENSORY_LAYER_1_INPUT_SIZE, SENSORY_LAYER_1_SIZE, SENSORY_LAYER_2_INPUT_SIZE, SENSORY_LAYER_2_SIZE, THETA_MAX, THETA_MIN, F_MAX, F_MIN, sensory_layer_1, R

#########################################################

SEQUENCE_LENGTH = 240
TRAINING_EXAMPLES = 5000
BATCH_SIZE = 100
BATCH_NUM = int(TRAINING_EXAMPLES / BATCH_SIZE)
EPOCHS = 30
RATIO = 0.3
STD = [0.08/2, 0.01/2]
MU = [5, 220]
LR = 0.1

#########################################################

class SNS_Sensory_Layer(nn.Module):
    def __init__(
        self,
        sensory_layer_1,
        sensory_layer_2,
        output_mu,
        output_sigma
    ):
        super(SNS_Sensory_Layer, self).__init__()
        self._sensory_layer_1 = sensory_layer_1
        self._sensory_layer_2 = sensory_layer_2
        self._output_mu = torch.nn.Parameter(output_mu)
        self._output_sigma = torch.nn.Parameter(output_sigma)
        self.register_parameter("output_mu", self._output_mu)
        self.register_parameter("output_sigma", self._output_sigma)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        sensory_layer_1_state = torch.zeros((batch_size, self._sensory_layer_1.state_size))
        sensory_layer_2_state = torch.zeros((batch_size, self._sensory_layer_2.state_size))
        outputs = []
        for t in range(seq_len):
            position_inputs = x[:, t, :-1]
            force_inputs = x[:, t, -1].reshape(-1, 1)
            _, sensory_layer_1_state = self._sensory_layer_1.forward(position_inputs, sensory_layer_1_state)
            output, sensory_layer_2_state = self._sensory_layer_2.forward(torch.cat((sensory_layer_1_state, force_inputs), dim=1), sensory_layer_2_state)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)

        return self.sigmoid(outputs)

    def sigmoid(self, outputs):
        num = outputs - self._output_mu + self._output_sigma
        den = 2*self._output_sigma
        new_outputs = num/den
        upthre = torch.ones_like(new_outputs)
        lowthre = torch.zeros_like(new_outputs)
        return torch.minimum(torch.maximum(new_outputs, lowthre), upthre)

def training_dataloader(in_features=SENSORY_LAYER_1_INPUT_SIZE + 1, out_features=SENSORY_LAYER_2_SIZE, N=SEQUENCE_LENGTH, batch_size=BATCH_SIZE, training_examples=TRAINING_EXAMPLES, ratio=RATIO, std=STD, mu=MU, theta_max=THETA_MAX, theta_min=THETA_MIN, f_max=F_MAX, f_min=F_MIN):

    data_x = []
    data_y = []
    position_max = theta_max[0:3]
    position_min = theta_min[0:3]
    force_max = f_max
    force_min = f_min
    position_upper_bound = np.maximum(position_max, position_min)
    position_lower_bound = np.minimum(position_max, position_min)
    for i in range(training_examples):
        joint_feedback = position_min + np.random.rand(3) * (position_max - position_min)
        if np.random.rand(1) < ratio:
            object_position = position_min + np.random.rand(3) * (position_max - position_min)
            target_position = position_min + np.random.rand(3) * (position_max - position_min)
            force_feedback = force_min + np.random.rand(1) * (force_max - force_min)
        else:
            std_selected = np.random.choice(std)
            object_position = joint_feedback + np.random.normal(0, std_selected, 3) * np.abs(position_max - position_min)
            target_position = joint_feedback + np.random.normal(0, std_selected, 3) * np.abs(position_max - position_min)
            object_position = np.maximum(np.minimum(object_position, position_upper_bound), position_lower_bound)
            target_position = np.maximum(np.minimum(target_position, position_upper_bound), position_lower_bound)
            mu_selected = np.random.choice(mu)
            force_feedback = mu_selected + np.random.normal(0, 0.01/2, 1) * np.abs(force_max - force_min)
            force_feedback = np.maximum(np.minimum(force_feedback, force_max), force_min)
        sensory_feedback = np.concatenate([joint_feedback, object_position, target_position, force_feedback])
        data_x.append(np.repeat(sensory_feedback.reshape(1, in_features), N, axis=0))

        obj_err = np.abs(joint_feedback - object_position)
        target_err = np.abs(joint_feedback - target_position)
        move_to_pre_grasp = (np.sum(obj_err) > 0.08) * 1.0
        move_to_pre_target = (np.sum(target_err) > 0.08) * 1.0
        move_to_grasp = (np.sum(obj_err) > 0.01) * 1.0
        move_to_target = (np.sum(target_err) > 0.01) * 1.0
        grasped = (force_feedback > 220) * 1.0
        released = (force_feedback > 5) * 1.0
        sensory_layer_2_out = np.array([move_to_pre_grasp, move_to_grasp, move_to_pre_target, move_to_target, grasped, released])
        data_y.append(np.repeat(sensory_layer_2_out.reshape(1, out_features), N, axis=0))

    data_x_training = torch.Tensor(data_x)
    data_y_training = torch.Tensor(data_y)
    print("data_x.size: ", str(data_x_training.size()))
    print("data_y.size: ", str(data_y_training.size()))
    dataloader = data.DataLoader(data.TensorDataset(data_x_training, data_y_training), batch_size=batch_size, shuffle=True, num_workers=0)

    return dataloader


# sensory_layer_2
sparsity_mask = np.zeros([SENSORY_LAYER_2_INPUT_SIZE, SENSORY_LAYER_2_SIZE], dtype=np.int32)
sparsity_mask[0:6, 0:2] = 1
sparsity_mask[6:12, 2:4] = 1
sparsity_mask[12, 4:6] = 1
theta_max_in = torch.Tensor(np.concatenate((R * np.ones(SENSORY_LAYER_1_SIZE), F_MAX)))
sensory_layer_2 = SNS_layer(layer_input_size=SENSORY_LAYER_2_INPUT_SIZE, layer_size=SENSORY_LAYER_2_SIZE, sparsity_mask=sparsity_mask, theta_max_in=theta_max_in, R=R)
sensory_layer_2._params["b"].requires_grad = True
sensory_layer_2._params["sensory_erev"].requires_grad = True
sensory_layer_2._params["sensory_w"].requires_grad = True
output_mu = torch.Tensor(10*np.ones(SENSORY_LAYER_2_SIZE))
output_sigma = torch.Tensor(10*np.ones(SENSORY_LAYER_2_SIZE))
SNS_sequence = SNS_Sensory_Layer(sensory_layer_1, sensory_layer_2, output_mu, output_sigma)

#########################################################
# Training
training_data = training_dataloader()
train_err = np.zeros([1, EPOCHS*BATCH_NUM])
optimizer = torch.optim.NAdam(SNS_sequence.parameters(), lr=LR)
def lambda1(epoch): return 0.9 ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
batch_index = 0
for epoch in tqdm(range(EPOCHS)):
    ave_err = 0
    for data_x, data_y in training_data:
        with torch.no_grad():
            y_hat = SNS_sequence.forward(data_x)
            y_hat = y_hat.view_as(data_y)
            loss = nn.MSELoss()(y_hat, data_y)
            train_err[0, batch_index] = loss.numpy()
            ave_err += train_err[0, batch_index]
            print(train_err[0, batch_index])

        def closure():
            y_hat = SNS_sequence.forward(data_x)
            y_hat = y_hat.view_as(data_y)
            loss = nn.MSELoss()(y_hat, data_y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(SNS_sequence.parameters(), 1e-2)
            return loss

        optimizer.step(closure)
        SNS_sequence._sensory_layer_1.apply_weight_constraints()
        SNS_sequence._sensory_layer_2.apply_weight_constraints()
        SNS_sequence._output_sigma.data = torch.maximum(SNS_sequence._output_sigma.data, 1e-3*torch.ones_like(SNS_sequence._output_sigma.data))
        batch_index += 1
    print('ave_err =', ave_err/BATCH_NUM)
    scheduler.step()

#torch.save(SNS_sequence.state_dict(), "sensory_layer_param")
#torch.save(perceptor._sensory_layer_1.state_dict(), "sensory_layer_1_param")
#torch.save(perceptor._sensory_layer_2.state_dict(), "sensory_layer_2_param")
#torch.save(perceptor._command_layer.state_dict(), "command_layer_param")