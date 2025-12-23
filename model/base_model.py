import torch
from torch import nn
import torch.nn.functional as F
class DragonNetBase(nn.Module):
    """
    Parameters
    ----------
    input_dim: int
        input dimension for convariates
    shared_hidden: int
        layer size for hidden shared representation layers
    outcome_hidden: int
        layer size for conditional outcome layers
    """
    def __init__(self, input_dim, shared_hidden=200, outcome_hidden=100):
        super(DragonNetBase, self).__init__()
        # NOTE: Shared representation layers - Dragon Body
        self.full_connect_1 = nn.Linear(in_features=input_dim, out_features=shared_hidden)
        self.full_connect_2 = nn.Linear(in_features=shared_hidden, out_features=shared_hidden)
        self.full_connect_3 = nn.Linear(in_features=shared_hidden, out_features=shared_hidden)

        # NOTE: Output of the Dragon Body
        self.treat_out = nn.Linear(in_features=shared_hidden, out_features=1)
        #---------------------------------------------------#

        # NOTE: Prediction heads - 1st Dragon Head - Control
        self.control_head_full_connect_1 = nn.Linear(in_features=shared_hidden, out_features=outcome_hidden)
        self.control_head_full_connect_2 = nn.Linear(in_features=outcome_hidden, out_features=outcome_hidden)
        self.control_head_full_connect_out = nn.Linear(in_features=outcome_hidden, out_features=1)

        # NOTE: Prediction heads - 2nd Dragon Head - Treatment
        self.treatment_head_full_connect_1 = nn.Linear(in_features=shared_hidden, out_features=outcome_hidden)
        self.treatment_head_full_connect_2 = nn.Linear(in_features=outcome_hidden, out_features=outcome_hidden)
        self.treatment_head_full_connect_out = nn.Linear(in_features=outcome_hidden, out_features=1)

        # NOTE: Propensity score head - 3rd Dragon Head - uses linear epsilon
        self.epsilon = nn.Linear(in_features=1, out_features=1)
        torch.nn.init.xavier_normal_(self.epsilon.weight)

    def forward(self, inputs):
        """
        forward method to train model.

        Parameters
        ----------
        inputs: torch.Tensor
            covariates

        Returns
        -------
        y0: torch.Tensor
            outcome under control
        y1: torch.Tensor
            outcome under treatment
        t_pred: torch.Tensor
            predicted treatment
        eps: torch.Tensor
            trainable epsilon parameter
        """
        #shared layer
        x = F.elu(self.full_connect_1(inputs))
        x = F.elu(self.full_connect_2(x))
        z = F.elu(self.full_connect_3(x))

        #propensity
        t_pred = torch.sigmoid(self.treat_out(z))

        y0 = F.elu(self.control_head_full_connect_1(z))
        y0 = F.elu(self.control_head_full_connect_2(y0))
        y0 = self.control_head_full_connect_out(y0)

        y1 = F.elu(self.treatment_head_full_connect_1(z))
        y1 = F.elu(self.treatment_head_full_connect_2(y1))
        y1 = self.treatment_head_full_connect_out(y1)

        eps = self.epsilon(torch.ones_like(t_pred)[:, 0:1])

        return y0, y1, t_pred, eps