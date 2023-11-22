import torch
import numpy as np

from utils.darcy_loss import LpLoss

class PhysicsLoss(LpLoss):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(PhysicsLoss, self).__init__(d, p, size_average, reduction)

    def rel(self, x, y):
        num_examples = x.size()[0]

        physics_norms = self.physics(y, num_examples)

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        diff_norms += physics_norms
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def physics(self, data, num_examples):
        # Retrieve variables from the data
        u, w, v = data[idx_u], data[idx_w], data[idx_v]
        x, y, p = data[idx_x], data[idx_y], data[idx_p]
        phi = data[idx_phi]
        t = data[idx_t]
        lat = data[idx_lat]

        # Coriolis force term
        Omega = 7.3E-5
        f = 2*Omega*np.sin(np.radians(lat))

        # Calculate the derivatives (gradients) in the physics equations
        du_dx = np.gradient(u, x)
        du_dy = np.gradient(u, y)
        du_dp = np.gradient(u, p)
        du_dt = np.gradient(u, t)

        dw_dy = np.gradient(w, y)

        dv_dx = np.gradient(v, x)
        dv_dy = np.gradient(v, y)
        dv_dp = np.gradient(v, p)
        dv_dt = np.gradient(v, t)

        dphi_dx = np.gradient(phi, x)
        dphi_dy = np.gradient(phi, y)

        Du_Dt = du_dt + u*du_dx + v*du_dy + w*du_dp
        Dv_Dt = dv_dt + u*dv_dx + v*dv_dy + w*dv_dp

        # Solve physics equations
        lt1 = torch.norm(du_dx.reshape(num_examples,-1) + dw_dy.reshape(num_examples,-1) + dv_dp.reshape(num_examples,-1), self.p, 1)

        lt2 = torch.norm(Du_Dt.reshape(num_examples,-1) - (f*v).reshape(num_examples,-1) + dphi_dx.reshape(num_examples,-1), self.p, 1)
        lt3 = torch.norm(Dv_Dt.reshape(num_examples,-1) + (f*u).reshape(num_examples,-1) + dphi_dy.reshape(num_examples,-1), self.p, 1)

        # Find the physics-imposed loss term
        return (lt1 + lt2 + lt3)

    def __call__(self, x, y):
        return self.rel(x, y)