import torch
from torch.nn import functional as F


class Gate(torch.nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 gate_activation=torch.sigmoid):

        super(Gate, self).__init__()
        self.output_size = output_size

        self.gate_activation = gate_activation
        self.g = torch.nn.Linear(input_size, output_size)
        self.g1 = torch.nn.Linear(output_size, output_size, bias=False)
        self.g2 = torch.nn.Linear(input_size-output_size, output_size, bias=False)
        self.gate_bias = torch.nn.Parameter(torch.zeros(output_size))

    def forward(self, x_ent, x_lit):
        x = torch.cat([x_ent, x_lit], 1)
        g_embedded = F.tanh(self.g(x))
        gate = self.gate_activation(self.g1(x_ent) + self.g2(x_lit) + self.gate_bias)
        output = (1-gate) * x_ent + gate * g_embedded

        return output


class SingleGate(torch.nn.Module):

    def __init__(self, emb_size, lit_size, gate_activation=torch.sigmoid):
        super(SingleGate, self).__init__()

        self.emb_size = emb_size
        self.lit_size = lit_size

        self.gate_activation = gate_activation
        # self.g = torch.nn.Linear(emb_size + num_lit_size + txt_lit_size, emb_size)
        self.g = torch.nn.Linear(emb_size + lit_size * emb_size, emb_size)

        self.gate_ent = torch.nn.Linear(emb_size, emb_size, bias=False)
        self.gate_lit = torch.nn.Linear(lit_size * emb_size, emb_size, bias=False)
        self.gate_bias = torch.nn.Parameter(torch.zeros(emb_size))

    def forward(self, x_ent, x_lit):
        x_ent = x_ent.float()
        x_lit = torch.flatten(x_lit, start_dim=1).float()
        x = torch.cat([x_ent, x_lit], 1)
        g_embedded = F.tanh(self.g(x))
        gate = self.gate_activation(self.gate_ent(x_ent) + self.gate_lit(x_lit) + self.gate_bias)
        output = (1-gate) * x_ent + gate * g_embedded

        return output

class MultiGate(torch.nn.Module):

    def __init__(self, emb_size, num_lit_size, txt_lit_size, gate_activation=torch.sigmoid):
        super(MultiGate, self).__init__()

        self.emb_size = emb_size
        self.num_lit_size = num_lit_size
        self.txt_lit_size = txt_lit_size

        self.gate_activation = gate_activation
        # self.g = torch.nn.Linear(emb_size + num_lit_size + txt_lit_size, emb_size)
        self.g = torch.nn.Linear(emb_size + num_lit_size * emb_size + txt_lit_size * emb_size, emb_size)

        self.gate_ent = torch.nn.Linear(emb_size, emb_size, bias=False)
        self.gate_num_lit = torch.nn.Linear(num_lit_size * emb_size, emb_size, bias=False)
        self.gate_txt_lit = torch.nn.Linear(txt_lit_size * emb_size, emb_size, bias=False)
        self.gate_bias = torch.nn.Parameter(torch.zeros(emb_size))

    def forward(self, x_ent, x_lit_num, x_lit_txt):
        x_ent = x_ent.float()
        x_lit_num = torch.flatten(x_lit_num, start_dim=1).float()
        x_lit_txt = torch.flatten(x_lit_txt, start_dim=1).float()
        x = torch.cat([x_ent, x_lit_num, x_lit_txt], 1)
        g_embedded = F.tanh(self.g(x))
        gate = self.gate_activation(self.gate_ent(x_ent)
                                    + self.gate_num_lit(x_lit_num)
                                    + self.gate_txt_lit(x_lit_txt)
                                    + self.gate_bias)
        output = (1-gate) * x_ent + gate * g_embedded

        return output