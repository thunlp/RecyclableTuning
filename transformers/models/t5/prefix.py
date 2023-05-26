import torch
import torch.nn as nn

# class PrefixSingleton(type):
#     _instances = {}
#     def __call__(cls, prefix_r, init_emb=None):
#         if cls not in cls._instances:
#             prefix = super(PrefixSingleton, cls).__call__(prefix_r, init_emb)
#             cls._instances[cls] = prefix
#         else:
#             assert prefix_r == cls._instances[cls].prefix_r
#         return cls._instances[cls]

# class PrefixBase(nn.Module, metaclass=PrefixSingleton):
#     def __init__(self, prefix_r, init_emb=None):
#         super().__init__()
#         self.prefix_r = prefix_r
#         if init_emb is None:
#             self.weight = torch.nn.Parameter(torch.empty(1, prefix_r))
#             self.reset_parameters()
#         else:
#             self.weight = init_emb

#     def reset_parameters(self):
#         nn.init.normal_(self.weight, std=0.02)

class IntrinsicPrefix(nn.Module):
    def __init__(self, intrinsic_dim, prefix_num, num_layers, d_model, head_num, init_emb=None, bias=True):
        super().__init__()
        self.intrinsic_dim = intrinsic_dim
        self.num_layers = num_layers
        self.prefix_num = prefix_num
        self.d_model = d_model
        self.head_num = head_num
        self.share_prefix = init_emb is not None
        self.hyper_prefix_project = nn.Linear(intrinsic_dim, 2 * num_layers * prefix_num * d_model, bias=bias)

        if init_emb is not None:
            self.hyper_prefix = init_emb
        else:
            self.hyper_prefix = nn.Parameter(torch.empty(1, intrinsic_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.hyper_prefix_project.weight, std=0.02)
        if self.hyper_prefix_project.bias is not None:
            nn.init.zeros_(self.hyper_prefix_project.bias)
        if not self.share_prefix:
            nn.init.normal_(self.hyper_prefix, std=0.02)

    def get_prefix(self):
        return self.hyper_prefix_project(self.hyper_prefix.view(1, -1)).view(self.num_layers, 2, self.head_num, self.prefix_num, -1)

class OriginalPrefix(nn.Module):
    def __init__(self, prefix_r, prefix_num, num_layers, d_model, head_num, init_emb=None):
        super().__init__()
        self.prefix_r = prefix_r
        self.prefix_num = prefix_num
        self.num_layers = num_layers
        self.d_model = d_model
        self.head_num = head_num
        self.prefix_project = nn.Sequential(
            nn.Linear(d_model, prefix_r),
            nn.Tanh(),
            nn.Linear(prefix_r, num_layers * d_model * 2)
        )

        if init_emb is None:
            self.prefix = torch.nn.Parameter(torch.empty(prefix_num, self.prefix_r))
        else:
            self.prefix = init_emb
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.prefix, std=0.02)

    def get_prefix(self):
        return self.prefix_project(self.prefix).view(self.num_layers, 2, self.head_num, self.prefix_num, -1)

if __name__ == '__main__':
    prefix1 = IntrinsicPrefix(8, 16, 768)
    prefix2 = IntrinsicPrefix(8, 16, 768)
    assert prefix1.prefix is prefix2.prefix
    print("Pass")
