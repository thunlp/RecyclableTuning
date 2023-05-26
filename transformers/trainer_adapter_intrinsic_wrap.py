import torch
from torch import nn
from IPython import embed

class WrapModelCF:
    def __init__(self, module: nn.Module, args, model1, model2, intrinsic, device="cpu"):
        super().__init__()
        self.args = args
        self.device = device
        # model1 is the teacher
        model_dict1 = [v for v in model1.state_dict().items()][1:]
        # model2 is the base model for evaluation
        model_dict2 = [v for v in model2.state_dict().items()][1:]

        flat1 = torch.tensor([]).cuda()
        flat1_nograd = torch.tensor([]).cuda()
        for i in model_dict1:
            if "adapter" in i[0] or "classifier" in i[0]:
                flat1 = torch.cat((flat1, i[1].flatten().cuda()), dim=0)
            else:
                flat1_nograd = torch.cat((flat1_nograd, i[1].flatten().cuda()), dim=0)
        
        flat2 = torch.tensor([]).cuda()
        flat2_nograd = torch.tensor([]).cuda()
        for i in model_dict2:
            if "adapter" in i[0] or "classifier" in i[0]:
                flat2 = torch.cat((flat2, i[1].flatten().cuda()), dim=0)
            else:
                flat2_nograd = torch.cat((flat2_nograd, i[1].flatten().cuda()), dim=0)

        self.end_w = torch.stack([flat1, flat2],dim=0)
        self.end_w.requires_grad = False
        self.end_w_nograd = torch.stack([flat1_nograd, flat2_nograd],dim=0)
        self.end_w.requires_grad = False

        self.flatten_size = self.end_w.size()[1]
        self.flatten_size_nograd = self.end_w_nograd.size()[1]
        self.name_base_localname = []
        self.name_base_localname_nograd = []
        
        # 随机初始化投影矩阵
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.flatten_size, intrinsic)).cuda())
        self.W2 = nn.Parameter(torch.normal(0, 0.02, size=(intrinsic, self.flatten_size)).cuda())
        self.bias = nn.Parameter(torch.normal(0, 0.02, size=(self.flatten_size,)).cuda())
        
        model_dict = [v for v in module.state_dict().items()][1:]
        flat = torch.tensor([]).cuda()
        names = []
        flat_nograd = torch.tensor([]).cuda()
        names_nograd = []
        for i in model_dict:
            if "adapter" in i[0] or "classifier" in i[0]:
                flat = torch.cat((flat, i[1].flatten().cuda()), dim=0)
                names.append(i[0])
            else:
                flat_nograd = torch.cat((flat_nograd, i[1].flatten().cuda()), dim=0)
                names_nograd.append(i[0])
        
        # train_itp_grad used as the base of the adapter
        self.train_itp_grad = flat
        self.train_itp_nograd = flat_nograd
        self.init_state = module.state_dict()

        module.register_parameter(
            "W1", self.W1)
        setattr(module, "W1", self.W1)
        module.register_parameter(
            "W2", self.W2)
        setattr(module, "W2", self.W2)
        module.register_parameter(
            "bias", self.bias)
        setattr(module, "bias", self.bias)
        
        self.W1.requires_grad = True
        self.W2.requires_grad = True
        self.bias.requires_grad = True
        self.train_itp_grad.requires_grad = False
        self.train_itp_nograd.requires_grad = False
        
        for name in names:
            base, localname = module, name
            while "." in localname:
                prefix, localname = localname.split(".", 1)
                base = base.__getattr__(prefix)
            self.name_base_localname.append((name, base, localname))
        
        for name in names_nograd:
            base, localname = module, name
            while "." in localname:
                prefix, localname = localname.split(".", 1)
                base = base.__getattr__(prefix)
            self.name_base_localname_nograd.append((name, base, localname))
    
    
    def __call__(self, module, *inputs, **kwargs):
        
        x = -1
        if "x" in kwargs:
            x = kwargs['x']
            kwargs.pop('x')
            x = float(x)

        if x == -1:
            # print("Attention! Now in pre-forward hook of evaluation without grad function!")
            with torch.no_grad():
                # x = -1 means doing evaluation mode! load the end_w[1] as the base model
                pos = 0
                flatten_pet = self.end_w[1] @ self.W1 @ self.W2 + self.bias     
                for name, base, localname in self.name_base_localname:
                    length = self.init_state[name].numel()
                    shape = self.init_state[name].size()
                    param = flatten_pet[pos:pos+length].view(shape)
                    delattr(base, localname)
                    setattr(base, localname, param)
                    pos = pos + length
                assert pos == len(flatten_pet)
                pos = 0
                flatten_pet = self.train_itp_nograd        
                for name, base, localname in self.name_base_localname_nograd:
                    length = self.init_state[name].numel()
                    shape = self.init_state[name].size()
                    param = flatten_pet[pos:pos+length].view(shape)
                    delattr(base, localname)
                    setattr(base, localname, param)
                    pos = pos + length
                assert pos == len(flatten_pet)
                return

        fix_model = 1
        if "fix_model" in kwargs:
            fix_model = kwargs['fix_model']
            kwargs.pop('fix_model')
            fix_model = int(fix_model)

        # interpolation!
        pos = 0
        train_itp = self.end_w[0] @ self.W1 @ self.W2 + self.bias
        flatten_pet = x * train_itp + (1 - x) * self.end_w[0]
        with torch.enable_grad():
            for name, base, localname in self.name_base_localname:
                length = self.init_state[name].numel()
                shape = self.init_state[name].size()
                param = flatten_pet[pos:pos+length].view(shape)
                delattr(base, localname)
                setattr(base, localname, param)
                pos = pos + length
            assert pos == len(flatten_pet)
        pos = 0
        flatten_pet = x * self.train_itp_nograd + (1 - x) * self.end_w_nograd[0]
        with torch.no_grad():
            for name, base, localname in self.name_base_localname_nograd:
                length = self.init_state[name].numel()
                shape = self.init_state[name].size()
                param = flatten_pet[pos:pos+length].view(shape)
                delattr(base, localname)
                setattr(base, localname, param)
                pos = pos + length
            assert pos == len(flatten_pet)

    @staticmethod
    def apply(module, args, model1, model2, intrinsic, device="cpu"):
        for k, hook in module._forward_pre_hooks.items():
            assert False
            if isinstance(hook, WrapModelCF) and hook.name == name:
                raise RuntimeError("Cannot register two intrinsic dimension hooks on "
                                   "the same parameter {}".format(name))
        fn = WrapModelCF(
            module, args, model1, model2, intrinsic, device)
        module.register_forward_pre_hook(fn)
        return fn
    
def itp_find_intrinsic(module, args, model1, model2, intrinsic=16, device="cpu"):
    ID_wrap = WrapModelCF.apply(
        module, args, model1, model2, intrinsic, device)
    return module, ID_wrap
