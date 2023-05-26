import torch
from torch import nn
from IPython import embed

class WrapModelCF:
    def __init__(self, module: nn.Module, args, model1, model2, device="cpu"):
        super().__init__()
        self.args = args
        self.device = device

        model_dict1 = [v for v in model1.state_dict().items()][1:]
        model_dict2 = [v for v in model2.state_dict().items()][1:]
        fix1_dict = [v for v in model1.state_dict().items()][1:]
        fix2_dict = [v for v in model1.state_dict().items()][1:]

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
        
        # module初始化展平得到train_itp
        self.train_itp = nn.Parameter(torch.zeros(self.flatten_size).cpu() if device=="cpu" else torch.zeros(self.flatten_size).cuda())
        self.train_itp_nograd = nn.Parameter(torch.zeros(self.flatten_size_nograd).cpu() if device=="cpu" else torch.zeros(self.flatten_size_nograd).cuda())
        
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
        
        # use roberta as init state
        self.train_itp.data = flat
        self.train_itp_nograd.data = flat_nograd
        self.init_state = module.state_dict()
        
        module.register_parameter(
            "train_itp", self.train_itp)
        setattr(module, "train_itp", self.train_itp)
        module.register_parameter(
            "train_itp_nograd", self.train_itp_nograd)
        setattr(module, "train_itp_nograd", self.train_itp_nograd)

        # Get fixed point from scratch
        fix1 = torch.tensor([]).cuda()
        fix1_nograd = torch.tensor([]).cuda()
        for i in fix1_dict:
            if "adapter" in i[0] or "classifier" in i[0]:
                fix1 = torch.cat((fix1, i[1].flatten().cuda()), dim=0)
            else:
                fix1_nograd = torch.cat((fix1_nograd, i[1].flatten().cuda()), dim=0)
        
        fix2 = torch.tensor([]).cuda()
        fix2_nograd = torch.tensor([]).cuda()
        for i in fix2_dict:
            if "adapter" in i[0] or "classifier" in i[0]:
                fix2 = torch.cat((fix2, i[1].flatten().cuda()), dim=0)
            else:
                fix2_nograd = torch.cat((fix2_nograd, i[1].flatten().cuda()), dim=0)

        self.fix_point = 0.5 * fix1 + 0.5 * fix2
        self.fix_point.requires_grad = True
        self.fix_point_nograd = 0.5 * fix1_nograd + 0.5 * fix2_nograd
        self.fix_point_nograd.requires_grad = False
        
        self.train_itp.requires_grad = True
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
    
        x = kwargs['x']
        kwargs.pop('x')
        x = float(x)

        if x == -1:
            with torch.no_grad():
                # x = -1 means doing evaluation! load the train_itp as model parameters
                pos = 0
                flatten_pet = self.train_itp      
                for name, base, localname in self.name_base_localname:
                    length = self.init_state[name].numel()
                    shape = self.init_state[name].size()
                    param = flatten_pet[pos:pos+length].view(shape)
                    delattr(base, localname)
                    setattr(base, localname, param)
                    pos = pos + length
                pos = 0
                flatten_pet = self.train_itp_nograd        
                for name, base, localname in self.name_base_localname_nograd:
                    length = self.init_state[name].numel()
                    shape = self.init_state[name].size()
                    param = flatten_pet[pos:pos+length].view(shape)
                    delattr(base, localname)
                    setattr(base, localname, param)
                    pos = pos + length
                return

        fix_model = kwargs['fix_model']
        kwargs.pop('fix_model')
        fix_model = int(fix_model)

        # interpolation!
        pos = 0
        flatten_pet = x * x * self.train_itp + 2 * x * (1 - x) * self.fix_point + (1 - x) * (1 - x) * self.end_w[fix_model - 1]
        with torch.enable_grad():
            for name, base, localname in self.name_base_localname:
                length = self.init_state[name].numel()
                shape = self.init_state[name].size()
                param = flatten_pet[pos:pos+length].view(shape)
                delattr(base, localname)
                setattr(base, localname, param)
                pos = pos + length
        pos = 0
        flatten_pet = x * x * self.train_itp_nograd + 2 * x * (1 - x) * self.fix_point_nograd + (1 - x) * (1 - x) * self.end_w_nograd[fix_model - 1]
        with torch.no_grad():
            for name, base, localname in self.name_base_localname_nograd:
                length = self.init_state[name].numel()
                shape = self.init_state[name].size()
                param = flatten_pet[pos:pos+length].view(shape)
                delattr(base, localname)
                setattr(base, localname, param)
                pos = pos + length

    @staticmethod
    def apply(module, args, model1, model2, device="cpu"):
        for k, hook in module._forward_pre_hooks.items():
            assert False
            if isinstance(hook, WrapModelCF) and hook.name == name:
                raise RuntimeError("Cannot register two intrinsic dimension hooks on "
                                   "the same parameter {}".format(name))
        fn = WrapModelCF(
            module, args, model1, model2, device)
        module.register_forward_pre_hook(fn)
        return fn
    
def itp_find_curve(module, args, model1, model2, device="cpu"):
    ID_wrap = WrapModelCF.apply(
        module, args, model1, model2, device)
    return module, ID_wrap
