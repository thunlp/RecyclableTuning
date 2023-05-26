import torch
from torch import nn
from IPython import embed

class WrapModelCF:
    def __init__(self, module: nn.Module, args, model1, model2, device="cpu"):
        super().__init__()
        self.args = args
        self.device = device
        
        model_dict1 = [v for v in model1.state_dict().values()][1:]
        model_dict2 = [v for v in model2.state_dict().values()][1:]
        flat1 = torch.tensor([]).cuda()
        for v in model_dict1:
            flat1 = torch.cat((flat1, v.flatten().cuda()), dim=0)
        flat2 = torch.tensor([]).cuda()
        for v in model_dict2:
            flat2 = torch.cat((flat2, v.flatten().cuda()), dim=0)
        self.end_w = torch.stack([flat1, flat2],dim=0)
        self.end_w.requires_grad = False

        self.flatten_size = self.end_w.size()[1]
        self.name_base_localname = []
        
        # module初始化展平得到train_itp
        self.train_itp = nn.Parameter(torch.zeros(self.flatten_size).cpu() if device=="cpu" else torch.zeros(self.flatten_size).cuda())
        
        pet_name_modules = []
        flatten = torch.Tensor([]).cuda()
        for pet_name_module, params in module.named_parameters():
            flatten = torch.cat((flatten, params.flatten().cuda()),dim=0)
            pet_name_modules.append(pet_name_module)
        self.pet_name_modules = pet_name_modules

        # embed()
        
        # use roberta as init state
        self.train_itp.data = flatten
        self.init_state = module.state_dict()
        
        module.register_parameter(
            "train_itp", self.train_itp)
        setattr(module, "train_itp", self.train_itp)
        
        self.train_itp.requires_grad = True
        
        for name, param in module.named_parameters():
            base, localname = module, name
            while "." in localname:
                prefix, localname = localname.split(".", 1)
                base = base.__getattr__(prefix)
            self.name_base_localname.append((name, base, localname))
    
    
    def __call__(self, module, *inputs, **kwargs):
        pos = 0

        x = kwargs['x']
        kwargs.pop('x')
        x = float(x)

        if x == -1:
            # x = -1 means doing evaluation! load the train_itp as model parameters
            flatten_pet = self.train_itp
        
            with torch.no_grad():
                for name, base, localname in self.name_base_localname:
                    if localname == "train_itp":
                        continue

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
        flatten_pet = x * self.train_itp + (1 - x) * self.end_w[fix_model - 1]
        
        with torch.enable_grad():
            for name, base, localname in self.name_base_localname:
                if localname == "train_itp":
                    continue

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
    
def itp_find_finetune(module, args, model1, model2, device="cpu"):
    ID_wrap = WrapModelCF.apply(
        module, args, model1, model2, device)
    return module, ID_wrap
