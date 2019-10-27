from torch.nn.modules.module import Module
from functions.dynamic_prog import DynamicProgFunction

class DynamicProgModule(Module):
    def forward(self, input1, input2):
        return DynamicProgFunction()(input1, input2)
