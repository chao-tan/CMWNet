from modules import network
from modules.components import FlowNetS,FlowNetS_SF



def create_backbone(net_name, init_type, init_gain, gpu_ids):
    net = None

    if net_name is "FlowNetS":
        net = FlowNetS.FlowNetS()
    elif net_name is 'FlowNetS_SF':
        net = FlowNetS_SF.FlowNetS()



    else:
        raise NotImplementedError("net_name mismatched!")

    return network.init_net(net, init_type, init_gain, gpu_ids)

