from modules.comdel import cmodel
import torch
from modules import backbone
from torch.nn import functional as F
from modules.components import loss
from utils import plot_motion



class network(cmodel):

    def __init__(self, config):
        cmodel.__init__(self,config)

        self.config = config
        self.loss_names = ['REGRESSION','SMOOTH']
        self.visual_names = ['INPUT_X','INPUT_Y',"FLOW_LABEL_SHOW",'FLOW_PREDICT_SHOW']
        self.model_names = ["G"]

        self.netG = backbone.create_backbone(net_name='FlowNetS',
                                             init_type=config['init_type'],
                                             init_gain=float(config['init_gain']),
                                             gpu_ids=config['gpu_ids'])

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),lr=float(config['learning_rate']),betas=(0.5, 0.999),weight_decay=1e-4)
        self.optimizers.append(['G',self.optimizer_G])

        self.INPUT_X = None
        self.INPUT_Y = None
        self.FLOW_LABEL_SHOW = None
        self.FLOW_PREDICT_SHOW = None
        self.FLOW_LABEL = None
        self.FLOW_PREDICT = None

        self.LOSS_REGRESSION = None
        self.LOSS_SMOOTH = None


    def set_input(self, inputs):
        self.INPUT_X = inputs['IMAGE_X'].to(self.device)
        self.INPUT_Y = inputs['IMAGE_Y'].to(self.device)
        self.FLOW_LABEL = inputs['FLOW'].to(self.device)
        self.FLOW_LABEL_SHOW = plot_motion.vis_flow_tensor(F.interpolate(F.interpolate(self.FLOW_LABEL,(128,128),mode='area'),(512,512),mode='area'))



    def forward(self):
        self.FLOW_PREDICT = self.netG.forward(torch.cat([self.INPUT_X,self.INPUT_Y],dim=1))
        self.FLOW_PREDICT_SHOW = plot_motion.vis_flow_tensor(F.upsample(self.FLOW_PREDICT[0],(512,512)))



    def test_forward(self):
        self.test_result = []
        with torch.no_grad():
            FLOWS = self.netG(torch.cat([self.INPUT_X,self.INPUT_Y],dim=1))
            self.test_result.append(['PREDICTION_FLOW',FLOWS[0]])
            self.test_result.append(['INPUT_X',self.INPUT_X])




    def backward_G(self):
        self.LOSS_REGRESSION = loss.MultiEPE(self.FLOW_PREDICT,self.FLOW_LABEL)
        self.LOSS_SMOOTH = 0.0

        self.LOSS_REGRESSION.backward()


    def optimize_parameters(self):
        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()


