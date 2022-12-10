import torch
from torch import nn
import timm

# from hybernet.hybridnets.model import BiFPN, BiFPNDecoder #Regressor, Classifier, 

from hybernet.hybridnets.model import BiFPN, BiFPNDecoder, SegmentationHead, ASPP    

# from hybridnets.model import SegmentationHead
# from utils.utils import Anchors
# from hybernet.hybridnets.model import SegmentationHead
from hybernet.encoders import get_encoder
# from hybernet.encoders import get_encoder

class HybridNetsBackbone(nn.Module):
    def __init__(self, num_classes=80, compound_coef=3, seg_classes=19, backbone_name='vit_base_patch16_224', **kwargs):
        super(HybridNetsBackbone, self).__init__()
        self.compound_coef = compound_coef

        self.seg_classes = seg_classes

        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6, 7]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5, 5]
        self.pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]
        self.anchor_scale = [1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,]
        self.aspect_ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
        self.num_scales = len(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
            8: [80, 224, 640],
        }

        num_anchors = len(self.aspect_ratios) * self.num_scales

        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[compound_coef],
                    True if _ == 0 else False,
                    attention=True if compound_coef < 6 else False,
                    use_p8=compound_coef > 7)
              for _ in range(self.fpn_cell_repeats[compound_coef])])

        self.num_classes = num_classes
        # self.regressor = Regressor(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
        #                            num_layers=self.box_class_repeats[self.compound_coef],
        #                            pyramid_levels=self.pyramid_levels[self.compound_coef])

        '''Modified by Dat Vu'''
        # self.decoder = DecoderModule()
        self.bifpndecoder = BiFPNDecoder(pyramid_channels=self.fpn_num_filters[self.compound_coef])

        self.segmentation_head = SegmentationHead(
            in_channels=64,
            out_channels=self.seg_classes if self.seg_classes > 1 else self.seg_classes,
            activation='softmax2d' if self.seg_classes > 1 else 'sigmoid',
            kernel_size=1,
            upsampling=4,
        )

        # self.classifier = Classifier(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
        #                              num_classes=num_classes,
        #                              num_layers=self.box_class_repeats[self.compound_coef],
        #                              pyramid_levels=self.pyramid_levels[self.compound_coef])

        # self.anchors = Anchors(anchor_scale=self.anchor_scale[compound_coef],
        #                        pyramid_levels=(torch.arange(self.pyramid_levels[self.compound_coef]) + 3).tolist(),
        #                        **kwargs)
# resnet34
        # if backbone_name:
        # self.encoder = timm.create_model(backbone_name, pretrained=True, features_only=True, out_indices=(2,3,4))  # P3,P4,P5
        self.encoder1 = timm.create_model('resnet34', pretrained=True, features_only=True, out_indices=(1,2,3,4))  # P3,P4,P5
        # else:
        # self.encoder = timm.create_model('resnet34', pretrained=True, features_only=True)  # P3,P4,P5
        #     # EfficientNet_Pytorch
        self.encoder = get_encoder(
            'efficientnet-b' + str(self.backbone_compound_coef[compound_coef]),
            in_channels=3,
            depth=5,
            weights='imagenet',
        )
        self.dil2 = ASPP(32,32)
        self.dil3 = ASPP(48,48)
        self.dil4 = ASPP(136,136)
        self.dil5 = ASPP(384,384)

        self.initialize_decoder(self.bifpndecoder)
        self.initialize_head(self.segmentation_head)
        self.initialize_decoder(self.bifpn)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        max_size = inputs.shape[-1]

        # p1, p2, p3, p4, p5 = self.backbone_net(inputs)
        p2, p3, p4, p5 = self.encoder(inputs)[-4:]  # self.backbone_net(inputs)

        # print('encoder:',self.encoder.shape)
        # print('P2:', p2.shape)
        # print('P3:', p3.shape)
        # print('P4:', p4.shape)
        # print('P5:', p5.shape)

        p2 = self.dil2(p2)
        p3 = self.dil3(p3)
        p4 = self.dil4(p4)
        p5 = self.dil5(p5)

        # print('P2:', p2.shape)
        # print('P3:', p3.shape)
        # print('P4:', p4.shape)
        # print('P5:', p5.shape)

        features = (p3, p4, p5)
        

        features = self.bifpn(features)
        
        p3,p4,p5,p6,p7 = features
        # print('P3_1:', p3.shape)
        # print('P4_1:', p4.shape)
        # print('P5_1:', p5.shape)
        # print('P6_1:', p6.shape)
        # print('P7_1:', p7.shape)
        
        outputs = self.bifpndecoder((p2,p3,p4,p5,p6,p7))
        # print('output:', outputs.shape)

        segmentation = self.segmentation_head(outputs)
        
        # regression = self.regressor(features)
        # classification = self.classifier(features)
        # anchors = self.anchors(inputs, inputs.dtype)

        return segmentation
    
    def initialize_decoder(self, module):
        for m in module.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def initialize_head(self, module):
        for m in module.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    #model = get_net(False)
    model =HybridNetsBackbone(seg_classes=20)
    input_ = torch.randn((1, 3, 512, 512))
    # gt_ = torch.rand((1, 2, 256, 256))

#     model_out,SAD_out, lout = model(input_)
#     model_out,SAD_out, lout = model(input_)
#     detects, dring_area_seg, lane_line_seg = model_out
    dring_area_seg= model(input_)
    # Da_fmap, LL_fmap = SAD_out

    print(dring_area_seg.shape)
    print(dring_area_seg)