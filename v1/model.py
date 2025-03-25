import torch
import torch.nn as nn

def base_conv(in_channels, out_channels, kernel_size, stride=1, padding=1):
    conv1 = nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, padding, bias=False)
    bn1 = nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5)
    leaky1 = nn.LeakyReLU(negative_slope=0.1)
    
    return nn.Sequential(conv1, bn1, leaky1)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        half_channels = in_channels // 2
        self.conv_01 = base_conv(in_channels=in_channels,
                               out_channels=half_channels,
                               kernel_size=1, stride=1, padding=0)
        self.conv_02 = base_conv(half_channels, in_channels,
                                kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        identity = x
        x = self.conv_01(x)
        x = self.conv_02(x)
        x += identity
        return x

def make_residual_stage(in_channels, num_blocks):
    residual_blocks = []

    for i in range(0, num_blocks):
        residual_blocks.append(ResidualBlock(in_channels))

    return nn.Sequential(*residual_blocks)

class DarknetBody(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_conv_01 = base_conv(in_channels=3, out_channels=32,
                                 kernel_size=3, stride=1, padding=1)
        self.base_conv_02 = base_conv(32, 64, 3, stride=2, padding=1)
        self.residual_stage_01 = make_residual_stage(64, 1)
        
        self.base_conv_03 = base_conv(64, 128, 3, stride=2, padding=1)
        self.residual_stage_02 = make_residual_stage(128, 2)
        
        self.base_conv_04 = base_conv(128, 256, 3, stride=2, padding=1)
        self.residual_stage_03 = make_residual_stage(256, 8)

        self.base_conv_05 = base_conv(256, 512, kernel_size=3, stride=2, padding=1)
        self.residual_stage_04 = make_residual_stage(512, 8)

        self.base_conv_06 = base_conv(512, 1024, kernel_size=3, stride=2, padding=1)
        self.residual_stage_05 = make_residual_stage(1024, 4)

    def forward(self, x):
        x = self.base_conv_01(x)
        x = self.base_conv_02(x)
        x = self.residual_stage_01(x)
        
        x = self.base_conv_03(x)
        x = self.residual_stage_02(x)
        
        x = self.base_conv_04(x)
        x = self.residual_stage_03(x)
        route_03 = x

        x = self.base_conv_05(x)
        x = self.residual_stage_04(x)
        route_02 = x

        x = self.base_conv_06(x)
        x = self.residual_stage_05(x)
        route_01 = x

        return route_01, route_02, route_03

# Upsample 이전 FPN에 사용되는 연속 Conv block 생성. 
def create_fpn_conv_block(in_channels, out_channels):
    mid_channels = out_channels * 2
    
    base_conv_01 = base_conv(in_channels, out_channels, kernel_size=1, padding=0)
    base_conv_02 = base_conv(out_channels, mid_channels, kernel_size=3, padding=1)
    base_conv_03 = base_conv(mid_channels, out_channels, kernel_size=1, padding=0)
    base_conv_04 = base_conv(out_channels, mid_channels, kernel_size=3, padding=1)
    base_conv_05 = base_conv(mid_channels, out_channels, kernel_size=1, padding=0)
    
    fpn_conv_block = nn.Sequential(
        base_conv_01, base_conv_02, base_conv_03, base_conv_04, base_conv_05)

    return fpn_conv_block

# Upsample Block 생성
def create_fpn_upsample_block(in_channels, out_channels):
    base_conv_01 = base_conv(in_channels, out_channels, kernel_size=1, padding=0)
    upsample = nn.Upsample(scale_factor=2) #mode='nearest'

    fpn_upsample_block = nn.Sequential(base_conv_01, upsample)

    return fpn_upsample_block

# 판별 클래스에 따라 out_channels 값 변화. 
# MS-COCO의 경우 80개 Class  + 1개 object + 4개 좌표 = 85 개 * 3개 anchor 이므로 255
# VOC의 경우 20개 Class + 5 = 25 * 3개 anchor이므로 75
def create_to_head_block(in_channels, out_channels):
    double_channels = in_channels * 2
    base_conv_01 = base_conv(in_channels, double_channels, kernel_size=3, padding=1)
    conv_02 = nn.Conv2d(double_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    necktohead_block = nn.Sequential(base_conv_01, conv_02)
    
    return necktohead_block

class YoloLayer(nn.Module):
    def __init__(self, img_size, anchors, num_classes): #img_size
        super().__init__()
        # self.img_size = img_size
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.img_size = img_size
        self.device='cpu' # test용도

    def forward(self, x):
        # x의 shape는 (batch_size, num_anchors * (num_classes + 5), grid_size, grid_size)
        # 중심좌표, 너비, 높이, object, class값 연산을 편리하기 위해서 (batch_size, num_anchors, grid_size, grid_size, num_classes+5) 로 변환
        batch_size = x.shape[0]
        grid_size = x.shape[2]
        preds = x.view(batch_size, self.num_anchors, self.num_classes + 5, grid_size, grid_size)\
                      .permute(0, 1, 3, 4, 2).contiguous()
        pred_cxy = torch.sigmoid(preds[..., 0:2])
        pred_wh = preds[..., 2:4]
        pred_obj = torch.sigmoid(preds[..., 4:5])
        pred_class = torch.sigmoid(preds[..., 5:])

        scaled_anchors = self.get_scaled_anchors(self.anchors, self.img_size, grid_size)

        adj_pred_cxy, adj_pred_wh = self.adjust_to_grid(pred_cxy, pred_wh, 
                                                   scaled_anchors, grid_size, self.device)
        adj_preds = torch.cat([adj_pred_cxy, adj_pred_wh, pred_obj, pred_class], dim=-1)
        
        if not self.training: # 학습이 아니고 예측(inference) 시 별도 로직 필요. 
            pass

        return adj_preds
        
    def get_scaled_anchors(self, org_anchors, img_size, grid_size):
        stride = img_size / grid_size
        scaled_anchors = torch.tensor([(w/stride, h/stride) for w, h in org_anchors], device=self.device)
        return scaled_anchors
        
    def adjust_to_grid(self, pred_cxy, pred_wh, scaled_anchors, grid_size, device):
        grid_x, grid_y = torch.meshgrid(
                torch.arange(grid_size).to(device),  # X coordinates
                torch.arange(grid_size).to(device),  # Y coordinates
                indexing='xy' # Ensures matrix-like indexing
        )
        grid = torch.stack([grid_y, grid_x], dim=-1).view(1, 1, grid_size, grid_size, 2).to(device)
        adj_pred_cxy = pred_cxy  + grid
        #print('pred_wh shape:', pred_wh.shape, 'anchors shape:', anchors.shape)
        # pred_wh와 곱하기 위해서 scaled_anchors의 shape를 변경. 
        reshaped_anchors = scaled_anchors.view(1, scaled_anchors.shape[0], 1, 1, scaled_anchors.shape[1])
        adj_pred_wh = torch.exp(pred_wh) * reshaped_anchors
    
        return adj_pred_cxy, adj_pred_wh
    
class YoloV3(nn.Module):
    def __init__(self, img_size=416, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        self.all_anchors = {'layer_01': [(116, 90), (156, 198), (373, 326)],
                           'layer_02': [(30, 61), (62, 45), (59, 119)],
                           'layer_03': [(10, 13), (16, 30), (33, 23)]}
        num_anchors = 3

        self.darknet_body = DarknetBody()
        # Darknet 출력의 최종 Feature Map에 적용되는 FPN Conv와 Upsample
        self.fpn_conv_block_01 = create_fpn_conv_block(1024, 512)
        self.to_head_block_01 = create_to_head_block(512, (num_classes+5) * num_anchors)
        self.yolo_layer_01 = YoloLayer(img_size=img_size, 
                                       anchors=self.all_anchors['layer_01'], num_classes=num_classes)

        
        self.fpn_up_block_02 = create_fpn_upsample_block(512, 256)
        self.fpn_conv_block_02 = create_fpn_conv_block(768, 256)
        self.to_head_block_02 = create_to_head_block(256, (num_classes+5) * num_anchors)
        self.yolo_layer_02 = YoloLayer(img_size=img_size, 
                                       anchors=self.all_anchors['layer_02'], num_classes=num_classes)

        self.fpn_up_block_03 = create_fpn_upsample_block(256, 128)
        self.fpn_conv_block_03 = create_fpn_conv_block(384, 128)
        self.to_head_block_03 = create_to_head_block(128, (num_classes+5) * num_anchors)
        self.yolo_layer_03 = YoloLayer(img_size=img_size, 
                                       anchors=self.all_anchors['layer_03'], num_classes=num_classes)

    def forward(self, x):
        route_01, route_02, route_03 = self.darknet_body(x)
        # 첫번째 FPN Conv Block -> Head Block Output(shape는 (75, 13, 13))
        fpn_conv_output_01 = self.fpn_conv_block_01(route_01)
        h_output_01 = self.to_head_block_01(fpn_conv_output_01)
        output_01 = self.yolo_layer_01(h_output_01)
        
        # Upsampling 적용 후 route_02 feature map과 Concat 수행.
        # Concat 적용 후 Output shape는 (768, 26, 26)
        fpn_up_output_02 = self.fpn_up_block_02(fpn_conv_output_01)
        concat_02 = torch.concat((route_02, fpn_up_output_02), dim=1) 

        # route_02와 concat된 결과를 FPN Conv -> Head Block 적용 후 Output shape는 (75, 26, 26)
        fpn_conv_output_02 = self.fpn_conv_block_02(concat_02)
        h_output_02 = self.to_head_block_02(fpn_conv_output_02)
        output_02 = self.yolo_layer_02(h_output_02)

        # Upsampling 적용 후 route_03 feature map과 Concat 수행.
        # Concat 적용 후 Output shape는 (768, 26, 26)
        fpn_up_output_03 = self.fpn_up_block_03(fpn_conv_output_02)
        concat_03 = torch.concat((route_03, fpn_up_output_03), dim=1) 

        # route_03와 concat된 결과를 FPN Conv -> Head Block 적용 후 Output shape는 (75, 52, 52)
        fpn_conv_output_03 = self.fpn_conv_block_03(concat_03)
        h_output_03 = self.to_head_block_03(fpn_conv_output_03)
        output_03 = self.yolo_layer_03(h_output_03)

        return output_01, output_02, output_03