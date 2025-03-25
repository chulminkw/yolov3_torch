import torch
from torch.utils.data import Dataset
import torchvision

import glob
import xml.etree.ElementTree as ET
import os
import shutil
import pandas as pd

sets = [('2007', 'train'), ('2007', 'val')]

CLASS_NAMES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def create_meta_df(list_path, image_dir, xml_anno_dir, gubun='tr'):
  df = pd.read_csv(list_path, header=None, names=["img_name"], dtype=str)
  df['img_path'] = df['img_name'].apply(lambda x: f"{image_dir}/{x}.jpg")
  df['xml_anno_path'] = df['img_name'].apply(lambda x: f"{xml_anno_dir}/{x}.xml")
  df['tr_val'] = gubun
  return df

def get_added_df(tr_list_path, val_list_path, image_dir, xml_anno_dir):
  tr_df = create_meta_df(tr_list_path, image_dir, xml_anno_dir, gubun='tr')
  val_df = create_meta_df(val_list_path, image_dir, xml_anno_dir, gubun='val')
  df = pd.concat([tr_df, val_df], ignore_index=True)
  return df
  

# 1개의 voc xml 파일을 Yolo 포맷용 txt 파일로 변경하는 함수 
def xml_to_txt(input_xml_file, output_txt_file):
  # ElementTree로 입력 XML파일 파싱. 
  tree = ET.parse(input_xml_file)
  root = tree.getroot()
  img_node = root.find('size')
  # img_node를 찾지 못하면 종료
  if img_node is None:
    return None
  # 원본 이미지의 너비와 높이 추출. 
  img_width = int(img_node.find('width').text)
  img_height = int(img_node.find('height').text)
  # xml 파일내에 있는 모든 object Element를 찾음. 
  value_str = None
  with open(output_txt_file, 'w') as output_fp:
    for obj in root.findall('object'):
        # bndbox를 찾아서 좌상단(xmin, ymin), 우하단(xmax, ymax) 좌표 추출. 
        object_name = obj.find('name').text
        xmlbox = obj.find('bndbox')
        x1 = int(xmlbox.find('xmin').text)
        y1 = int(xmlbox.find('ymin').text)
        x2 = int(xmlbox.find('xmax').text)
        y2 = int(xmlbox.find('ymax').text)
        # 만약 좌표중에 하나라도 0보다 작은 값이 있으면 종료. 
        if (x1 < 0) or (x2 < 0) or (y1 < 0) or (y2 < 0):
          break
        #object_name과 원본 좌표를 입력하여 Yolo 포맷으로 변환하는 convert_yolo_coord()함수 호출. 
        class_id, cx_norm, cy_norm, w_norm, h_norm = convert_yolo_coord(object_name, img_width, img_height, x1, y1, x2, y2)
        # 변환된 yolo 좌표를 object 별로 출력 text 파일에 write
        value_str = ('{0} {1} {2} {3} {4}').format(class_id, cx_norm, cy_norm, w_norm, h_norm)
        output_fp.write(value_str+'\n')

# object_name과 원본 좌표를 입력하여 Yolo 포맷으로 변환
def convert_yolo_coord(object_name, img_width, img_height, x1, y1, x2, y2):
  # class_id는 CLASS_NAMES 리스트에서 index 번호로 추출. 
  class_id = CLASS_NAMES.index(object_name)
  # 중심 좌표와 너비, 높이 계산. 
  center_x = (x1 + x2)/2
  center_y = (y1 + y2)/2
  width = x2 - x1
  height = y2 - y1
  # 원본 이미지 기준으로 중심 좌표와 너비 높이를 0-1 사이 값으로 scaling
  center_x_norm = center_x / img_width
  center_y_norm = center_y / img_height
  width_norm = width / img_width
  height_norm = height / img_height

  return class_id, round(center_x_norm, 7), round(center_y_norm, 7), round(width_norm, 7), round(height_norm, 7)

def make_yolo_anno_file(df, tgt_labels_dir):
  tgt_label_path_list = []
  for index, row in df.iterrows():
    xml_anno_path = row['xml_anno_path']
    
    # yolo format으로 annotation할 txt 파일의 절대 경로명을 지정. 
    tgt_label_path = tgt_labels_dir + row['img_name']+'.txt'
    # # image의 경우 target images 디렉토리로 단순 copy
    # shutil.copy(src_image_path, tgt_images_dir)
    # annotation의 경우 xml 파일을 target labels 디렉토리에 Ultralytics Yolo format으로 변환하여  만듬
    xml_to_txt(xml_anno_path, tgt_label_path)
    tgt_label_path_list.append(tgt_label_path)
  
  return tgt_label_path_list

def get_scaled_anchors(img_size, grid_size, device):
    all_anchors = {'scale1': [(10, 13), (16, 30), (33, 23)],
                   'scale2': [(30, 61), (62, 45), (59, 119)],
                   'scale3': [(116, 90), (156, 198), (373, 326)]}
    if grid_size == 13:
        anchors = all_anchors['scale3']
    elif grid_size == 26:
        anchors = all_anchors['scale2']
    elif grid_size == 52:
        anchors = all_anchors['scale1']

    stride = img_size / grid_size
    # print('grid_size:', grid_size, 'stride:', stride, anchors)
    scaled_anchors = torch.tensor([(w/stride, h/stride) for w, h in anchors])
    scaled_anchors = scaled_anchors.to(device)
    return scaled_anchors
        

    stride = img_size / grid_size
    scaled_anchors = torch.tensor([(w/stride, h/stride) for w, h in anchors])
    return scaled_anchors

def box_iou_wh(wh_1, wh_2):
    w1, h1 = wh_1[..., 0], wh_1[..., 1]
    w2, h2 = wh_2[..., 0], wh_2[..., 1]
    intersection = torch.min(w1, w2) * torch.min(h1, h2)
    union = w1 * h1 + w2 * h2 - intersection
    ious = intersection / (union + 1e-14) # 1e-14는 0으로 나뉘어지는것을 막기 위해서
    
    return ious

def mapping_targets_on_grid(pred_raw, targets, scaled_anchors, grid_size, device, is_visual=False):
    '''
    targets는 배치 포함 2차원
    pred_raw는 배치 포함 5차원
    '''
    # target box와 가장 많이 겹치는 iou를 가지는 anchor의 index 찾기. 
    ious = box_iou_wh(scaled_anchors, targets[:, None, 4:6] * grid_size)
    #ious = torch.cat([box_iou_wh(scaled_anchors, box_wh).unsqueeze(0) for box_wh in targets[:, 4:6]*7], dim=0)
    best_iou_idx = ious.argmax(-1)

    tgt_on_grid = torch.zeros_like(pred_raw, dtype=torch.float, device=device)
    # indexing으로 사용될 tensor type은 int/long/bool
    b_idx = targets[:, 0].long()
    tgt_label = targets[:, 1].long()
    tgt_xy = targets[:, 2:4] * grid_size
    tgt_wh = targets[:, 4:6] * grid_size
    tgt_grid_xy = tgt_xy.long()
    tgt_grid_x = tgt_grid_xy[:, 0]
    tgt_grid_y = tgt_grid_xy[:, 1]

    tgt_on_grid[b_idx, best_iou_idx, tgt_grid_y, tgt_grid_x, :2] = tgt_xy - tgt_grid_xy
    if is_visual:
        tgt_on_grid[b_idx, best_iou_idx, tgt_grid_y, tgt_grid_x, 2:4] = tgt_wh
    else:
        tgt_on_grid[b_idx, best_iou_idx, tgt_grid_y, tgt_grid_x, 2:4] = tgt_wh / scaled_anchors[best_iou_idx]
        
    # target object confidence
    tgt_on_grid[b_idx, best_iou_idx, tgt_grid_y, tgt_grid_x, 4:5] = 1
    # one-hot encoding
    tgt_on_grid[b_idx, best_iou_idx, tgt_grid_y, tgt_grid_x, 5+tgt_label] = 1
    
    # object_mask = torch.zeros_like(pred_raw[:, 0], dtype=torch.bool, device=device)
    # object_mask[b_idx, best_iou_idx, tgt_grid_y, tgt_grid_x] = 1
    # no_object_mask = 1 - object_mask[b_idx, best_iou_idx, tgt_grid_y, tgt_grid_x]
    return best_iou_idx, tgt_on_grid
    
def get_obj_values_indexes(tgt_on_grid):
    mask = tgt_on_grid[..., 4] > 0
    filtered_values = tgt_on_grid[mask]
    indices = torch.nonzero(mask)
    return filtered_values, indices