import numpy as np

# config
path = 'deepstream_out.txt'
deepstream_out = open(path, 'w')
path = 'gt_sort.txt'
gt_sort_out = open(path, 'w')
GT_path = 'elementary_school.csv'
GT_resolution = [1920, 1080]
deepstream_resolution = [416, 416]
total_frame = 3601

def iou(a, b):
    """calculate IOU

    Args:
        a (int or float): the first coordinate, x1, y1, x2, y2
        b (int or float): the second coordinate, x1, y1, x2, y2
    
    Returns:
        float: the result of IOU
    """
    # get area of a
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    # get area of b
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    # get left top x of IoU
    iou_x1 = np.maximum(a[0], b[0])
    # get left top y of IoU
    iou_y1 = np.maximum(a[1], b[1])
    # get right bottom of IoU
    iou_x2 = np.minimum(a[2], b[2])
    # get right bottom of IoU
    iou_y2 = np.minimum(a[3], b[3])

    # get width of IoU
    iou_w = iou_x2 - iou_x1
    # get height of IoU
    iou_h = iou_y2 - iou_y1

    # get area of IoU
    area_iou = iou_w * iou_h
    # get overlap ratio between IoU and all area
    iou = area_iou / (area_a + area_b - area_iou)

    return iou

def MOTA_cal(iou_config):
    """calculate MOTA

    Args:
        iou_config (int or float): determine how many IOU can be seen as TP 
    
    Returns:
        float: TP, FN, FP, ID_SW, len_gt
    """
    FN = 0
    FP = 0
    TP = 0
    TN = 0
    ID_SW = 0
    len_gt = 0
    GT_predict = np.full(1000, -1) # for ID_SW
    for frame_num in range(1, total_frame):
        frame_gt = gt[np.where(gt[:,0] == frame_num)]
        frame_predict = predict[np.where(predict[:,0] == frame_num)]
        total_frame_predict = frame_predict
        TP_frame = 0
        for j in range(len(frame_gt)):
            # ignore small item
            if (frame_gt[j][4] - frame_gt[j][2])* (frame_gt[j][5] - frame_gt[j][3]) >= 0:
                len_gt = len_gt + 1
            else:
                continue
            all_iou = []
            for i in range(len(frame_predict)):
                # the IOU between a GT and many predicted item
                a = np.array((frame_gt[j][2], frame_gt[j][3], frame_gt[j][4], frame_gt[j][5]), dtype=np.float32)
                b = np.array((frame_predict[i][2], frame_predict[i][3], frame_predict[i][4], frame_predict[i][5]), dtype=np.float32)
                all_iou.append(iou(a, b))
            if len(all_iou) <= 0: 
                continue
            max_iou = max(all_iou)
            if max_iou > iou_config:
                # This GT object has corresponding predicted object 
                TP_frame = TP_frame + 1
                predict_index = np.where(all_iou[:] == max_iou)

                # whether there is ID_SW
                if GT_predict[frame_gt[j][1]] == -1:
                    pass
                else:
                    if GT_predict[frame_gt[j][1]] != frame_predict[predict_index[0].tolist()[0]][1]:
                        ID_SW = ID_SW + 1
                GT_predict[frame_gt[j][1]] = frame_predict[predict_index[0].tolist()[0]][1]
                
                frame_predict = np.delete(frame_predict, predict_index[0].tolist()[0], 0)
            else:
                # This GT object has no corresponding predicted object 
                FN = FN + 1
            # if frame_num == 1:

                # print(frame_predict)
                # print(max_iou)
        FP_frame = len(frame_predict)
        for j in range(len(frame_predict)):
            if (frame_predict[j][4] - frame_predict[j][2])* (frame_predict[j][5] - frame_predict[j][3]) <= 0:
                FP_frame = FP_frame - 1
        FP = FP + FP_frame
        TP = TP + TP_frame
        # if len(frame_predict) - TP_frame > 3:
        #     print("frame: " + str(frame_num))
        #     print("frame: " + str(len(frame_predict) - TP_frame))
    return TP, FN, FP, ID_SW, len_gt

def read_GT_file():
    """convert GT MOT format to deepstream format

    Returns:
        int: gt: frame, id, x1, y1, x2, y2
    """
    gt = []
    with open(GT_path, 'r') as f:
        for data in f.readlines():
            data = data.split(",")
            data = data[:6]
            data = [float(e) for e in data]

            # width, height to coordinate
            data[4] = data[2] + data[4]
            data[5] = data[3] + data[5]
            data[2] = data[2] / GT_resolution[0] * deepstream_resolution[0]
            data[4] = data[4] / GT_resolution[0] * deepstream_resolution[0]
            data[3] = data[3] / GT_resolution[1] * deepstream_resolution[1]
            data[5] = data[5] / GT_resolution[1] * deepstream_resolution[1]
            data = [int(e) for e in data]

            # shift frame
            data[0] = data[0] + 1
            gt.append(data)

    gt = np.array(gt)
    return gt

gt = read_GT_file()
len_gt = len(gt)

# sort by frame number and store into txt file
gt = gt[gt[:,0].argsort()]
cnt = 0
for i in range(len(gt)):
    print(str(gt[i]), file=gt_sort_out)

# combine all deepstream output into a file
predict = []
for i in range(total_frame):
    with open('result_deepstream_format/00_000_00'+str(i).zfill(4)+'.txt', 'r') as f:
        for data in f.readlines():
            if data != '\n':
                data = data.strip("\n")
            vdata = data.split(" ")
            vdata = [i + 1, vdata[1], vdata[5], vdata[6], vdata[7], vdata[8]]
            vdata = [int(float(e)) for e in vdata]
            print(str(vdata), file=deepstream_out)
            predict.append(vdata)
predict = np.array(predict)

gt_sort_out.close()
deepstream_out.close()

path = 'IOU_acc.txt'
f = open(path, 'w')
for i in range(100):
    TP, FN, FP, ID_SW, len_gt = MOTA_cal(i/100)

    MOTA = 1 - (FN + FP + ID_SW)/len_gt
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    print("iou: " + str(i)+" MOTA: " + str(MOTA)+" FN: " + str(FN)+" FP: " + str(FP)+" len_gt: " + str(len_gt)+" Precision: " + str(Precision)+" Recall: " + str(Recall))
    print(str(i)+", " + str(MOTA)+", " + str(FN)+", " + str(FP)+", " + str(len_gt)+", " + str(Precision)+", " + str(Recall), file=f)
f.close()

# FN, FP, ID_SW, len_gt = MOTA_cal(0.3)
# MOTA = 1 - (FN + FP + ID_SW)/len_gt
# print("MOTA: " + str(MOTA))
# print("FN: " + str(FN))
# print("FP: " + str(FP))
# print("ID_SW: " + str(ID_SW))
# print("len_gt: " + str(len_gt))