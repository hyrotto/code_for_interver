import cv2
import pandas as pd
import os
import numpy as np

img = cv2.imread("GS__0018.JPG")
img360_h, img360_w = img.shape[:2]

num_rect=0
rects=[]
scores=[]
classes=[]

#二つの矩形のIoU計算
def iou(a, b):
    # a, bは矩形を表すリストで、a=[xmin, ymin, xmax, ymax]
    ax_mn, ay_mn, ax_mx, ay_mx = a[0], a[1], a[2], a[3]
    bx_mn, by_mn, bx_mx, by_mx = b[0], b[1], b[2], b[3]

    a_area = (ax_mx - ax_mn + 1) * (ay_mx - ay_mn + 1)
    b_area = (bx_mx - bx_mn + 1) * (by_mx - by_mn + 1)

    abx_mn = max(ax_mn, bx_mn)
    aby_mn = max(ay_mn, by_mn)
    abx_mx = min(ax_mx, bx_mx)
    aby_mx = min(ay_mx, by_mx)
    w = max(0, abx_mx - abx_mn + 1)
    h = max(0, aby_mx - aby_mn + 1)
    intersect = w*h

    iou = intersect / (a_area + b_area - intersect)
    return iou

#矩形が縦長か横長か判定（正方形：0 縦長：１　横長：２）
def VorH(bbox):
    #左上と右下の座標を整える
    if bbox[2]<bbox[0]:
        bbox[2],bbox[0]=bbox[0],bbox[2]
    if bbox[3]<bbox[1]:
        bbox[3],bbox[1]=bbox[1],bbox[3]
        
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    
    if width == height:
        return 0
    # 幅が高さよりも大きい場合は横長（返り値: 2）
    elif width > height:
        return 2
    # 幅が高さよりも小さい場合は縦長（返り値: 1）
    else:
        return 1
    
#矩形の面積を求める
def BBarea(bbox):
    #左上と右下の座標を整える
    if bbox[2]<bbox[0]:
        bbox[2],bbox[0]=bbox[0],bbox[2]
    if bbox[3]<bbox[1]:
        bbox[3],bbox[1]=bbox[1],bbox[3]
    
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    
    return width*height

#矩形の中心座標を算出
def rectangle_center(bbox):
    x_center = (bbox[0]+ bbox[2]) / 2
    y_center = (bbox[1] + bbox[3]) / 2
    return x_center, y_center    

#矩形の幅、高さを算出
def rectangle_length(bbox):
    x_length = abs(bbox[0]-bbox[2])
    y_length = abs(bbox[1]-bbox[3])
    return x_length, y_length

#高さ、幅、中心座標を指定して矩形を生成
def creat_BBox(xylength,center):
    BBox = [int(center[0]-xylength[0]/2),int(center[1]-xylength[1]/2),int(center[0]+xylength[0]/2),int(center[1]+xylength[1]/2)]
    return BBox

#Non-Maximum Suppresion実装関数
def nms(bboxes, scores, classes, iou_threshold):
    new_bboxes = [] # NMS適用後の矩形リスト
    new_scores = [] # NMS適用後の信頼度(スコア値)リスト
    new_classes = [] # NMS適用後のクラスのリスト
    while len(bboxes) > 0:
        # スコア最大の矩形のインデックスを取得
        argmax = scores.index(max(scores))
        # スコア最大の矩形、スコア値、クラスをそれぞれのリストから消去
        bbox = bboxes.pop(argmax)
        score = scores.pop(argmax)
        clss = classes.pop(argmax)        
        # スコア最大の矩形と、対応するスコア値、クラスをNMS適用後のリストに格納
        new_bboxes.append(bbox)
        new_scores.append(score)
        new_classes.append(clss)
        pop_i = []
        for i, bbox_tmp in enumerate(bboxes):#添え字と要素でループ
            if classes[i] == clss:
            # スコア最大の矩形bboxとのIoUがiou_threshold以上のインデックスを取得
                if iou(bbox, bbox_tmp) >= iou_threshold:
                    pop_i.append(i)
        # 取得したインデックス(pop_i)の矩形、スコア値、クラスをそれぞれのリストから消去
        for i in pop_i[::-1]:
            bboxes.pop(i)
            scores.pop(i)
            classes.pop(i)
    return new_bboxes, new_scores, new_classes

#NMS2 縦長横長の多数決をとり、面積の一番大きいものを残す
def nms2(bboxes, scores, classes, iou_threshold):
    new_bboxes = [] # NMS適用後の矩形リスト
    new_scores = [] # NMS適用後の信頼度(スコア値)リスト
    new_classes = [] # NMS適用後のクラスのリスト
    
    while len(bboxes) > 0:
        # スコア最大の矩形のインデックスを取得
        argmax = scores.index(max(scores))
        VNum=0
        HNum=0
        #縦長横長それぞれの面積最大の矩形の添え字を保存
        mxBBidx_H=-1
        mxBBidx_V=-1
        
        #縦長か横長か判定
        if (VorH(bboxes[argmax])==1):
            VNum+=1
            mxBBidx_V=argmax
        elif(VorH(bboxes[argmax])==2):
            HNum+=1
            mxBBidx_H=argmax  
            
        pop_i = [] #iou取り出す矩形の添え字保存用の配列
        pop_i.append(argmax)
        
        for i, bbox_tmp in enumerate(bboxes):#添え字と要素でループ
            # スコア最大の矩形bboxとのIoUがiou_threshold以上のインデックスを取得
            if not i == argmax:
                if classes[i] == classes[argmax]:#同じクラスなら
                    if iou(bboxes[argmax], bbox_tmp) >= iou_threshold:#iou値が閾値以上なら
                        pop_i.append(i)
                        if (VorH(bboxes[i])==1):
                            VNum+=1
                            if mxBBidx_V == -1:
                                mxBBidx_V = i
                            elif BBarea(bboxes[mxBBidx_V]) < BBarea(bbox_tmp):
                                mxBBidx_V = i
                        elif(VorH(bboxes[i])==2):
                            HNum+=1
                            if mxBBidx_H == -1:
                                mxBBidx_H = i
                            elif BBarea(bboxes[mxBBidx_H]) < BBarea(bbox_tmp):
                                mxBBidx_H = i
        if(VNum > HNum):
            new_bboxes.append(bboxes[mxBBidx_V])
            new_scores.append(scores[mxBBidx_V])
            new_classes.append(classes[mxBBidx_V])
        else:
            new_bboxes.append(bboxes[mxBBidx_H])
            new_scores.append(scores[mxBBidx_H])
            new_classes.append(classes[mxBBidx_H])
        
        pop_i.sort()
        # 取得したインデックス(pop_i)の矩形、スコア値、クラスをそれぞれのリストから消去
        for i in pop_i[::-1]:
            bboxes.pop(i)
            scores.pop(i)
            classes.pop(i)

    return new_bboxes, new_scores, new_classes

#NMS2 縦長横長の多数決をとり、平均的な位置、長さの矩形を残す
def nms3(bboxes, scores, classes, iou_threshold):
    new_bboxes = [] # NMS適用後の矩形リスト
    new_scores = [] # NMS適用後の信頼度(スコア値)リスト
    new_classes = [] # NMS適用後のクラスのリスト
    
    while len(bboxes) > 0:
        # スコア最大の矩形のインデックスを取得
        argmax = scores.index(max(scores))
        
        VNum,HNum=0,0
        Center_Ave_V=np.zeros(2)#[0]=x,[1]=y
        xylength_Ave_V=np.zeros(2)
        Center_Ave_H=np.zeros(2)
        xylength_Ave_H=np.zeros(2)
        
        #bboxes[argmax]の縦横判定
        if (VorH(bboxes[argmax])==1):
            VNum+=1
            Center_Ave_V+=rectangle_center(bboxes[argmax])
            xylength_Ave_V+=rectangle_length(bboxes[argmax])
        elif(VorH(bboxes[argmax])==2):
            HNum+=1
            Center_Ave_H+=rectangle_center(bboxes[argmax])
            xylength_Ave_H+=rectangle_length(bboxes[argmax])
        
        #対象矩形のインデックス格納
        pop_i = []
        pop_i.append(argmax)

        for i, bbox_tmp in enumerate(bboxes):#添え字と要素でループ
            #argmaxでない+クラスが等しい+iouが閾値以上なら
            if  i != argmax and classes[i] == classes[argmax] and iou(bboxes[argmax], bbox_tmp) >= iou_threshold:
                pop_i.append(i)
                if (VorH(bboxes[i])==1):
                    VNum+=1
                    #縦長矩形のcenter座標平均の更新
                    Center_Ave_V+=rectangle_center(bbox_tmp)
                    #縦長矩形のxyの長さの平均の更新
                    xylength_Ave_V+=rectangle_length(bbox_tmp)
                        
                elif(VorH(bboxes[i])==2):
                    HNum+=1
                    #横長矩形のcenter座標平均の更新
                    Center_Ave_H+=rectangle_center(bbox_tmp)
                    #横長矩形のxyの長さの平均の更新
                    xylength_Ave_H+=rectangle_length(bbox_tmp)
                    

        if VNum != 0:
            Center_Ave_V=Center_Ave_V/VNum
            xylength_Ave_V=xylength_Ave_V/VNum
            
        if HNum != 0:
            Center_Ave_H=Center_Ave_H/HNum
            xylength_Ave_H=xylength_Ave_H/HNum

        if(VNum > HNum):
            created_bbox = creat_BBox(xylength_Ave_V,Center_Ave_V)
            new_bboxes.append(created_bbox)
            new_scores.append(-1)#スコアなし
            new_classes.append(classes[argmax])
        else:
            created_bbox = creat_BBox(xylength_Ave_H,Center_Ave_H)
            new_bboxes.append(created_bbox)
            new_scores.append(-1)#スコアなし
            new_classes.append(classes[argmax])
                  
        pop_i.sort()
        # 取得したインデックス(pop_i)の矩形、スコア値、クラスをそれぞれのリストから消去
        for i in pop_i[::-1]:
            bboxes.pop(i)
            scores.pop(i)
            classes.pop(i)

    return new_bboxes, new_scores, new_classes

def divide_BBox(bboxes, scores, classes):
    new_bboxes = [] 
    new_scores = [] 
    new_classes = [] 
    for i in range(len(scores)):
        L1=bboxes[i][2]-bboxes[i][0]
        L2=bboxes[i][0]+(img360_w-bboxes[i][2])
        if(L1<L2):
            new_bboxes.append(bboxes[i])
            new_scores.append(bboxes[i])
            new_classes.append(classes[i])
        else:
            print("L1>L2")
            lx1=0
            rx1=bboxes[i][2]
            lx2=bboxes[i][0]
            rx2=img360_w
            ly1=ry1=bboxes[i][1]
            ly2=ry2=bboxes[i][3]
              
            print("矩形の座標1　　左上："+str(lx1)+","+str(ly1)+" 右下："+str(lx2)+","+str(ly2))
            print("矩形の座標2　　左上："+str(rx1)+","+str(ry1)+" 右下："+str(rx2)+","+str(ry2))
            #↓矩形付け
            #cv2.rectangle(img,(thetal1,phil1), (thetal2,phil2), (255, 0, 0), thickness=10)
            #cv2.rectangle(img,(thetar1,phir1), (thetar2,phir2), (255, 0, 0), thickness=10)
              
            #配列に矩形情報を追加
            new_bboxes.append([lx1,ly1,lx2,ly2])
            new_scores.append(scores[i])
            new_classes.append(classes[i])
            
            new_bboxes.append([rx1,ry1,rx2,ry2])
            new_scores.append(scores[i])
            new_classes.append(classes[i]) 
        
    return new_bboxes, new_scores, new_classes

#for numH in range(1):
for numH in range(0,360,45):
    #for numV in range(1):
    for numV in range(45,-91,-45):
        conf_path = "C:/Users/s1936/ultralytics-main/runs/detect/GS__0043Gopro(25面)/labels/[theta]"+str(numH)+"[phi]"+str(numV)+".txt"
        is_file = os.path.isfile(conf_path)
        if is_file:
            phi_data="F:/4/卒業研究Ⅱ/py/360viewer/360out/GS__0043gopro25/csv/pvX-[theta]"+str(numH)+"[phi]"+str(numV)+'.csv'
            theta_data="F:/4/卒業研究Ⅱ/py/360viewer/360out/GS__0043gopro25/csv/pvY-[theta]"+str(numH)+"[phi]"+str(numV)+'.csv'
            df = pd.read_csv(conf_path,sep = ' ',header=None, names=['id', 'x1', 'y1', 'x2', 'y2','conf'])
            phif = pd.read_csv(phi_data,header=None,skiprows=0)
            thetaf = pd.read_csv(theta_data,header=None,skiprows=0)
            """
            img_w = 1920
            img_h = 1080
            """
            img_w=1366
            img_h=768
            
            print("\n[theta]"+str(numH)+"[phi]"+str(numV))
            for index in range(len(df)): #[0]id [1]x1 [2]y1 [3]x2 [4]y2 [5]conf
                  item = df.iloc[index]
                  if item[0] != -1:#id選択
                    # 中心と縦横幅
                    x_center = int(item[1]*img_w)
                    y_center = int(item[2]*img_h)
                    width = int(item[3]*img_w)
                    height = int(item[4]*img_h)
                    #左上と右下
                    x_min = int(x_center - width/2)
                    x_max = int(x_center + width/2)
                    y_min = int(y_center - height/2)
                    y_max = int(y_center + height/2)
                    #対応するthetaとphi
                    phi1 = int(phif.iloc[y_min,x_min])
                    theta1 = int(thetaf.iloc[y_min,x_min])
                    phi2 = int(phif.iloc[y_max,x_max])
                    theta2 = int(thetaf.iloc[y_max,x_max])
   
                    if theta1<theta2:
                        """
                        if(phi1>phi2):
                            phi1,phi2 = phi2,phi1
                        if(theta1>theta2):
                            theta1,theta2 = theta2,theta1
                        """
                        print("矩形",num_rect+1,"　id=",item[0])
                        print("中心と縦横幅",x_center,y_center,width,height)
                        print("左上と右下",x_min,y_min,x_max,y_max)                                
                        print("矩形の座標　　左上："+str(theta1)+","+str(phi1)+" 右下："+str(theta2)+","+str(phi2))
                            
                        #↓矩形付け
                        #cv2.rectangle(img,(theta1,phi1), (theta2,phi2), (255, 0, 0), thickness=10)
                    
                        #配列に矩形情報を追加
                        rects.append([theta1,phi1,theta2,phi2])
                        classes.append(item[0])
                        scores.append(item[5])
                        
                        num_rect=num_rect+1


print(f"\nnms処理前　矩形数{len(rects)}")

#nms処理
nms_rects, nms_scores, nms_classes = nms(rects, scores, classes,0.01)
print(f"\nnms処理後　矩形数{len(nms_rects)}")

nms_rects, nms_scores, nms_classes = divide_BBox(nms_rects, nms_scores, nms_classes)
print(f"\ndivideBBox後　矩形数{len(nms_rects)}")
   
for i in range(len(nms_scores)):
    cv2.rectangle(img,(nms_rects[i][0],nms_rects[i][1]),(nms_rects[i][2],nms_rects[i][3]), (0, 165, 255), thickness=25) 


#↓imgウィンドウ表示
img_resize = cv2.resize(img,None,None,0.3,0.3)
cv2.imshow('img', img_resize)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(r"C:\Users\s1936\ultralytics-main\runs\detect\detect _marge\NSMdemo_{num_rect}.jpg", img)
          