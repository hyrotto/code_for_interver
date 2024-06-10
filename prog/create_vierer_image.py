import numpy as np
import cv2
import csv

#for numH in range(0,360,45):
for numH in range(1):   
    #for numV in range(45,-91,-45):
    for numV in range(1):
        """
        img_w = 1920#出力画素数
        img_h = 1080
        """
        img_w=1366
        img_h=768
        
        senser_w = 0.75#三次元空間での画面サイズの半分
        senser_h = senser_w * img_h / img_w#0.421
        x1 = -1  # 視点の位置
        x2 = 0.85 + x1  #　撮像面の位置(必ず視点より前)
        
        w = np.arange(-senser_w, senser_w, senser_w * 2 / img_w)#画面画素数分の配列
        h = np.arange(-senser_h, senser_h, senser_h * 2 / img_h)#画面画素数
        
        
        
        # センサの座標
        ww, hh = np.meshgrid(w, h)#その座標における「w」若しくは「h」
    
        #回転させる角度roll=x軸回転(首をかしげる)
        #    ↓左の数字を変える(正面から何度動かすか)
        roll=0*np.pi/180
        pitch=numV*np.pi/180
        yaw=numH*np.pi/180
    
        #回転変換行列を計算し、後で位置ベクトルにかける
        rollMX=np.array([[1,0,0],
                         [0,np.cos(roll),-np.sin(roll)],
                         [0,np.sin(roll),np.cos(roll)]])
        
        pitchMX=np.array([[np.cos(pitch),0,-np.sin(pitch)],
                         [0,1,0],
                         [np.sin(pitch),0,np.cos(pitch)]])
    
        yawMX=np.array([[np.cos(yaw),-np.sin(yaw),0],
                        [np.sin(yaw),np.cos(yaw),0],
                        [0,0,1]])
    
        #上の三つの線形変換を合成
        rotaMX = np.dot(yawMX, np.dot(pitchMX, rollMX))
        #print(rotaMX)
    
        # 直線の式
        a1 = ww / (x2 - x1)
        a2 = hh / (x2 - x1)
       
        b1 = -a1 * x1
        b2 = -a2 * x1
     
        a = 1 + a1**2 + a2**2
        b = 2 * (a1 * b1 + a2 * b2)
        c = b1**2 + b2**2 - 1
     
        d = (b**2 - 4*a*c) ** (1/2)
     
        # 球面上の3次元座標
        x = (-b + d) / (2 * a)
        y = a1 * x + b1
        z = a2 * x + b2
    
        #ベクトル変換
        xd = rotaMX[0][0] * x + rotaMX[0][1] * y + rotaMX[0][2] * z
        yd = rotaMX[1][0] * x + rotaMX[1][1] * y + rotaMX[1][2] * z
        zd = rotaMX[2][0] * x + rotaMX[2][1] * y + rotaMX[2][2] * z
    
    
        # 緯度・経度へ変換
        phi = np.arcsin(zd)
        theta = np.arcsin(yd / np.cos(phi))
    
        xd[xd > 0] = 0
        xd[xd < 0] = 1
        yd[yd > 0] = np.pi
        yd[yd < 0] = -np.pi
        
        ofst = yd * xd
        gain = -2 * xd + 1
        theta = gain * theta + ofst
        #画像読み込み
        img = cv2.imread("GS__0043.JPG")
        img_h, img_w = img.shape[:2]#読み込んだ画像の行数,列数を取得
    
        #print(phi*180/np.pi)
        # 受け取った画像サイズに合わせて角度(画素位置)を再定義
        #φは縦
        phi = (phi * img_h / np.pi + img_h / 2).astype(np.float32)
        #print(phi)
        #ラジアン→画素位置に変換。画素数は下(南極)から数えたいので画素数の半分を足す
        #θは横
        theta = (theta * img_w / (2 * np.pi) + img_w / 2).astype(np.float32)
        #print(theta)
        
        #画像の生成（remap）には元画像
        out = cv2.remap(img, theta, phi, cv2.INTER_CUBIC)
        
        #filename = "360out/demo.jpeg"
        filename ="360out/GS__0043gopro25/[theta]"+str(numH)+"[phi]"+str(numV)+".jpeg"
        cv2.imwrite(filename,out)
        
        
        with open('360out/GS__0043Gopro25/csv/pvX-[theta]'+str(numH)+"[phi]"+str(numV)+'.csv', 'w') as file:
            writer = csv.writer(file, lineterminator='\n')
            writer.writerows(phi)
        with open('360out/GS__0043Gopro25/csv/pvY-[theta]'+str(numH)+"[phi]"+str(numV)+'.csv', 'w') as file:
            writer = csv.writer(file, lineterminator='\n')
            writer.writerows(theta)
        print("numH："+str(numH)+" numV："+str(numV)+"　OK")
        
        
        #ウィンドウ表示
        size=(960,540)
        img_resize = cv2.resize(out,   # 画像データを指定
                                size   # リサイズ後のサイズを指定
                               )
        cv2.imshow("dst.jpg", img_resize)
        cv2.waitKey(0)
        
