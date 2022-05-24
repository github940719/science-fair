import math
import random
import numpy as np


# 可調控變數
people_num      = 72
[size_x,size_y] = [12,12]
barrier         = []
k_door          = 10
[k_barrier,r_barrier] = [0,0]
[k_people ,r_people ] = [7,6]
[k_wall   ,r_wall   ] = [0,0]
[k_foot   ,r_foot   ] = [8,2]
mess_bool       = False
vacant_bool     = False
conflict_p      = 50
square          = True
squ_tag         = 0   # 廣場邊長大小index
exittag         = 0   # 廣場出口個數index
squ_out_p       = 0   # 出去廣場vs疲勞模式人數比例
turn_to_next    = 10  # 若廣場入口被霸占多久，該出口被封閉，無人後打開
ktired          = 2   # 疲勞模式常數


# 廣場(1)/地圖(0)資訊 (已經乘以2)
squ_size        = [12,13,14,15,16,17,18]  # 廣場邊長大小(尚未乘以2))
map_squ_fix     = [0,0,2,2,4,4,6]         # 房間到廣場的修正項(已經乘以2))


# 房間與廣場出口位置pos(已經乘以2)
exit_room = [[2],[2,24],[14],[8,18],[6,12,18],[4,10,16,22] \
            ,[4,8,12,16,20],[2,8,18,24]]
outside   = []
door      = []
squ_out_n = []
exit_squ  = []


# 向量設定(已經乘以2)
matrix   = np.array([[0,0,0],[0,2,0],[2,0,270],[0,-2,180],[2,-2,225] \
          ,[2,2,325],[-2,-2,135],[-2,0,90],[-2,2,45]])


# 每輪實驗重置
def ex_reset():

    global door
    door = []
    global exit_squ
    global squ_out_n

    # 出口位置
    for i in exit_room[exittag]:
        door.append(np.array([2,i,1]))  # [2] 1 = open 0 = closed
    exit_squ  = np.zeros((7,8), dtype = list)
    for i in range(7):
        for j in range(8):
            exit_squ[i][j] = [k + map_squ_fix[i] for k in exit_room[j]]


    # 廣場想要出去vs疲勞模式人數比例與人的編號設定 
    squ_out_n   = random.sample(range(1, people_num + 1), \
                int(people_num * squ_out_p/100))  

    for p in squ_out_n :
        ev1[p].tired = [False,False]
                
    PSL.finish    = 0                                            
    PSL.tag       = 0   
    PSL.step      = 0 
    PSL.in_room   = [i for i in range(1,people_num + 1)]
    PSL.in_squ    = []
    PSL.room_pos  = np.zeros((2*size_x+3,2*size_y+3), dtype = list)
    PSL.room_tem  = np.zeros((2*size_x+3,2*size_y+3), dtype = list)
    PSL.room_foot = np.zeros((2*size_x+3,2*size_y+3), dtype = list)
    PSL.room_att  = np.zeros((2*size_x+3,2*size_y+3), dtype = list)
    for x in range(2*size_x+3):
        for y in range(2*size_y+3):
            PSL.room_pos[x][y]     = [0]
            PSL.room_tem[x][y]     = []
            PSL.room_foot[x][y]    = []
            PSL.room_att[x][y]     = room_add_att(np.array([x,y]))

    n = squ_size[squ_tag]
    PSL.squ_pos  = np.zeros((2*n+3,2*n+3), dtype = list)
    PSL.squ_tem  = np.zeros((2*n+3,2*n+3), dtype = list)
    PSL.squ_foot = np.zeros((2*n+3,2*n+3), dtype = list)
    PSL.squ_att  = np.zeros((2*n+3,2*n+3), dtype = list)
    for x in range(2*squ_size[squ_tag]+3):
        for y in range(2*squ_size[squ_tag]+3):
            PSL.squ_pos[x][y]    = [0]
            PSL.squ_tem[x][y]    = []
            PSL.squ_foot[x][y]   = []
            PSL.squ_att[x][y]   = squ_add_att(np.array([x,y]))


# 輸入座標，回傳屬性
def room_add_att(pos):
    if (pos==0).any() or pos[0] >= 2*size_x+2 or pos[1] >= 2*size_y+2:
        return 'wall'
    for item in door:
        if pos[0] == item[0] and pos[1] == item[1] :
            return 'door'
    for item in barrier:
        if (pos == item).all():
            return 'barrier'
    return 'indoor'


# 輸入座標，回傳屬性
def squ_add_att(pos):
    global outside
    if pos[0] == 1*2: 
        for i in exit_squ[squ_tag][exittag]:
            if pos[1] == i:
                return 'enter'
    if pos[0] == 0:
        return 'wall'
    if pos[0] == 2*squ_size[squ_tag]+2 or pos[1] == 0 or \
        pos[1] == 2*squ_size[squ_tag]+2:
        outside.extend([np.array([pos[0],pos[1]])])
        return 'outdoor'

    return 'indoor'


# 計算方向角(弧度)
def count_theta(force):
    if   force[0] >  0 and force[1] > 0 : 
        return round(2 * math.pi - math.atan(force[0] / force[1]),5)
    elif force[0] >  0 and force[1] < 0 : 
        return round(math.pi + math.atan(- force[0] / force[1]),5)
    elif force[0] <  0 and force[1] < 0 : 
        return round(math.pi - math.atan(force[0] / force[1]),5)
    elif force[0] <  0 and force[1] > 0 : 
        return round(math.atan(-force[0] / force[1]),5)
    elif force[0] >  0 and force[1] == 0: return round(math.pi * 3 / 2,5)
    elif force[0] <  0 and force[1] == 0: return round(math.pi / 2,5)
    elif force[0] == 0 and force[1] >0  : return 0
    elif force[0] == 0 and force[1] <0  : return round(math.pi,5)


# 足跡地圖新增0
def foot_append():
    for x in range(2,2*size_x+1,2):
        for y in range(2,2*size_y+1,2):
            PSL.room_pos[x][y].extend([0])
            PSL.room_foot[x][y].extend([0])

    for x in range(2,2*squ_size[squ_tag]+1,2):
        for y in range(2,2*squ_size[squ_tag]+1,2):
            PSL.squ_pos[x][y].extend([0])


# 生成起始座標 輸出:座標list (index = self.tag)
def setup_pos():                                                     
    candidate=[]                                        
    for x in range(2,2*size_x+1,2):
        for y in range(2,2*size_y+1,2):
            if PSL.room_att[x][y] == 'indoor' :
                candidate.append(np.array([x,y]))

    number = [i for i in range(len(candidate))]
    rand = random.sample(number,people_num)
    global setup 
    setup = [None]*people_num
    for i in range(len(rand)):
        setup[i] = candidate[rand[i]]
    setup_act(setup)


# 在PSL.room_pos & PSL.room_tem中記錄起始座標
def setup_act(setup):
    for i in PSL.in_room:
        ev1[i].pos[0]   = setup[i-1][0]
        ev1[i].pos[1]   = setup[i-1][1]
        ev1[i].pos[2]   = 0
        ev1[i].tem[0]   = ev1[i].pos[0]
        ev1[i].tem[1]   = ev1[i].pos[1]
        ev1[i].tem[2]   = 0
        t = ev1[i].pos
        ev1[i].history.extend([np.array([t[0],t[1],t[2]])])
        PSL.room_pos[setup[i-1][0]][setup[i-1][1]] = [i]
        PSL.room_tem[setup[i-1][0]][setup[i-1][1]] = [i]


# 檢查是否有重複座標
def conflict_check(): 

    def run(map_tem,x,y):

        # 有人本輪必須hold 其他人也必須hold
        if True in [ev1[i].hold for i in map_tem[x][y]] :
            num_h = []
            for i in map_tem[x][y]:
                num_h.extend([i])
            for i in num_h:
                weight = [ev1[i].conflict_p,100-ev1[i].conflict_p]
                if random.choices([True,False],weights = weight)[0] :
                    ev1[i].former_fail[0] = x
                    ev1[i].former_fail[1] = y
                    ev1[i].former_fail[2] = ev1[i].tem[2]
                hold(i)
            return

        # 決定爭搶或退讓
        num_c = []
        for i in map_tem[x][y]:
            num_c.extend([i])
        for i in num_c:
            weight = [100-ev1[i].conflict_p,ev1[i].conflict_p]
            if random.choices([True,False],weights = weight)[0] :
                hold(i)
                ev1[i].hold = True

        # 爭搶人數仍超過1 爭搶失敗 hold 兩輪
        if len(map_tem[x][y]) > 1:  
            num_h = []
            for i in map_tem[x][y]:
                num_h.extend([i])
            for i in num_h:
                ev1[i].former_fail[0] = x
                ev1[i].former_fail[1] = y
                ev1[i].former_fail[2] = ev1[i].tem[2]
                hold(i)


    while 1 :
        break_botton = [1,1]
        for a in range(2):
            if (PSL.finish == 0 and a == 1) :
                break
            if (a == 1 and not square):
                break
            if a == 0: 
                map_tem = PSL.room_tem
                s_x     = size_x
                s_y     = size_y
            else: 
                map_tem = PSL.squ_tem     
                s_x     = squ_size[squ_tag]
                s_y     = squ_size[squ_tag]    

            for x in range(2*s_x+3):
                for y in range(2*s_y+3):
                    if len(map_tem[x][y]) > 1:
                        break_botton[a] = 0
                        run(map_tem,x,y)  

        if break_botton[0] * break_botton[1] == 1:
            return end_check()


# 延遲
def hold(i):
    ev1[i].hold   = True
    ev1[i].remove_tag()
    ev1[i].tem[0] = ev1[i].pos[0]
    ev1[i].tem[1] = ev1[i].pos[1]
    ev1[i].tem[2] = ev1[i].pos[2]
    ev1[i].append_tag()


# 判斷抵達出口
def end_check():

    need_remove = [[],[]]  # 房間 廣場

    # 廣場
    for i in PSL.in_squ :
        ev1[i].move()

        if PSL.squ_att[ev1[i].pos[0]][ev1[i].pos[1]] == 'outdoor' :
            PSL.squ_tem[ev1[i].pos[0]][ev1[i].pos[1]].remove(i)
            need_remove[1].extend([i])

        elif ev1[i].tired[0] == True and ev1[i].moved >= ev1[i].squ_max :
            ev1[i].tired[1] = True
            for j in exit_squ[squ_tag][exittag]:
                if ev1[i].pos[0] == 2 and ev1[i].pos[1] == j :
                    ev1[i].moved -= 1
                    ev1[i].tired[1] = False
                    break
    

    # 房間
    for i in PSL.in_room:
        ev1[i].move()
        if  ev1[i].pos[2] == 1 :
            need_remove[0].extend([i])
            ev1[i].squ_max = k_tired(ev1[i].moved)
            ev1[i].moved   = 0
            PSL.finish     += 1

        elif not square :
            for item in door:
                if ev1[i].pos[0] == item[0] and ev1[i].pos[1] == item[1]:
                    need_remove[0].extend([i])
                    ev1[i].remove_tag()
                    PSL.finish += 1
                    break
        
    for i in need_remove[0]:
        if square : 
            PSL.in_squ.extend([i])
        PSL.in_room.remove(i)
    for i in need_remove[1]:
        PSL.in_squ.remove(i)

    return 


# 疲勞模式公式
def k_tired(x):
    return ktired



class PSL:  

    finish    = 0   # 完成人數                       
    tag       = 0   # 編號累加
    step      = 0   # 第幾時步
    room_pos  = []  # 已確定的房間地圖(對應self.pos)
    squ_pos   = []  # 已確定的廣場地圖(對應self.pos)
    room_tem  = []  # 未確定的房間地圖(對應self.tem)
    squ_tem   = []  # 未確定的廣場地圖(對應self.tem)
    room_att  = []  # 房間的屬性地圖
    squ_att   = []  # 廣場的屬性地圖
    room_foot = []  # 足跡的房間地圖(是否有人通過)
    squ_foot  = []  # 足跡的廣場地圖(是否有人通過)
    in_room   = []  # 仍在房間的人數
    in_squ    = []  # 仍在廣場的人數
    open_door = []


    # 初始化涵式
    def __init__(self):       

        self.tag     = PSL.tag                 # 個人編號       
        PSL.tag      += 1                      # 編號累加
        self.pos     = np.zeros(3,dtype = int) # 真實位置 [x,y,房間0;廣場1]
        self.tem     = np.zeros(3,dtype = int) # 暫時位置 [x,y,房間0;廣場1]
        self.moved   = 0            # 已經移動幾步(房間到廣場歸零)
        self.squ_max = 0            # 廣場中最多可走幾步
        self.tired   = [True,False] # 疲勞(T)或走出去(F)模式
        self.history = []           # 軌跡歷史


        # 力量參數
        self.foot_step = []                   # 有看足跡的時步
        self.force_dic = {'barrier': [k_barrier, r_barrier],             
                          'people' : [k_people , r_people ], 
                          'wall'   : [k_wall   , r_wall   ],
                          'foot'   : [k_foot   , r_foot   ],
                          'door'   : [k_door              ]}

        # 模型
        self.mess_bool   = mess_bool          # 混亂模型
        self.vacant_bool = vacant_bool        # 鑽空模型
        self.conflict_p  = conflict_p         # 爭搶機率

        # 前輪數據
        self.former_best = np.array([-2,-2,-2])  # 上一輪最佳走法
        self.former_bump = np.array([-2,-2,-2])  # 上一輪撞牆/障礙座標
        self.former_fail = np.array([-2,-2,-2])  # 上一輪爭搶失敗座標


    # 執行以下程式(房間)
    def carryout_room(self):
        self.step_reset()                

        # 混亂模型
        if self.mess_bool :
            self.confusion() 
            return             

        # 位置在房間出口 不是前進廣場 就是停留房間出口
        for item in door:
            if self.pos[0] == item[0] and self.pos[1] == item[1] and \
                square :
                self.remove_tag()
                self.tem[0] = 2
                self.tem[1] = self.pos[1] +  map_squ_fix[squ_tag]
                self.tem[2] = 1
                self.append_tag()
                return

        # 鑽空模型
        if self.vacant_bool :
            self.vacant()          

        for type in self.force_dic:

            # 自驅動力
            if type == 'door':     
                if PSL.open_door != []:
                    nearest = min (range(len(PSL.open_door)) , key = \
                    lambda i : np.linalg.norm(self.pos[:2] - PSL.open_door[i])) 
                    self.count_force(self.pos,door[nearest],'door')                      
                    
            # 足跡力
            elif type == 'foot':     
                if self.force_dic['foot'][0] == 0:
                    self.foot_step = []
                else:
                    self.foot_step.extend([PSL.step])
                    self.footprint()

            # 排斥力
            else:                    
                if self.force_dic[type][0] != 0:
                    self.perception(type)

        # 依照合力方向角差值排走法順序
        self.force_order()  


    # 執行以下程式(廣場)
    def carryout_squ(self):
        self.step_reset()

        # 走向廣場外圈模式 : 類似自驅動力
        if not self.tired[0] :
            nearest = min (range(len(outside)) , key = lambda i : \
            np.linalg.norm(self.pos[:2] - outside[i])) 
            self.count_force(self.pos, outside[nearest], 'door') 
        
        # 霸佔廣場入口: 強制離開(不撞牆為前提)
        for item in exit_squ[squ_tag][exittag]:
            if self.pos[0] == 2 and self.pos[1] == item:
                self.vacant()
                self.mustleave = True

        # 已達最大步數 : 自此後維持原位
        if self.mustleave == False:
            if self.tired[1] :
                for i in range(9):
                    self.pos_choice[i][0] = self.pos[0]
                    self.pos_choice[i][1] = self.pos[1]
                    self.pos_choice[i][2] = 1
                return
                

        # 只開設排斥人群力       
        if self.force_dic['people'][0] != 0:
            self.perception('people')

        # 依照合力方向角差值排走法順序
        self.force_order() 


    # 每一輪開始需要重置的數據
    def step_reset(self):
        self.index      = 0                 # 目前輪到第幾個走法
        self.hold       = False             # 是否延遲(不論如何本輪保持原位)
        self.force      = np.zeros(2,dtype = float) 
        self.pos_choice = np.zeros((9,3), dtype = int)
        self.choice_1   = []                # (鑽空)沒有人的座標index
        self.choice_2   = []                # 廣場擋住入口用 1~5
        self.choice_3   = [1,2,3,4,5,6,7,8] # (鑽空)有人的座標index
        self.mustleave  = False             # 是否在廣場中堵住出口


    # 混亂的模型:走法順序亂數 
    def confusion(self):
        order = [0,1,2,3,4,5,6,7,8]
        random.shuffle(order)
        for i in range(9):
            self.pos_choice[i][0] = self.pos[0] + matrix[order[i]][0]
            self.pos_choice[i][1] = self.pos[1] + matrix[order[i]][1]
            self.pos_choice[i][2] = self.pos[2]
        self.find_choice()


    # 鑽空的模型:找沒有人的座標，從second移至first
    def vacant(self):
        if self.pos[2] == 0 : 
            map_pos = PSL.room_pos
            map_att = PSL.room_att
        else : 
            map_pos = PSL.squ_pos
            map_att = PSL.squ_att
            
        for i in range(1,9):
            tem = self.pos[:2] + matrix[i][:2]
            # 檢查indoor or door 的座標是否是空位
            if 'door' in map_att[tem[0]][tem[1]] :
                if map_att[tem[0]][tem[1]] != 'outdoor':
                    if  map_pos[tem[0]][tem[1]][-2] == 0 :
                        self.choice_1.extend([i])
                        self.choice_3.remove(i)

                elif map_att[tem[0]][tem[1]] == 'outdoor' and \
                    self.tired[0] == False:
                    self.choice_1.extend([i])


    # 感知 輸入:力量類型，感知四周特定範圍 接續執行count force
    def perception(self,type):
        r = self.force_dic[type][1]

        def run(map_pos,map_att):

            # 排斥人群力:檢查是否有人
            if type == 'people' :
                if map_pos[x][y][-2] != 0 and \
                (np.array([x,y]) != self.pos[:2]).any():
                    self.count_force(self.pos,np.array([x,y]),type)

            # 其他排斥力:屬性是否和目標物相符
            if map_att[x][y] == type:  
                self.count_force(self.pos,np.array([x,y]),type) 

        if self.pos[2] == 0 :
            s_x = size_x
            s_y = size_y
            map_pos = PSL.room_pos
            map_att = PSL.room_att
        else:
            s_x = squ_size[squ_tag]
            s_y = squ_size[squ_tag]
            map_pos = PSL.squ_pos
            map_att = PSL.squ_att

        # 先找到界內座標
        for x in range(self.pos[0]-r,self.pos[0]+r+1,2):
            if 2 <= x <= 2*s_x:
                for y in range(self.pos[1]-r,self.pos[1]+r+1,2):
                    if 2 <= y <= 2*s_y :
                        run(map_pos,map_att)


    # 判定足跡:看PSL.foot，感知四周特定範圍 接續執行count force
    def footprint(self):

        if len(self.foot_step) > 1:

            # 先確認採計足跡step是否超過感知時步
            if len(self.foot_step) <= self.force_dic['foot'][1] + 1: 
                start_index = self.foot_step[0] - 1
            else: 
                start_index = (-self.force_dic['foot'][1]-1)

            if self.pos[2] == 0 : map_foot = PSL.room_foot
            else                : map_foot = PSL.squ_foot

            # 尋找足跡
            for i in range(1,9):
                item = self.pos[:2] + matrix[i][:2]
                for j in map_foot[item[0]][item[1]][start_index:-1]:
                    if j != 0 and j != self.tag :
                        self.count_force(self.pos,item,'foot')


    # 計算力量 輸入:某人位置，目標位置，力量種類 執行:force累加
    def count_force(self,PSL_pos,target_pos,type):      
        rx = (target_pos[0] - PSL_pos[0])//2
        ry = (target_pos[1] - PSL_pos[1])//2
        r  = ((rx**2) + (ry**2))**0.5 

        if r == 0 :
            return

        if 'door' not in type:  

            # 足跡力:吸引力       
            if type == 'foot':
                self.force[0] += ((self.force_dic[type][0]/r) * (rx/r))
                self.force[1] += ((self.force_dic[type][0]/r) * (ry/r))
            
            # 排斥力
            else: 
                self.force[0] -= ((self.force_dic[type][0]/r) * (rx/r))
                self.force[1] -= ((self.force_dic[type][0]/r) * (ry/r))
        
        # 自驅動力:吸引力 無關距離
        else:
            self.force[0] += (self.force_dic['door'][0]*(rx/r))
            self.force[1] += (self.force_dic['door'][0]*(ry/r)) 


    # 尋找各種走法(by力量)
    def force_order(self):   

        # 合力為0: 看鑽空 
        if self.force[0] == 0 and self.force[1] == 0:
            if self.choice_1 != []:
                random.shuffle(self.choice_1)

            # 廣場中：5種前進步撞牆走法亂數 剩下的再亂數 留在原位插入第六順位
            if self.pos[2] == 1 and self.mustleave == True:
                self.choice_2 = [1,2,3,4,5]
                if self.choice_1 != []:
                    for g in self.choice_1 :
                        self.choice_2.remove(g)
                if self.choice_2 != []:
                    random.shuffle(self.choice_2)
                self.choice_3 = [6,7,8]
                random.shuffle(self.choice_3)
                self.choice_order = self.choice_1  + self.choice_2 + self.choice_3
                self.choice_order.insert(5,0)

            # 房間內：其餘亂數 最佳走法為留在原位
            else:
                if self.choice_3 != []:
                    random.shuffle(self.choice_3)
                self.choice_order = self.choice_1  + self.choice_3
                self.choice_order.insert(0,0)


        else:

            # 先算方向角
            self.force_theta = count_theta(self.force)  

            # 依照方向角差值小到大排序
            if self.choice_1 != []:
                self.choice_1 = sorted(self.choice_1,key = lambda i: \
                abs(round(matrix[i][2]/180*math.pi,2) - self.force_theta))

            # 廣場：前進不撞牆走法依照方向角排序 插入留在原位 剩下走法再亂數
            if self.pos[2] == 1 and self.mustleave == True:
                self.choice_2 = [1,2,3,4,5]
                if self.choice_1 != []:
                    for g in self.choice_1 :
                        self.choice_2.remove(g)

                if self.choice_2 != []:
                    self.choice_2 = sorted(self.choice_2,key = lambda i: \
                    abs(round(matrix[i][2]/180*math.pi,2) - self.force_theta))

                self.choice_3 = [6,7,8]
                random.shuffle(self.choice_3)
                self.choice_order = self.choice_1 + self.choice_2 + self.choice_3
                self.choice_order.insert(5,0)

            # 房間：choice 1 & 3各自排序 中間插入留在原位
            else:
                if self.choice_3 != []:
                    self.choice_3 = sorted(self.choice_3,key = lambda i \
                    :abs(round(matrix[i][2]/180*math.pi,2) - self.force_theta))

                self.choice_order = self.choice_1 + self.choice_3
                self.choice_order.insert(4,0)


        # 建立走法順序
        for i in range(len(self.choice_order)):
            k = self.choice_order[i]
            self.pos_choice[i][0] = self.pos[0] + matrix[k][0]
            self.pos_choice[i][1] = self.pos[1] + matrix[k][1]
            self.pos_choice[i][2] = self.pos[2]

        self.find_choice()


    # 刪除世界座標tag
    def remove_tag(self):
        if self.tem[2] == 0 : map_tem = PSL.room_tem
        else : map_tem = PSL.squ_tem

        map_tem[self.tem[0]][self.tem[1]].remove(self.tag)

        # 中點座標移除
        if (self.tem[0] != self.pos[0] or self.tem[1] != self.pos[1]) \
            and (self.pos[2] == self.tem[2]):
            map_tem[(self.pos[0] + self.tem[0])//2] \
            [(self.pos[1] + self.tem[1])//2].remove(self.tag)
                

    # 將tag加入世界座標
    def append_tag(self):
        if self.tem[2] == 0 : map_tem = PSL.room_tem
        else  : map_tem = PSL.squ_tem
        map_tem[self.tem[0]][self.tem[1]].extend([self.tag])

        # 中點座標加入
        if (self.tem[0] != self.pos[0] or self.tem[1] != self.pos[1]) \
            and (self.pos[2] == self.tem[2]):
            map_tem[(self.pos[0] + self.tem[0])//2] \
            [(self.pos[1] + self.tem[1])//2].extend([self.tag])


    # 找走法
    def find_choice(self):
        self.remove_tag()

        # 設定 tem = temp
        def set_tem(temp):
            self.tem[0]  = temp[0]
            self.tem[1]  = temp[1]
            self.tem[2]  = temp[2]

        # 設定 former_best = temp
        def set_f_best(temp):
            self.former_best[0] = temp[0]
            self.former_best[1] = temp[1]
            self.former_best[2] = temp[2]


        # 設定 tem = pos
        def set_hold():
            self.tem[0] = self.pos[0]
            self.tem[1] = self.pos[1]
            self.tem[2] = self.pos[2]

        # 最佳走法流程圖
        def if_or_not():
            
            if self.tem[2] == 0 : map_att = PSL.room_att
            else : map_att = PSL.squ_att
            temp = self.pos_choice[self.index]

            # 撞到障礙物或牆壁
            if 'door' not in map_att[temp[0]][temp[1]] and \
                map_att[temp[0]][temp[1]] != 'enter':

                # 前一輪也撞到 -> 直接找下一個走法
                if (temp == self.former_bump).all() :
                    self.index += 1
                    if_or_not()
                    return

                # 前一輪沒有撞到 -> 待在原位
                else:
                    self.hold   = True
                    set_hold()
                    self.former_fail    = np.array([-2,-2,-2])
                    self.former_bump[0] = temp[0]
                    self.former_bump[1] = temp[1]
                    self.former_bump[2] = temp[2]
                    self.former_best    = np.array([-2,-2,-2])
                    return

            # 現在要去的座標，是前一輪爭搶失敗的座標
            if (temp == self.former_fail).all() or ((self.pos+temp)/2 \
                == self.former_fail).all():
                self.hold   = True
                set_hold()
                self.former_fail    = np.array([-2,-2,-2])
                self.former_bump    = np.array([-2,-2,-2])
                set_f_best(temp)
                return
            
            # 現在要去的座標是上一輪最佳座標(上一輪退讓) -> by爭搶機率決定
            if (temp == self.former_best).all() and (self.force != 0).all():
                weight = [self.conflict_p ,100 - self.conflict_p]
                
                # 決定要進去
                if random.choices([True,False], weights = weight)[0] :
                    self.former_fail = np.array([-2,-2,-2])
                    self.former_bump = np.array([-2,-2,-2])
                    set_f_best(temp)
                    set_tem(temp)
                    return

                # 決定不進去 -> 直接找下一個走法
                self.index += 1
                if_or_not()  
                return 

            # 沒有發生任何前述狀況 
            self.former_fail    = np.array([-2,-2,-2])
            self.former_bump    = np.array([-2,-2,-2])
            set_f_best(temp)
            set_tem(temp)

        if_or_not()
        self.append_tag()


    # 移動
    def move(self):
        if self.pos[2] == 0: map_foot = PSL.room_foot
        else : map_foot = PSL.squ_foot

        if self.tem[2] == 0: map_tem = PSL.room_tem
        else : map_tem = PSL.squ_tem

        # 足跡標記(打開出口?) 計算移動步數 移除tem地圖中點座標標記
        if self.pos[0] != self.tem[0] or self.pos[1] != self.tem[1] \
        or self.pos[2] != self.tem[2] :
            if self.pos[2] == 0:
                map_foot[self.pos[0]][self.pos[1]][-1] = self.tag
            if self.pos[2] == self.tem[2] :
                self.moved += 1
                map_tem[(self.pos[0] + self.tem[0])//2] \
                [(self.pos[1] + self.tem[1])//2].remove(self.tag)

                for e in range(len(door)):
                    item = door[e]
                    if self.pos[0] == item[0] and self.pos[1] == item[1]:
                        if door[e][2] == 0:
                            door[e][2] = 1

        self.pos[0] = self.tem[0]
        self.pos[1] = self.tem[1]
        self.pos[2] = self.tem[2]
        self.history.extend([np.array([self.pos[0],self.pos[1],self.pos[2]])])

        # 霸佔出口，設定出口closed
        if self.pos[2] == 1 and len(self.history) > 10:
            for d in range(len(exit_squ[squ_tag][exittag])):
                item = exit_squ[squ_tag][exittag][d]
                if self.pos[0] == 2 and self.pos[1] == item:
                    same = 1
                    for c in range(1,9):
                        if (self.history[-c] == self.history[-c-1]).all():
                            same *= 1
                        else: 
                            same *= 0
                            break
                    if same == 1:
                        door[d][2] = 0


        # pos 地圖標記
        if self.pos[2] == 0: map_pos = PSL.room_pos
        else : map_pos = PSL.squ_pos
        map_pos[self.pos[0]][self.pos[1]][-1] = self.tag
        if self.pos[2] == 0: map_foot = PSL.room_foot
        else : map_foot = PSL.squ_foot

        if self.tem[2] == 0: map_tem = PSL.room_tem
        else : map_tem = PSL.squ_tem

        # 足跡標記(打開出口?) 計算移動步數 移除tem地圖中點座標標記
        if self.pos[0] != self.tem[0] or self.pos[1] != self.tem[1] \
        or self.pos[2] != self.tem[2] :
            map_foot[self.pos[0]][self.pos[1]][-1] = self.tag
            if self.pos[2] == self.tem[2] :
                self.moved += 1
                map_tem[(self.pos[0] + self.tem[0])//2] \
                [(self.pos[1] + self.tem[1])//2].remove(self.tag)

                for e in range(len(door)):
                    item = door[e]
                    if self.pos[0] == item[0] and self.pos[1] == item[1]:
                        if door[e][2] == 0:
                            door[e][2] = 1

        self.pos[0] = self.tem[0]
        self.pos[1] = self.tem[1]
        self.pos[2] = self.tem[2]
        self.history.extend([np.array([self.pos[0],self.pos[1],self.pos[2]])])

        # 霸佔出口，設定出口closed
        if self.pos[2] == 1 and len(self.history) > 10:
            for d in range(len(exit_squ[squ_tag][exittag])):
                item = exit_squ[squ_tag][exittag][d]
                if self.pos[0] == 2 and self.pos[1] == item:
                    same = 1
                    for c in range(1,9):
                        if (self.history[-c] == self.history[-c-1]).all():
                            same *= 1
                        else: 
                            same *= 0
                            break
                    if same == 1:
                        door[d][2] = 0


        # pos 地圖標記
        if self.pos[2] == 0: map_pos = PSL.room_pos
        else : map_pos = PSL.squ_pos
        map_pos[self.pos[0]][self.pos[1]][-1] = self.tag

# --------------------------

ev1 = [PSL() for i in range(people_num + 1)]
ex_reset()
setup_pos()

while PSL.finish < people_num :  
    PSL.step += 1 

    foot_append()

    PSL.open_door = []
    for item in door:
        if item[2] == 1: PSL.open_door.extend([item[:2]])
    
    if PSL.open_door == [] :
        closed += 1
        if closed >= 50 : print('wrong')

    else : closed = 0

    for i in PSL.in_room:
        ev1[i].carryout_room()

    for i in PSL.in_squ:
        ev1[i].carryout_squ()

    conflict_check()

print(PSL.step)