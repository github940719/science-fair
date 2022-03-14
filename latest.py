import math
import random
import numpy as np

# 基本資訊
people_num = 72            # 人數
size_x = 12                # 房間長
size_y = 12                # 房間寬
door = [np.array([2,2])]   # 出口位置
barrier = []               # 障礙物位置

# 力量參數(k:常數 r:感知範圍/時步 i:公式的次方)
k_door = 10                               # 自驅動力常數
[k_barrier,r_barrier,i_barrier] = [0,0,1] # 排斥障礙力
[k_people ,r_people ,i_people ] = [0,0,1] # 排斥人群力
[k_wall   ,r_wall   ,i_wall   ] = [0,0,1] # 排斥牆壁力
[k_foot   ,r_foot   ,i_foot   ] = [0,0,1] # 足跡力

# 其他模型
mess_bool   = False # 混亂模型
vacant_bool = False # 鑽空模型
conflict_p = 50     # 爭搶機率


# 向量設定
matrix   = np.array([[0,0,0] , [-2,-2,125] , [-2,0,90] , [-2,2,45] , \
           [0,-2,180] , [0,2,0] , [2,-2,225] , [2,0,270] , [2,2,325]])


# 輸入座標，回傳屬性
def attribute(pos):
    if (pos==0).any() or pos[0] >= 2*size_x+2 or pos[1] >= 2*size_y+2:
        return 'wall'
    for item in door:
        if (pos == item).all():
            return 'door'
    for item in barrier:
        if (pos == item).all():
            return 'barrier'
    return 'indoor'


# 計算方向角(弧度)
def count_theta(force):
    if   force[0] >  0 and force[1] > 0 : 
        theta = 2 * math.pi - math.atan(force[0] / force[1])
    elif force[0] >  0 and force[1] < 0 : 
        theta = math.pi + math.atan(force[0] / abs(force[1]))
    elif force[0] <  0 and force[1] < 0 : 
        theta = math.pi - math.atan(force[0] / force[1])
    elif force[0] <  0 and force[1] > 0 : 
        theta = math.atan(abs(force[0]) / force[1])
    elif force[0] >  0 and force[1] == 0: theta = math.pi * 3 / 2
    elif force[0] <  0 and force[1] == 0: theta = math.pi / 2
    elif force[0] == 0 and force[1] >0  : theta = 0
    elif force[0] == 0 and force[1] <0  : theta = math.pi
    return round(theta,2)


# 生成起始座標 輸出:座標list (index = self.tag-1)
def setup_pos():                                                     
    candidate=[]                                        
    for x in range(2,2*size_x+1,2):
        for y in range(2,2*size_y+1,2):
            if PSL.attri[x][y] == 'indoor' or PSL.attri[x][y] == 'tdoor':
                candidate.append(np.array([x,y]))

    number = [i for i in range(len(candidate))]
    rand = random.sample(number,people_num)
    global setup 
    setup = [None]*people_num
    for i in range(len(rand)):
        setup[i] = candidate[rand[i]]
    setup_act(setup)


# 在PSL.map_pos & PSL.map_tem中記錄起始座標
def setup_act(setup):
    for i in PSL.still:
        ev1[i].pos[0]   = setup[i-1][0]
        ev1[i].pos[1]   = setup[i-1][1]
        ev1[i].tem[0]   = ev1[i].pos[0]
        ev1[i].tem[1]   = ev1[i].pos[1]
        PSL.map_pos[setup[i-1][0]][setup[i-1][1]] = [i]
        PSL.map_tem[setup[i-1][0]][setup[i-1][1]] = [i]


# 檢查是否有重複座標
def conflict_check(): 

    def run():

        if True in [ev1[i].hold for i in PSL.map_tem[x][y]] :
            num_h = []
            for i in PSL.map_tem[x][y]:
                num_h.extend([i])
            for i in num_h:
                weight = [ev1[i].conflict_p,100-ev1[i].conflict_p]
                if random.choices([True,False],weights = weight)[0]  :
                    ev1[i].former_fail[0] = x
                    ev1[i].former_fail[1] = y
                hold(i)
            return

        num_c = []
        for i in PSL.map_tem[x][y]:
            num_c.extend([i])
        for i in num_c:
            weight = [100-ev1[i].conflict_p,ev1[i].conflict_p]
            if random.choices([True,False],weights = weight)[0] :
                hold(i)
                ev1[i].hold = True

        if len(PSL.map_tem[x][y]) > 1:  
            num_h = []
            for i in PSL.map_tem[x][y]:
                num_h.extend([i])
            for i in num_h:
                ev1[i].former_fail[0] = x
                ev1[i].former_fail[1] = y
                hold(i)
    
    while 1:                     
        break_botton = True
        for x in range(2,2*size_x+1):
            for y in range(2,2*size_y+1):
                if len(PSL.map_tem[x][y]) > 1:
                    break_botton = False
                    run()  

        if break_botton == True:
            return end_check()


# 延遲
def hold(i):
    ev1[i].hold   = True
    ev1[i].remove_tag()
    ev1[i].tem[0] = ev1[i].pos[0]
    ev1[i].tem[1] = ev1[i].pos[1]
    ev1[i].append_tag()


# 判斷抵達出口
def end_check():
    arrive=[]
    for i in PSL.still:
        ev1[i].move()
        for j in range(len(door)):
            if (ev1[i].pos == door[j]).all():
                PSL.finish += 1
                PSL.map_tem[ev1[i].pos[0]][ev1[i].pos[1]].remove(i)
                arrive.extend([i])
    for i in arrive:
        PSL.still.remove(i)
        ev1[i].out = PSL.step
    return arrive

class PSL:  

    finish   = 0    # 完成人數                       
    tag      = 0    # 編號累加
    step     = 0    # 第幾時步
    map_pos  = []   # 已確定的座標地圖(對應self.pos)
    map_tem  = []   # 未確定的座標地圖(對應self.tem)
    foot     = []   # 足跡的座標地圖(是否有人通過)
    attri    = []   # 座標的屬性地圖
    still    = []   # 仍在場上的人數

    # 初始化涵式
    def __init__(self):       
        self.tag         = PSL.tag                 # 個人編號       
        PSL.tag          += 1                      # 編號累加
        self.pos         = np.zeros(2,dtype = int) # 真實位置
        self.tem         = np.zeros(2,dtype = int) # 暫時位置

        # 力量參數
        self.foot_step = []                        # 有看足跡的時步
        self.force_dic = {'barrier': [k_barrier, r_barrier, i_barrier],             
                          'people' : [k_people , r_people , i_people ], 
                          'wall'   : [k_wall   , r_wall   , i_wall   ],
                          'foot'   : [k_foot   , r_foot   , i_foot   ],
                          'door'   : [k_door                         ]}

        # 模型
        self.mess_bool   = mess_bool               # 混亂模型
        self.vacant_bool = vacant_bool             # 鑽空模型
        self.conflict_p  = conflict_p              # 爭搶機率

        # 前輪數據
        self.former_best = np.array([-2,-2])       # 上一輪最佳走法
        self.former_bump = np.array([-2,-2])       # 上一輪撞牆/障礙座標
        self.former_fail = np.array([-2,-2])       # 上一輪爭搶失敗座標


    # 執行以下程式
    def carryout(self):
        self.step_reset()                

        if self.mess_bool :
            self.confusion() 
            return             

        if self.vacant_bool :
            self.vacant()          

        for type in self.force_dic:

            if type == 'door':       
                if self.force_dic['door'] != 0:     
                    nearest = min (range(len(door)) , key = lambda i : \
                    np.linalg.norm(self.pos - door[i])) 
                    self.count_force(self.pos,door[nearest],'door')                       
                    
            elif type == 'foot':     
                if self.force_dic['foot'][0] == 0:
                    self.foot_step = []
                else:
                    self.foot_step.extend([PSL.step])
                    self.footprint()

            else:                    
                if self.force_dic[type][0] != 0:
                    self.perception(type)

        self.force_order()            


    # 每一輪開始需要重置的數據
    def step_reset(self):
        self.index         = 0                         # 目前輪到第幾個走法
        self.hold          = False                     # 是否延遲(不論如何本輪保持原位)
        self.force         = np.zeros(2,dtype = float) # 力量(合力)
        self.pos_choice    = []                        # 走法順序
        self.choice_first  = []                        # (鑽空)沒有人的座標index
        self.choice_second = [1,2,3,4,5,6,7,8]         # (鑽空)有人的座標index


    # 混亂的模型:走法順序亂數 
    def confusion(self):
        order = [0,1,2,3,4,5,6,7,8]
        random.shuffle(order)
        for i in range(9):
            self.pos_choice.append(self.pos + matrix[order[i]][:2])
        self.find_choice()


    # 鑽空的模型:找沒有人的座標，從second移至first
    def vacant(self):
        for i in range(1,9):
            tem = self.pos + matrix[i][:2]
            if 'door' in PSL.attri[tem[0]][tem[1]]:
                if  PSL.map_pos[tem[0]][tem[1]][-2] == 0 :
                    self.choice_first.extend([i])
                    self.choice_second.remove(i)


    # 感知 輸入:力量類型，感知四周特定範圍 接續執行count force
    def perception(self,type):
        r = self.force_dic[type][1]

        def run():
            if type == 'people' and x != 0 and y != 0 and \
                    x != 2+2*size_x and y != 2+2*size_y:
                if PSL.map_pos[x][y][-2] != 0 and \
                (np.array([x,y]) != self.pos).any():
                    self.count_force(self.pos,np.array([x,y]),type)
            if PSL.attri[x][y] == type:  
                self.count_force(self.pos,np.array([x,y]),type) 

        for x in range(self.pos[0]-r,self.pos[0]+r+1,2):
            if 0 <= x <= 2*size_x+2:
                for y in range(self.pos[1]-r,self.pos[1]+r+1,2):
                    if 0 <= y <= 2*size_y+2 :
                        run()


    # 判定足跡:看PSL.foot，感知四周特定範圍 接續執行count force
    def footprint(self):
        if len(self.foot_step) > 1:
            if len(self.foot_step) <= self.force_dic['foot'][1] + 1: 
                start_index = self.foot_step[0]-1
            else: 
                start_index = (-self.force_dic['foot'][1]-1)

            for i in range(1,9):
                item = self.pos + matrix[i][:2]
                for j in PSL.foot[item[0]][item[1]][start_index:-1]:
                    if j != 0 and j != self.tag :
                        self.count_force(self.pos,item,'foot')


    # 計算力量 輸入:某人位置，目標位置，力量種類 執行:force累加
    def count_force(self,PSL_pos,target_pos,type):      
        rx = (target_pos[0] - PSL_pos[0])//2
        ry = (target_pos[1] - PSL_pos[1])//2
        r  = ((rx**2) + (ry**2))**0.5 
        if 'door' not in type:
            i  = self.force_dic[type][2]           
            if type == 'foot':
                self.force[0] += ((self.force_dic[type][0]/r**i) * (rx/r))
                self.force[1] += ((self.force_dic[type][0]/r**i) * (ry/r))
            else: 
                self.force[0] -= ((self.force_dic[type][0]/r**i) * (rx/r))
                self.force[1] -= ((self.force_dic[type][0]/r**i) * (ry/r))
        else:
            self.force[0] += (self.force_dic['door'][0]*(rx/r))
            self.force[1] += (self.force_dic['door'][0]*(ry/r)) 


    # 尋找各種走法(by力量)
    def force_order(self):          
        if self.force[0] == 0 and self.force[1] == 0:
            self.force_theta = None
            if self.choice_first != []:
                random.shuffle(self.choice_first)
            if self.choice_second != []:
                random.shuffle(self.choice_second)
            self.choice_order = self.choice_first + self.choice_second
            self.choice_order.insert(0,0)

        else:
            self.force_theta = count_theta(self.force)   
            if self.choice_first != []:
                a = sorted(self.choice_first,key = lambda i: \
                abs(round(matrix[i][2]/180*math.pi,2) - self.force_theta))
            else:
                a = []
            if self.choice_second != []:
                b = sorted(self.choice_second,key = lambda i \
                :abs(round(matrix[i][2]/180*math.pi,2) - self.force_theta))
            else:
                b = []
            self.choice_order = a + b
            self.choice_order.insert(4,0)

        for i in self.choice_order:
            self.pos_choice.append(self.pos+matrix[i][:2])

        self.find_choice()


    # 刪除世界座標tag
    def remove_tag(self):
        PSL.map_tem[self.tem[0]][self.tem[1]].remove(self.tag)
        if self.tem[0] != self.pos[0] or self.tem[1] != self.pos[1]:
            PSL.map_tem[(self.pos[0] + self.tem[0])//2] \
            [(self.pos[1] + self.tem[1])//2].remove(self.tag)


    # 將tag加入世界座標
    def append_tag(self):
        PSL.map_tem[self.tem[0]][self.tem[1]].extend([self.tag])
        if self.tem[0] != self.pos[0] or self.tem[1] != self.pos[1]:
            PSL.map_tem[(self.pos[0] + self.tem[0])//2] \
            [(self.pos[1] + self.tem[1])//2].extend([self.tag])


    # 找走法
    def find_choice(self):
        self.remove_tag()

        def if_or_not():
            temp = self.pos_choice[self.index]
            if 'door' not in PSL.attri[temp[0]][temp[1]] :
            # 撞到障礙物或牆壁
                if (temp == self.former_bump).all() :
                    # 前一輪也撞到 -> 直接找下一個走法
                    self.index += 1
                    if_or_not()
                    return
                else:
                    # 前一輪沒有撞到 -> 待在原位
                    self.hold   = True
                    self.tem[0] = self.pos[0]
                    self.tem[1] = self.pos[1]
                    self.former_fail    = np.array([-2,-2])
                    self.former_bump[0] = temp[0]
                    self.former_bump[1] = temp[1]
                    self.former_best    = np.array([-2,-2])
                    return

            if (temp == self.former_fail).all() or ((self.pos+temp)/2 == self.former_fail).all():
                # 現在要去的座標，是前一輪爭搶失敗的座標
                self.hold   = True
                self.tem[0] = self.pos[0]
                self.tem[1] = self.pos[1]
                self.former_fail    = np.array([-2,-2])
                self.former_bump    = np.array([-2,-2])
                self.former_best[0] = temp[0]
                self.former_best[1] = temp[1]
                return
            
            if (temp == self.former_best).all() and (self.force != 0).all():
                # 現在要去的座標，是上一輪的最佳座標(表上一輪退讓) -> by爭搶機率決定
                weight = [self.conflict_p , 100 - self.conflict_p]
               
                if random.choices([True,False], weights = weight)[0] :
                    # 決定要進去
                    self.former_fail = np.array([-2,-2])
                    self.former_bump = np.array([-2,-2])
                    self.former_best[0] = temp[0]
                    self.former_best[1] = temp[1]
                    self.tem[0]         = temp[0]
                    self.tem[1]         = temp[1] 
                    return

                # 決定不進去 -> 直接找下一個走法
                self.index += 1
                if_or_not()  
                return 

            # 沒有發生任何前述狀況 
            self.former_fail    = np.array([-2,-2])
            self.former_bump    = np.array([-2,-2])
            self.former_best[0] = temp[0]
            self.former_best[1] = temp[1]
            self.tem[0]         = temp[0]
            self.tem[1]         = temp[1]

        if_or_not()
        self.append_tag()


    # 移動
    def move(self):
        if self.pos[0] != self.tem[0] or self.pos[1] != self.tem[1]:
            PSL.foot[self.pos[0]][self.pos[1]][-1] = self.tag
            PSL.map_tem[(self.pos[0] + self.tem[0])//2] \
            [(self.pos[1] + self.tem[1])//2].remove(self.tag)
        self.pos[0] = self.tem[0]
        self.pos[1] = self.tem[1]
        PSL.map_pos[self.pos[0]][self.pos[1]][-1] = self.tag


# 主程式
ev1=[PSL() for i in range(people_num+1)]
setup_pos()

while PSL.finish < people_num :  
    PSL.step += 1    

    for x in range(2,2*size_x+1,2):
        for y in range(2,2*size_y+1,2):
            PSL.map_pos[x][y].extend([0])
            PSL.foot[x][y].extend([0])

    for i in PSL.still:
        ev1[i].carryout()

    conflict_check()