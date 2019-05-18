# Binary GA

########################   Import Model  ######################################
import numpy as np                                                              # 計算矩陣使用
import math                                                                     # 計算根號與sin使用
import matplotlib.pyplot as plt 
                 
###########################   Parameter  ######################################
population=100                                                                  # 人口數
generation=100                                                                    # 迭代次數
N=10                                                                            # 維度
gene_length=10                                                                  # 儲存每維度的變數在[-512,511]要2^10
Tournament_n=2                                                                  # 每次2人競爭
crossover_rate=0.9                                                              # 交配率
record_average_fitness=[]                                                       # 紀錄每次迭代後的平均適應值

#############################   函式   ########################################

## 解碼 Decode
def decode(gene):
    gene_length=len(gene)                                                       # 取得基因長度
    variable=0                                                                  # 轉換後的值(10進位)
    power=gene_length-1                                                         # 從最高冪次開始
    for i in range(0,gene_length):                         
        variable+=(2**power)*gene[i]                                            # 將binary乘上2的冪次方
        power-=1                                                                # 下降1個冪次
        # 因為陣列的第一個元素為MSB,最後一個元素為LSB,故冪次要反過來乘
    return variable

## 計算cost function 
def SCH_cost_function(x):
    # 先計算後面sigma項
    cost=0
    for i in range(0,len(x)):
        cost+=x[i]*math.sin(math.sqrt(abs(x[i])))
    cost=418.98291*len(x)-cost
    return cost

## 計算所有粒子的適應值
def fitness(swarm,N,gene_length):   
    swarm_fitness=np.zeros([np.size(swarm,0),1])                                # 紀錄每個粒子目前的適應值
    for swarm_number in range(0,np.size(swarm,0)):        
        phenotype=[]                                                            # 暫存這個粒子的基因型
        for dimension in range(0,N):                                            # 依序處理每個維度(分成N段來處理)
            start=10*dimension                                                  # 該維度起始點對應第幾個基因
            gene_segment=swarm[swarm_number,start:start+gene_length]            # 取出片段基因
            phenotype.append(decode(gene_segment.tolist())-512)                 # 將array轉成list,並進行解碼,放入phenotype  
        swarm_fitness[swarm_number]=SCH_cost_function(phenotype)
    return swarm_fitness                                                        # 全部粒子的適應值
            
## Tournament Selection
def Tournament_Selection(swarm,swarm_fitness,Tournament_n):
    crossover_candidate_index=[]                                                # 最後被選為交配者的名單(所在index)
    for selection in range(0,np.size(swarm,0)):                                 # 選出與目前人口數相同的粒子
        # 隨機找n=Tournament_n人來做競爭
        candidate=[]                                                            # 被選中要競爭的候選人
        candidate_fitness=[]                                                    # 被選中要競爭的候選人的適應值
        find_start=0                                                            # 開始尋找的起始點(避免重複尋找)
        for find_candidate in range(0,Tournament_n):                            # 要找到Tournament_n個人
            rand_find=np.random.randint(find_start,np.size(swarm,0)-Tournament_n+find_candidate) 
            candidate.append(rand_find)                                         # 儲存此人的index
            candidate_fitness.append(swarm_fitness[rand_find])                  # 儲存此人的fitness
            find_start=rand_find
        # 候選人做競爭
        best=candidate_fitness.index(min(candidate_fitness))                    # 找出fitness最小者是第幾位競爭者
        crossover_candidate_index.append(candidate[best])                       # 放入交配者名單(他的index)
    return crossover_candidate_index

## Crossover 2-point
def Crossover_two_point(parent1,parent2,crossover_rate):
    if np.random.random()<=crossover_rate:                                      # 隨機產生0~1的值若小於交配率,則交配
        # 產生的2個子代
        children1=[]
        children2=[]
        # 隨機產生2個切點
        rand1=np.random.randint(0,len(parent1)-1)                                   # 在基因段當中,隨機找到某點切點
        rand2=np.random.randint(rand1+1,len(parent1))                               # 目的是要使rand1不會是最後一點,rand2必定在rand1後面而不會重複   
        # 開始分配基因給子代
        children1.extend(parent1[:rand1])                                           # 在(rand1-1)以前(含)的同父親
        children2.extend(parent2[:rand1])                                           # 在(rand1-1)以前(含)的同母親
        children1.extend(parent2[rand1:rand2])                                      # 在rand1~(rand2-1)的同母親
        children2.extend(parent1[rand1:rand2])                                      # 在rand1~(rand2-1)的同父親
        children1.extend(parent1[rand2:])                                           # 在(rand2+1)以後(含)的同父親
        children2.extend(parent2[rand2:])                                           # 在(rand2+1)以後(含)的同母親
    else:
        children1=parent1
        children2=parent2
    return children1, children2

## Crossover Uniform
def Crossover_Uniform(parent1,parent2,crossover_rate):  
    if np.random.random()<=crossover_rate:                                      # 隨機產生0~1的值若小於交配率,則交配
        # 產生的2個子代
        children1=[]
        children2=[]
        for crossover_gene in range(0,len(parent1)):                            # 每個基因有50%機率來自父親或母親
            if np.random.random()<=0.5:
                children1.append(parent1[crossover_gene])
                children2.append(parent2[crossover_gene])
            else:
                children1.append(parent2[crossover_gene])
                children2.append(parent1[crossover_gene])
    else:
        children1=parent1
        children2=parent2
    return children1, children2
                
## Mutation Bit-flip
def Mutation(gene):
    gene_length=len(gene)                                                       # 先取得基因段的長度
    mutation_rate=1/gene_length                                                 # 每個基因的突變機率
    # 依序處理每個基因
    for i in range(0,gene_length):                                              # 每個基因都有機率會突變
        if np.random.random()<=mutation_rate:                                   # 隨機產生0~1的值若小於突變率,則突變
            gene[i]=1-gene[i]                                                   # bit-flip
    return gene

## Survivor Selection
def Survivor_Selection(swarm,swarm_fitness,population):                         # 只會留下population的人口數
    new_swarm=np.zeros([population,np.size(swarm,1)])                           # 存放生存者的基因(新的swarm)
    for survived_people in range(0,population):
        best=np.argmin(swarm_fitness, axis=0)                                   # 目前適應值最小者的index
        new_swarm[survived_people,:]=swarm[best,:]                              # 存活的基因
        swarm=np.delete(swarm, best, 0)                                         # 把best從swarm中移除掉
        swarm_fitness=np.delete(swarm_fitness, best, 0)                         # 把best從swarm_fitness中移除掉
    return new_swarm,swarm_fitness
        

#############################   Main   ########################################

## Binary GAs-Representation
swarm=np.random.randint(0,2,(population,N*gene_length))                         # 初始化每個粒子的基因組合{0,1},每個粒子有N=10個維度,每個維度範圍0~1023->10個binary

## 先計算目前每個粒子的適應值
swarm_fitness=fitness(swarm,N,gene_length)

## 開始演化
for generation_times in range(0,generation):
    
    ## Tournament Selection
    Crossover_pool=Tournament_Selection(swarm,swarm_fitness,Tournament_n)       # 獲得要進入交配池的名單
    
    ## 建立offspring子代的人口
    swarm_offspring=np.zeros([population,N*gene_length])
    
    ## Crossover
    for parent in range(0,len(Crossover_pool),2):                               # 每對父母
        # 2-point crossover
        #swarm_offspring[parent,:],swarm_offspring[parent+1,:]=Crossover_two_point(swarm[Crossover_pool[parent],:],swarm[Crossover_pool[parent+1],:],crossover_rate)  # 1對父母產生1對子代
        # Uniform crossover
        swarm_offspring[parent,:],swarm_offspring[parent+1,:]=Crossover_Uniform(swarm[Crossover_pool[parent],:],swarm[Crossover_pool[parent+1],:],crossover_rate)  # 1對父母產生1對子代
        
    
    ## offspring進行Mutation
    for mutation_num in range(0,population):                                    # 每個子代都做突變
        swarm_offspring[mutation_num,:]=Mutation(swarm_offspring[mutation_num,:])
    
    ## 合併親代與子代    
    swarm=np.insert(swarm, np.size(swarm,0), values=swarm_offspring, axis=0)

    ## 計算目前每個粒子的適應值
    swarm_fitness=fitness(swarm,N,gene_length)                                  # 目前2代同堂
    
    ## Survivor Selection
    swarm,swarm_fitness=Survivor_Selection(swarm,swarm_fitness,population)
    
    ## 印出目前的平均fitness
    average_fitness=np.mean(swarm_fitness)                                      # 平均適應值
    record_average_fitness.append(average_fitness)
    '''
    print("It is ",generation_times," generations.")
    print("The average fitness is: ",np.mean(average_fitness))
    '''

## Result    
plt.figure(1)
plt.plot(record_average_fitness)
plt.xlabel('generations')
plt.ylabel('f(x)')
plt.show()              
            
            







