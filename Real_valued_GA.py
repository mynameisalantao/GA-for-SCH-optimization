# Real-valued GA

########################   Import Model  ######################################
import numpy as np                                                              # 計算矩陣使用
import math                                                                     # 計算根號與sin使用
import matplotlib.pyplot as plt 
                 
###########################   Parameter  ######################################
population=100                                                                  # 人口數
generation=100                                                                  # 迭代次數
N=10                                                                            # 維度
Tournament_n=2                                                                  # 每次2人競爭
crossover_rate=0.9                                                              # 交配率
record_average_fitness=[]                                                       # 紀錄每次迭代後的平均適應值
afa=0.25                                                                        # Arithmetic線性組合參數

#############################   函式   ########################################


## 計算cost function 
def SCH_cost_function(x):
    # 先計算後面sigma項
    cost=0
    for i in range(0,len(x)):
        cost+=x[i]*math.sin(math.sqrt(abs(x[i])))
    cost=418.98291*len(x)-cost
    return cost

## 計算所有粒子的適應值
def fitness(swarm):   
    swarm_fitness=np.zeros([np.size(swarm,0),1])                                # 紀錄每個粒子目前的適應值
    for swarm_number in range(0,np.size(swarm,0)):        
        swarm_fitness[swarm_number]=SCH_cost_function(swarm[swarm_number,:])
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

## Crossover Whole Arithmetic
def Crossover_Whole_Arithmetic(parent1,parent2,crossover_rate,afa): 
    if np.random.random()<=crossover_rate:                                      # 隨機產生0~1的值若小於交配率,則交配
        children1=afa*parent1+(1-afa)*parent2
        children2=afa*parent2+(1-afa)*parent1
    else:
        children1=parent1
        children2=parent2
    return children1, children2

                
## Mutation Uniform
def Mutation(gene):
    gene_length=len(gene)                                                       # 先取得基因段的長度
    mutation_rate=1/gene_length                                                 # 每個基因的突變機率
    # 依序處理每個基因
    for i in range(0,gene_length):                                              # 每個基因都有機率會突變
        if np.random.random()<=mutation_rate:                                   # 隨機產生0~1的值若小於突變率,則突變
            gene[i]=-512+np.random.random()*1024                                # 隨機產生在[-512,511]
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

## Real-valued GAs-Representation
swarm=np.random.uniform(-512,511,(population,N))                                # 初始化每個維度在[-512,511],每個粒子有N=10個維度

## 先計算目前每個粒子的適應值
swarm_fitness=fitness(swarm)

## 開始演化
for generation_times in range(0,generation):
    
    ## Tournament Selection
    Crossover_pool=Tournament_Selection(swarm,swarm_fitness,Tournament_n)       # 獲得要進入交配池的名單
    
    ## 建立offspring子代的人口
    swarm_offspring=np.zeros([population,N])
    
    ## Crossover
    for parent in range(0,len(Crossover_pool),2):                               # 每對父母
        
        # Uniform crossover
        swarm_offspring[parent,:],swarm_offspring[parent+1,:]=Crossover_Uniform(swarm[Crossover_pool[parent],:],swarm[Crossover_pool[parent+1],:],crossover_rate)  # 1對父母產生1對子代
        # Whole Arithmetic
        #swarm_offspring[parent,:],swarm_offspring[parent+1,:]=Crossover_Whole_Arithmetic(swarm[Crossover_pool[parent],:],swarm[Crossover_pool[parent+1],:],crossover_rate,afa)  # 1對父母產生1對子代

    
    ## offspring進行Mutation
    for mutation_num in range(0,population):                                    # 每個子代都做突變
        swarm_offspring[mutation_num,:]=Mutation(swarm_offspring[mutation_num,:])
    
    ## 合併親代與子代    
    swarm=np.insert(swarm, np.size(swarm,0), values=swarm_offspring, axis=0)

    ## 計算目前每個粒子的適應值
    swarm_fitness=fitness(swarm)                                                # 目前2代同堂
    
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
            
            







