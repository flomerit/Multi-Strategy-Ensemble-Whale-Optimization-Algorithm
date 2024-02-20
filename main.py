# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 21:43:03 2020

@author: ZongSing_NB
"""

import time
import functools

import numpy as np
import pandas as pd

from MSWOA import MSWOA
import benchmark
import bound_X
import ideal_F
import dimension

D = 30    # 维数
G = 500    # 最大迭代数
P = 30    # 种群数
run_times = 50  # 运行次数
table = pd.DataFrame(np.zeros([6, 36]), index=['avg', 'std', 'worst', 'best', 'ideal', 'time'])
loss_curves = np.zeros([G, 36])  # 36是测试函数个数
F_table = np.zeros([run_times, 36])
for t in range(run_times):
    item = 0
    ub = bound_X.Sphere()[1]*np.ones(dimension.Sphere(D))    #  上界（np数组进行维数匹配）
    lb = bound_X.Sphere()[0]*np.ones(dimension.Sphere(D))
    optimizer = MSWOA(fitness=benchmark.Sphere,
                      D=dimension.Sphere(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()  # 开始计时
    optimizer.opt()   # 运行MSWOA
    ed = time.time()  # 结束计时
    F_table[t, item] = optimizer.gbest_F     #  F_table 存储每个测试函数每次迭代的最佳值--行是迭代次数，列为测试函数代号
    table[item]['avg'] += optimizer.gbest_F  # table的每一行为一个测试函数的数据，存储最佳适应度
    table[item]['time'] += ed - st    # table['time'][item] += ed - st
    table[item]['ideal'] = ideal_F.Sphere()  # 理想值
    loss_curves[:, item] += optimizer.loss_curve  # 将优化过程中的损失曲线添加到 loss_curves 中，其中 loss_curve 是 MSWOA 类中记录的每次迭代的目标函数值的数组

    
    item = item + 1
    ub = bound_X.Rastrigin()[1]*np.ones(dimension.Rastrigin(D))
    lb = bound_X.Rastrigin()[0]*np.ones(dimension.Rastrigin(D))
    optimizer = MSWOA(fitness=benchmark.Rastrigin,
                      D=dimension.Rastrigin(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Rastrigin()
    loss_curves[:, item] += optimizer.loss_curve


    item = item + 1
    ub = bound_X.Ackley()[1]*np.ones(dimension.Ackley(D))
    lb = bound_X.Ackley()[0]*np.ones(dimension.Ackley(D))
    optimizer = MSWOA(fitness=benchmark.Ackley,
                      D=dimension.Ackley(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Ackley()
    loss_curves[:, item] += optimizer.loss_curve


    item = item + 1
    ub = bound_X.Griewank()[1]*np.ones(dimension.Griewank(D))
    lb = bound_X.Griewank()[0]*np.ones(dimension.Griewank(D))
    optimizer = MSWOA(fitness=benchmark.Griewank,
                      D=dimension.Griewank(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Griewank()
    loss_curves[:, item] += optimizer.loss_curve
    

    item = item + 1
    ub = bound_X.Schwefel_P222()[1]*np.pi*np.ones(dimension.Schwefel_P222(D))
    lb = bound_X.Schwefel_P222()[0]*np.pi*np.ones(dimension.Schwefel_P222(D))
    optimizer = MSWOA(fitness=benchmark.Schwefel_P222,
                      D=dimension.Schwefel_P222(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Schwefel_P222()
    loss_curves[:, item] += optimizer.loss_curve
    

    item = item + 1
    ub = bound_X.Rosenbrock()[1]*np.ones(dimension.Rosenbrock(D))
    lb = bound_X.Rosenbrock()[0]*np.ones(dimension.Rosenbrock(D))
    optimizer = MSWOA(fitness=benchmark.Rosenbrock,
                      D=dimension.Rosenbrock(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Rosenbrock()
    loss_curves[:, item] += optimizer.loss_curve


    item = item + 1
    ub = bound_X.Sehwwefel_P221()[1]*np.ones(dimension.Sehwwefel_P221(D))
    lb = bound_X.Sehwwefel_P221()[0]*np.ones(dimension.Sehwwefel_P221(D))
    optimizer = MSWOA(fitness=benchmark.Sehwwefel_P221,
                      D=dimension.Sehwwefel_P221(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Sehwwefel_P221()
    loss_curves[:, item] += optimizer.loss_curve


    item = item + 1
    ub = bound_X.Quartic()[1]*np.ones(dimension.Quartic(D))
    lb = bound_X.Quartic()[0]*np.ones(dimension.Quartic(D))
    optimizer = MSWOA(fitness=benchmark.Quartic,
                      D=dimension.Quartic(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Quartic()
    loss_curves[:, item] += optimizer.loss_curve


    item = item + 1
    ub = bound_X.Schwefel_P12()[1]*np.ones(dimension.Schwefel_P12(D))
    lb = bound_X.Schwefel_P12()[0]*np.ones(dimension.Schwefel_P12(D))
    optimizer = MSWOA(fitness=benchmark.Schwefel_P12,
                      D=dimension.Schwefel_P12(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Schwefel_P12()
    loss_curves[:, item] += optimizer.loss_curve


    item = item + 1
    ub = bound_X.Penalized1()[1]*np.ones(dimension.Penalized1(D))
    lb = bound_X.Penalized1()[0]*np.ones(dimension.Penalized1(D))
    optimizer = MSWOA(fitness=benchmark.Penalized1,
                      D=dimension.Penalized1(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Penalized1()
    loss_curves[:, item] += optimizer.loss_curve
    

    item = item + 1
    ub = bound_X.Penalized2()[1]*np.ones(dimension.Penalized2(D))
    lb = bound_X.Penalized2()[0]*np.ones(dimension.Penalized2(D))
    optimizer = MSWOA(fitness=benchmark.Penalized2,
                      D=dimension.Penalized2(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Penalized2()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.Schwefel_226()[1]*np.ones(dimension.Schwefel_226(D))
    lb = bound_X.Schwefel_226()[0]*np.ones(dimension.Schwefel_226(D))
    optimizer = MSWOA(fitness=benchmark.Schwefel_226,
                      D=dimension.Schwefel_226(dimension.Schwefel_226(D)), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Schwefel_226(D)
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.Step()[1]*np.ones(dimension.Step(D))
    lb = bound_X.Step()[0]*np.ones(dimension.Step(D))
    optimizer = MSWOA(fitness=benchmark.Step,
                      D=dimension.Step(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Step()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.Kowalik()[1]*np.ones(dimension.Kowalik())
    lb = bound_X.Kowalik()[0]*np.ones(dimension.Kowalik())
    optimizer = MSWOA(fitness=benchmark.Kowalik,
                      D=dimension.Kowalik(), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Kowalik()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.ShekelFoxholes()[1]*np.ones(dimension.ShekelFoxholes())
    lb = bound_X.ShekelFoxholes()[0]*np.ones(dimension.ShekelFoxholes())
    optimizer = MSWOA(fitness=benchmark.ShekelFoxholes,
                      D=dimension.ShekelFoxholes(), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.ShekelFoxholes()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.GoldsteinPrice()[1]*np.ones(dimension.GoldsteinPrice())
    lb = bound_X.GoldsteinPrice()[0]*np.ones(dimension.GoldsteinPrice())
    optimizer = MSWOA(fitness=benchmark.GoldsteinPrice,
                      D=dimension.GoldsteinPrice(), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.GoldsteinPrice()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    Shekel5 = functools.partial(benchmark.Shekel, m=5)
    ub = bound_X.Shekel()[1]*np.ones(dimension.Shekel())
    lb = bound_X.Shekel()[0]*np.ones(dimension.Shekel())
    optimizer = MSWOA(fitness=Shekel5,
                      D=dimension.Shekel(), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Shekel()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.Branin()[2:]*np.ones(dimension.Branin())
    lb = bound_X.Branin()[:2]*np.ones(dimension.Branin())
    optimizer = MSWOA(fitness=benchmark.Branin,
                      D=dimension.Branin(), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Branin()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.Hartmann3()[1]*np.ones(dimension.Hartmann3())
    lb = bound_X.Hartmann3()[0]*np.ones(dimension.Hartmann3())
    optimizer = MSWOA(fitness=benchmark.Hartmann3,
                      D=dimension.Hartmann3(), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Hartmann3()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    Shekel7 = functools.partial(benchmark.Shekel, m=7)
    ub = bound_X.Shekel()[1]*np.ones(dimension.Shekel())
    lb = bound_X.Shekel()[0]*np.ones(dimension.Shekel())
    optimizer = MSWOA(fitness=Shekel7,
                      D=dimension.Shekel(), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Shekel()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    Shekel10 = functools.partial(benchmark.Shekel, m=10)
    ub = bound_X.Shekel()[1]*np.ones(dimension.Shekel())
    lb = bound_X.Shekel()[0]*np.ones(dimension.Shekel())
    optimizer = MSWOA(fitness=benchmark.Shekel,
                      D=dimension.Shekel(), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Shekel()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.SixHumpCamelBack()[1]*np.ones(dimension.SixHumpCamelBack())
    lb = bound_X.SixHumpCamelBack()[0]*np.ones(dimension.SixHumpCamelBack())
    optimizer = MSWOA(fitness=benchmark.SixHumpCamelBack,
                      D=dimension.SixHumpCamelBack(), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.SixHumpCamelBack()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.Hartmann6()[1]*np.ones(dimension.Hartmann6())
    lb = bound_X.Hartmann6()[0]*np.ones(dimension.Hartmann6())
    optimizer = MSWOA(fitness=benchmark.Hartmann6,
                      D=dimension.Hartmann6(), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Hartmann6()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.Zakharov()[1]*np.ones(dimension.Zakharov(D))
    lb = bound_X.Zakharov()[0]*np.ones(dimension.Zakharov(D))
    optimizer = MSWOA(fitness=benchmark.Zakharov,
                      D=dimension.Zakharov(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Zakharov()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.SumSquares()[1]*np.ones(dimension.SumSquares(D))
    lb = bound_X.SumSquares()[0]*np.ones(dimension.SumSquares(D))
    optimizer = MSWOA(fitness=benchmark.SumSquares,
                      D=dimension.SumSquares(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.SumSquares()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.Alpine()[1]*np.ones(dimension.Alpine(D))
    lb = bound_X.Alpine()[0]*np.ones(dimension.Alpine(D))
    optimizer = MSWOA(fitness=benchmark.Alpine,
                      D=dimension.Alpine(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Alpine()
    loss_curves[:, item] += optimizer.loss_curve
    

    item = item + 1
    ub = bound_X.Michalewicz()[1]*np.ones(dimension.Michalewicz())
    lb = bound_X.Michalewicz()[0]*np.ones(dimension.Michalewicz())
    optimizer = MSWOA(fitness=benchmark.Michalewicz,
                      D=dimension.Michalewicz(), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Michalewicz(dimension.Michalewicz())
    loss_curves[:, item] += optimizer.loss_curve
    

    item = item + 1
    ub = bound_X.Exponential()[1]*np.ones(dimension.Exponential(D))
    lb = bound_X.Exponential()[0]*np.ones(dimension.Exponential(D))
    optimizer = MSWOA(fitness=benchmark.Exponential,
                      D=dimension.Exponential(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Exponential()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.Schaffer()[1]*np.ones(dimension.Schaffer())
    lb = bound_X.Schaffer()[0]*np.ones(dimension.Schaffer())
    optimizer = MSWOA(fitness=benchmark.Schaffer,
                      D=dimension.Schaffer(), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Schaffer()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.BentCigar()[1]*np.ones(dimension.BentCigar(D))
    lb = bound_X.BentCigar()[0]*np.ones(dimension.BentCigar(D))
    optimizer = MSWOA(fitness=benchmark.BentCigar,
                      D=dimension.BentCigar(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.BentCigar()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.Bohachevsky1()[1]*np.ones(dimension.Bohachevsky1())
    lb = bound_X.Bohachevsky1()[0]*np.ones(dimension.Bohachevsky1())
    optimizer = MSWOA(fitness=benchmark.Bohachevsky1,
                      D=dimension.Bohachevsky1(), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Bohachevsky1()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.Elliptic()[1]*np.ones(dimension.Elliptic(D))
    lb = bound_X.Elliptic()[0]*np.ones(dimension.Elliptic(D))
    optimizer = MSWOA(fitness=benchmark.Elliptic,
                      D=dimension.Elliptic(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Elliptic()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.DropWave()[1]*np.ones(dimension.DropWave())
    lb = bound_X.DropWave()[0]*np.ones(dimension.DropWave())
    optimizer = MSWOA(fitness=benchmark.DropWave,
                      D=dimension.DropWave(), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.DropWave()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.CosineMixture()[1]*np.ones(dimension.CosineMixture(D))
    lb = bound_X.CosineMixture()[0]*np.ones(dimension.CosineMixture(D))
    optimizer = MSWOA(fitness=benchmark.CosineMixture,
                      D=dimension.CosineMixture(dimension.CosineMixture(D)), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.CosineMixture(dimension.CosineMixture(D))
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.Ellipsoidal(dimension.Ellipsoidal(D))[1]*np.ones(dimension.Ellipsoidal(D))
    lb = bound_X.Ellipsoidal(dimension.Ellipsoidal(D))[0]*np.ones(dimension.Ellipsoidal(D))
    optimizer = MSWOA(fitness=benchmark.Ellipsoidal,
                      D=dimension.Ellipsoidal(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Ellipsoidal()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.LevyandMontalvo1()[1]*np.ones(dimension.LevyandMontalvo1(D))
    lb = bound_X.LevyandMontalvo1()[0]*np.ones(dimension.LevyandMontalvo1(D))
    optimizer = MSWOA(fitness=benchmark.LevyandMontalvo1,
                      D=dimension.LevyandMontalvo1(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.LevyandMontalvo1()
    loss_curves[:, item] += optimizer.loss_curve

    
    print(t+1)

loss_curves = loss_curves / run_times
loss_curves = pd.DataFrame(loss_curves)    # 方便后续对数据的处理、分析和可视化。
loss_curves.columns = ['Sphere', 'Rastrigin', 'Ackley', 'Griewank', 'Schwefel P2.22',
                       'Rosenbrock', 'Sehwwefel P2.21', 'Quartic', 'Schwefel P1.2', 'Penalized 1',
                       'Penalized 2', 'Schwefel P2.26', 'Step', 'Kowalik', 'Shekel Foxholes',
                       'Goldstein-Price', 'Shekel 5', 'Branin', 'Hartmann 3', 'Shekel 7',
                       'Shekel 10', 'Six-Hump Camel-Back', 'Hartmann 6', 'Zakharov', 'Sum Squares',
                       'Alpine', 'Michalewicz', 'Exponential', 'Schaffer', 'Bent Cigar',
                       'Bohachevsky 1', 'Elliptic', 'Drop Wave', 'Cosine Mixture', 'Ellipsoidal',
                       'Levy and Montalvo 1']     # 列名设置为一个包含了多个字符串的列表
loss_curves.to_csv('loss_curves(MSWOA).csv')    # 将 DataFrame table 中的数据保存为一个名为 'table(MSWOA).csv' 的 CSV 文件。CSV 文件是一种常见的文本文件格式，用于存储表格数据

table.loc[['avg', 'time']] = table.loc[['avg', 'time']] / run_times    # table DataFrame 中索引为 'avg' 和 'time' 的行的值除以变量 run_times，然后重新赋值给相同的行
table.loc['worst'] = F_table.max(axis=0)  # 将 F_table 中每列的最大值（即迭代中每个测试函数表现最不好的）添加到 table DataFrame 中的一个新行，该行的索引为 'worst'
table.loc['best'] = F_table.min(axis=0)   
table.loc['std'] = F_table.std(axis=0)
table.columns = ['Sphere', 'Rastrigin', 'Ackley', 'Griewank', 'Schwefel P2.22',
                 'Rosenbrock', 'Sehwwefel P2.21', 'Quartic', 'Schwefel P1.2', 'Penalized 1',
                 'Penalized 2', 'Schwefel P2.26', 'Step', 'Kowalik', 'Shekel Foxholes',
                 'Goldstein-Price', 'Shekel 5', 'Branin', 'Hartmann 3', 'Shekel 7',
                 'Shekel 10', 'Six-Hump Camel-Back', 'Hartmann 6', 'Zakharov', 'Sum Squares',
                 'Alpine', 'Michalewicz', 'Exponential', 'Schaffer', 'Bent Cigar',
                 'Bohachevsky 1', 'Elliptic', 'Drop Wave', 'Cosine Mixture', 'Ellipsoidal',
                 'Levy and Montalvo 1']
table.to_csv('table(MSWOA).csv')
