import os
import time
import random
import math
from collections import defaultdict
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Bitstream Vera Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

class Quadrup:
    def __init__(self):
        self.uID = 0
        self.sID = 0
        self.tID = 0
        self.value = 0.0


class RTuple:
    def __init__(self):
        self.rowID = 0
        self.colID = 0
        self.mvalue = 0.0


class M5_BNLFT:
    def __init__(self, input_train_file, input_test_file, input_validation_file, separator):
        self.lambda_b = 0.5
        self.lambda_u = 1.0e-4
        self.lambda_s = self.lambda_u
        self.lambda_t = self.lambda_u
        self.trainingRound = 1000
        self.convergenceRound = 1000
        self.flagRMSE = False
        self.flagMAE = False
        self.minRMSE = 100.0
        self.minMAE = 100.0
        self.minRMSERound = 0
        self.minMAERound = 0
        self.delayCount = 10
        self.USlice = None
        self.SSlice = None
        self.TSlice = None
        self.randMax = 5.0e-4
        self.randMin = 4.0e-4
        self.init_lf = 5.0e-4
        self.biasMax = self.randMax
        self.biasMin = self.randMin
        self.inputTrainFile = input_train_file
        self.inputTestFile = input_test_file
        self.inputValidationFile = input_validation_file
        self.separator = separator
        self.trainData = []
        self.testData = []
        self.validationData = []
        self.uNum = 0
        self.sNum = 0
        self.tNum = 0
        self.trainDataNum = 0
        self.testDataNum = 0
        self.validationDataNum = 0
        self.U = None
        self.S = None
        self.T = None
        self.A = None
        self.B = None
        self.C = None
        self.everyRoundRMSE = None
        self.everyRoundMAE = None
        self.everyRoundValidationRMSE = None
        self.everyRoundValidationMAE = None

    def init_data(self, input_file, data, is_train, is_validation=False):
        with open(input_file, 'r') as f:
            for line in f:
                parts = line.strip().split(self.separator)
                if len(parts) < 4:
                    continue

                uID = int(parts[0])
                s = float(parts[1])
                sID = int(s)
                t = float(parts[2])
                tID = int(t)
                value = float(parts[3])

                self.uNum = max(self.uNum, uID)
                self.sNum = max(self.sNum, sID)
                self.tNum = max(self.tNum, tID)

                if is_train == 0 and not is_validation:
                    self.testDataNum += 1
                elif is_validation:
                    self.validationDataNum += 1
                else:
                    self.trainDataNum += 1

                qtemp = Quadrup()
                qtemp.uID = uID
                qtemp.sID = sID
                qtemp.tID = tID
                qtemp.value = value
                data.append(qtemp)

        if is_train == 0 and not is_validation:
            print(f"\n测试集用户的个数：{self.uNum}")
            print(f"测试集服务的个数：{self.sNum}")
            print(f"测试集时间的个数：{self.tNum}")
            print(f"测试集总样本数：{self.testDataNum}")
        elif is_validation:
            print(f"\n验证集用户的个数：{self.uNum}")
            print(f"验证集服务的个数：{self.sNum}")
            print(f"验证集时间的个数：{self.tNum}")
            print(f"验证集总样本数：{self.validationDataNum}")
        else:
            print(f"训练集用户的个数：{self.uNum}")
            print(f"训练集服务的个数：{self.sNum}")
            print(f"训练集时间的个数：{self.tNum}")
            print(f"训练集总样本数：{self.trainDataNum}")

    def part_slice(self):
        self.USlice = defaultdict(list)
        self.SSlice = defaultdict(list)
        self.TSlice = defaultdict(list)

        for slice1 in self.trainData:
            rtemp = RTuple()
            rtemp.rowID = slice1.sID
            rtemp.colID = slice1.tID
            rtemp.mvalue = slice1.value
            self.USlice[slice1.uID].append(rtemp)

            rtemp = RTuple()
            rtemp.rowID = slice1.uID
            rtemp.colID = slice1.tID
            rtemp.mvalue = slice1.value
            self.SSlice[slice1.sID].append(rtemp)

            rtemp = RTuple()
            rtemp.rowID = slice1.uID
            rtemp.colID = slice1.sID
            rtemp.mvalue = slice1.value
            self.TSlice[slice1.tID].append(rtemp)

    def init_UST(self, rank):
        self.U = [[0.0] * (rank + 1) for _ in range(self.uNum + 1)]
        self.S = [[0.0] * (rank + 1) for _ in range(self.sNum + 1)]
        self.T = [[0.0] * (rank + 1) for _ in range(self.tNum + 1)]

        for i in range(1, self.uNum + 1):
            for j in range(1, rank + 1):
                self.U[i][j] = self.randMin + random.random() * (self.randMax - self.randMin)

        for i in range(1, self.sNum + 1):
            for j in range(1, rank + 1):
                self.S[i][j] = self.randMin + random.random() * (self.randMax - self.randMin)

        for i in range(1, self.tNum + 1):
            for j in range(1, rank + 1):
                self.T[i][j] = self.randMin + random.random() * (self.randMax - self.randMin)

    def init_bias(self):
        self.A = [0.0] * (self.uNum + 1)
        self.B = [0.0] * (self.sNum + 1)
        self.C = [0.0] * (self.tNum + 1)

        for i in range(1, self.uNum + 1):
            self.A[i] = self.biasMin + random.random() * (self.biasMax - self.biasMin)

        for j in range(1, self.sNum + 1):
            self.B[j] = self.biasMin + random.random() * (self.biasMax - self.biasMin)

        for k in range(1, self.tNum + 1):
            self.C[k] = self.biasMin + random.random() * (self.biasMax - self.biasMin)

    def calculate_metrics(self, data, rank):
        RMSE_up = 0.0
        MAE_up = 0.0

        for item in data:
            y_temp = 0.0
            for r in range(1, rank + 1):
                y_temp += self.U[item.uID][r] * self.S[item.sID][r] * self.T[item.tID][r]

            y_temp += self.A[item.uID] + self.B[item.sID] + self.C[item.tID]
            RMSE_up += (item.value - y_temp) ** 2
            MAE_up += abs(item.value - y_temp)

        rmse = math.sqrt(RMSE_up / len(data))
        mae = MAE_up / len(data)
        return rmse, mae

    def train(self, rank):
        starttime = time.time()
        self.everyRoundRMSE = [0.0] * (self.trainingRound + 1)
        self.everyRoundMAE = [0.0] * (self.trainingRound + 1)
        self.everyRoundValidationRMSE = [0.0] * (self.trainingRound + 1)
        self.everyRoundValidationMAE = [0.0] * (self.trainingRound + 1)
        self.minRMSE = 100.0
        self.minMAE = 100.0
        self.minValidationRMSE = 100.0
        self.minRMSERound = 0
        self.minMAERound = 0
        self.minValidationRMSERound = 0

        print("\ncalculating RMSE and MAE when initializing the LF...")
        train_rmse, train_mae = self.calculate_metrics(self.trainData, rank)
        test_rmse, test_mae = self.calculate_metrics(self.testData, rank)
        validation_rmse, validation_mae = self.calculate_metrics(self.validationData, rank)
        
        print(f"初始训练集RMSE：{train_rmse}, MAE：{train_mae}")
        print(f"初始测试集RMSE：{test_rmse}, MAE：{test_mae}")
        print(f"初始验证集RMSE：{validation_rmse}, MAE：{validation_mae}")

        for tr in range(1, self.trainingRound + 1):
            starttime1 = time.time()

            # Update U
            for k in range(1, rank + 1):
                for i in range(1, self.uNum + 1):
                    MAEUp = 0.0
                    endtime1 = 0.0

                    if i in self.USlice:
                        for ttemp in self.USlice[i]:
                            tDownTemp = 0.0
                            for r in range(1, rank + 1):
                                tDownTemp += self.U[i][r] * self.S[ttemp.rowID][r] * self.T[ttemp.colID][r]

                            tDownTemp += self.A[i] + self.B[ttemp.rowID] + self.C[ttemp.colID]
                            MAEUp += ttemp.mvalue * self.S[ttemp.rowID][k] * self.T[ttemp.colID][k]
                            endtime1 += tDownTemp * self.S[ttemp.rowID][k] * self.T[ttemp.colID][k]

                        endtime1 += self.lambda_u * self.U[i][k]

                    MAEUp = self.U[i][k] * MAEUp
                    if endtime1 == 0.0:
                        endtime1 = sys.float_info.min

                    self.U[i][k] = MAEUp / endtime1

            # Update S
            for k in range(1, rank + 1):
                for j in range(1, self.sNum + 1):
                    MAEUp = 0.0
                    endtime1 = 0.0

                    if j in self.SSlice:
                        for ttemp in self.SSlice[j]:
                            tDownTemp = 0.0
                            for r in range(1, rank + 1):
                                tDownTemp += self.S[j][r] * self.U[ttemp.rowID][r] * self.T[ttemp.colID][r]

                            tDownTemp += self.A[ttemp.rowID] + self.B[j] + self.C[ttemp.colID]
                            MAEUp += ttemp.mvalue * self.U[ttemp.rowID][k] * self.T[ttemp.colID][k]
                            endtime1 += tDownTemp * self.U[ttemp.rowID][k] * self.T[ttemp.colID][k]

                        endtime1 += self.lambda_s * self.S[j][k]

                    MAEUp = self.S[j][k] * MAEUp
                    if endtime1 == 0.0:
                        endtime1 = sys.float_info.min

                    self.S[j][k] = MAEUp / endtime1

            # Update T
            for k in range(1, rank + 1):
                for t in range(1, self.tNum + 1):
                    MAEUp = 0.0
                    endtime1 = 0.0

                    if t in self.TSlice:
                        for ttemp in self.TSlice[t]:
                            tDownTemp = 0.0
                            for r in range(1, rank + 1):
                                tDownTemp += self.T[t][r] * self.U[ttemp.rowID][r] * self.S[ttemp.colID][r]

                            tDownTemp += self.A[ttemp.rowID] + self.B[ttemp.colID] + self.C[t]
                            MAEUp += ttemp.mvalue * self.U[ttemp.rowID][k] * self.S[ttemp.colID][k]
                            endtime1 += tDownTemp * self.U[ttemp.rowID][k] * self.S[ttemp.colID][k]

                        endtime1 += self.lambda_t * self.T[t][k]

                    MAEUp = self.T[t][k] * MAEUp
                    if endtime1 == 0.0:
                        endtime1 = sys.float_info.min

                    self.T[t][k] = MAEUp / endtime1

            # Update A (user bias)
            for i in range(1, self.uNum + 1):
                cUp = 0.0
                cDown = 0.0

                if i in self.USlice:
                    for ttemp in self.USlice[i]:
                        temp = 0.0
                        for r in range(1, rank + 1):
                            temp += self.U[i][r] * self.S[ttemp.rowID][r] * self.T[ttemp.colID][r]

                        temp += self.A[i] + self.B[ttemp.rowID] + self.C[ttemp.colID]
                        cUp += ttemp.mvalue
                        cDown += temp

                    cDown += self.lambda_b * self.A[i]
                    if cDown == 0.0:
                        cDown = sys.float_info.min

                    cUp = self.A[i] * cUp
                    self.A[i] = cUp / cDown

            # Update B (service bias)
            for j in range(1, self.sNum + 1):
                cUp = 0.0
                cDown = 0.0

                if j in self.SSlice:
                    for ttemp in self.SSlice[j]:
                        temp = 0.0
                        for r in range(1, rank + 1):
                            temp += self.U[ttemp.rowID][r] * self.S[j][r] * self.T[ttemp.colID][r]

                        temp += self.A[ttemp.rowID] + self.B[j] + self.C[ttemp.colID]
                        cUp += ttemp.mvalue
                        cDown += temp

                    cDown += self.lambda_b * self.B[j]
                    if cDown == 0.0:
                        cDown = sys.float_info.min

                    cUp = self.B[j] * cUp
                    self.B[j] = cUp / cDown

            # Update C (time bias)
            for t in range(1, self.tNum + 1):
                cUp = 0.0
                cDown = 0.0

                if t in self.TSlice:
                    for ttemp in self.TSlice[t]:
                        temp = 0.0
                        for r in range(1, rank + 1):
                            temp += self.U[ttemp.rowID][r] * self.S[ttemp.colID][r] * self.T[t][r]

                        temp += self.A[ttemp.rowID] + self.B[ttemp.colID] + self.C[t]
                        cUp += ttemp.mvalue
                        cDown += temp

                    cDown += self.lambda_b * self.C[t]
                    if cDown == 0.0:
                        cDown = sys.float_info.min

                    cUp = self.C[t] * cUp
                    self.C[t] = cUp / cDown

            print(f"\n第{tr}轮训练完成，开始计算评估指标")
            
            # 计算训练集指标
            train_rmse, train_mae = self.calculate_metrics(self.trainData, rank)
            
            # 计算测试集指标
            test_rmse, test_mae = self.calculate_metrics(self.testData, rank)
            
            # 计算验证集指标
            validation_rmse, validation_mae = self.calculate_metrics(self.validationData, rank)
            
            self.everyRoundRMSE[tr] = test_rmse
            self.everyRoundMAE[tr] = test_mae
            self.everyRoundValidationRMSE[tr] = validation_rmse
            self.everyRoundValidationMAE[tr] = validation_mae
            
            print(f"当前轮数：{tr}")
            print(f"  训练集 - RMSE：{train_rmse:.6f}, MAE：{train_mae:.6f}")
            print(f"  测试集 - RMSE：{test_rmse:.6f}, MAE：{test_mae:.6f}")
            print(f"  验证集 - RMSE：{validation_rmse:.6f}, MAE：{validation_mae:.6f}")


            if abs(validation_rmse - self.minValidationRMSE) >= 1.0e-4 and validation_rmse < self.minValidationRMSE:
                self.minValidationRMSE = validation_rmse
                self.minValidationRMSERound = tr
                best_U = [row[:] for row in self.U]
                best_S = [row[:] for row in self.S]
                best_T = [row[:] for row in self.T]
                best_A = self.A[:]
                best_B = self.B[:]
                best_C = self.C[:]
            elif tr - self.minValidationRMSERound >= self.delayCount:
                print(f"基于验证集的早停机制触发，在第 {tr} 轮停止训练")
                self.U = best_U
                self.S = best_S
                self.T = best_T
                self.A = best_A
                self.B = best_B
                self.C = best_C
                self.convergenceRound = tr
                break

            if abs(test_rmse - self.minRMSE) >= 1.0e-4 and test_rmse < self.minRMSE:
                self.minRMSE = test_rmse
                self.minRMSERound = tr
            elif tr - self.minRMSERound >= self.delayCount:
                self.flagRMSE = True
                if self.flagMAE:
                    self.convergenceRound = tr
                    break

            if abs(test_mae - self.minMAE) >= 1.0e-4 and test_mae < self.minMAE:
                self.minMAE = test_mae
                self.minMAERound = tr
            elif tr - self.minMAERound >= self.delayCount:
                self.flagMAE = True
                if self.flagRMSE:
                    self.convergenceRound = tr
                    break

            print(f"最小测试集RMSE：{self.minRMSE:.6f}, 轮数：{self.minRMSERound}")
            print(f"最小测试集MAE：{self.minMAE:.6f}, 轮数：{self.minMAERound}")
            print(f"最小验证集RMSE：{self.minValidationRMSE:.6f}, 轮数：{self.minValidationRMSERound}")
            endtime1 = time.time()
            print(f"此轮的训练时间为：{endtime1 - starttime1:.2f}秒")

        print(f"最终收敛的轮数为：{self.convergenceRound}")
        endtime = time.time()
        print(f"训练时间：{(endtime - starttime) / 60.0:.2f}分钟\n")
        print(f"最终收敛的轮数为：{self.convergenceRound}")
        print(f"最小的测试集RMSE为：{self.minRMSE:.6f}, 轮数为：{self.minRMSERound}")
        print(f"最小的测试集MAE为：{self.minMAE:.6f}, 轮数为：{self.minMAERound}")
        print(f"最小的验证集RMSE为：{self.minValidationRMSE:.6f}, 轮数为：{self.minValidationRMSERound}")