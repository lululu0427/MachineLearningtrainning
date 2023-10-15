import pandas as pd
import numpy as np
import tensorflow as tf
from keras import regularizers
from keras.regularizers import l1,l2
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

print("請將執行檔以及病患資料放置於桌面")
filename = input("請輸入檔案名稱：")
filepath = "C:\\Users\\user\\Desktop\\" + filename + ".csv"
dataset = pd.read_csv(filepath)

# 檢查每個欄位的資料型態是否符合預期
if dataset.dtypes[0] == 'object' and dataset.dtypes[1] == 'int64' and dataset.dtypes[2] == 'int64' and dataset.dtypes[3] == 'float64':
    print('資料格式正確')
else:
    print('資料格式不符合預期')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
# 使用SimpleImputer填補缺失值
imp = SimpleImputer(strategy='constant', fill_value=99999)
X = imp.fit_transform(X)
# 特徵篩選
ak=32
selector = SelectKBest(f_classif, k=ak)
X_new = selector.fit_transform(X, y)
X=X_new
selected_features = dataset.columns[3:-1][selector.get_support()]
# 資料集分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 對特徵進行特徵縮放
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_re, y_re = SMOTE(random_state=42).fit_resample(X_train, y_train)
X_train=X_re
y_train=y_re
# 使用TensorFlow建立模型
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32,)),
    #tf.keras.layers.Normalization(axis=-1),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(0.02)),
    tf.keras.layers.Dropout(0.2), 
    tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.02))
])
# Specify model training parameters

optimizer = tf.keras.optimizers.Adam(lr=1e-5) #選擇Adam optimizer來優化模型的權重。設置學習率（lr）為 1e-5，這是學習率的初始值。
loss = 'binary_crossentropy' 
metrics = ["accuracy"]

model.compile(optimizer, loss, metrics) 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# K-Fold Validation
k = 5  
kf = KFold(n_splits=k)
aepochs=128
abatch_size=16
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train, epochs=aepochs, batch_size=abatch_size, verbose=0)

# 評估訓練集模型的性能
Threshold =0.6
y_tpred = (model.predict(X_train) > Threshold).astype(int)
tcm = confusion_matrix(y_train, y_tpred)
tacc = accuracy_score(y_train, y_tpred)
tauc = roc_auc_score(y_train, y_tpred)
tp, fn, fp, tn = tcm.ravel()
trecall = tp / (tp + fn)
tf1_score = f1_score(y_train, y_tpred)

# 評估測試集模型的性能
y_pred = (model.predict(X_test) > Threshold).astype(int)
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)
p, n = np.bincount(y_test)
tp, fn, fp, tn = cm.ravel()
recall = tp / (tp + fn)
nf1_score = f1_score(y_test, y_pred)

moduledata = input("是否顯示模型運行之數據(y/n)")
modulesave = input("是否儲存模型運行之數據(y/n)")

if moduledata == 'y':
    #print("Selected features:", list(selected_features))
    print("\n以下為訓練集之數據:")
    print("訓練集之混淆矩陣:\n", tcm)
    print("訓練集模型準確率為：", tacc)
    print("訓練集AUC 值為：", tauc)
    print("訓練集模型recall值為:", trecall)
    print("訓練集模型F1-score為:", tf1_score)
    # 訓練集數據視覺化呈現
    title = 'Model Performance on Training Set'
    data = {'Accuracy': tacc, 'AUC': tauc, 'Recall': trecall, 'F1-score': tf1_score}
    # 設定圖表大小和字體大小
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 14})
    # 將數據轉換為list
    names = list(data.keys())
    values = list(data.values())
    # 繪製柱狀圖
    plt.bar(names, values)
    # 設定標題和標籤
    plt.title(title)
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    # 顯示圖表
    plt.show()
    print("\n以下為測試集之數據:")
    print("測試集之混淆矩陣:\n", cm)
    print("測試集模型準確率為：", acc)
    print("測試集AUC 值為：", auc)
    print("測試集模型recall值為:", recall)
    print("測試集模型F1-score為:", nf1_score)
    if modulesave == 'y':
        excel_file = 'C:\\Users\\user\\Desktop\\LNmodeldata.xlsx'
        with pd.ExcelFile(excel_file) as xls:
            df = xls.parse(xls.sheet_names[0])
            new_data = {
                'Model': ['Training Set', 'Test Set'],
                'Accuracy': [tacc, acc],
                'AUC': [tauc, auc],
                'Recall': [trecall, recall],
                'F1-score': [tf1_score, nf1_score],
                'Threshold': [Threshold,Threshold],
                'epochs':[aepochs,aepochs],
                'KBest':[ak,ak],
                'batch_size':[abatch_size,abatch_size],
                'Selected Features': [', '.join(selected_features)] * 2  # 將 selected_feature 轉為字串，並重複兩次
            }
            df = df.append(pd.DataFrame(new_data), ignore_index=True)
        with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name='Results', index=False)

    # 測試集數據視覺化呈現
    title = 'Model Performance on Test Set'
    data = {'Accuracy': acc, 'AUC': auc, 'Recall': recall, 'F1-score': nf1_score}
    # 設定圖表大小和字體大小
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 14})
    # 將數據轉換為list
    names = list(data.keys())
    values = list(data.values())
    # 繪製柱狀圖
    plt.bar(names, values)
    # 設定標題和標籤
    plt.title(title)
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    # 顯示圖表
    plt.show()
input("\n輸入enter以結束程式")