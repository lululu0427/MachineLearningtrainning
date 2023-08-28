#%matplotlib inline
import matplotlib.pyplot as plt

#Load libraries for data processing
import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from scipy.stats import norm

## Supervised learning.
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn import metrics, preprocessing
from sklearn.metrics import classification_report

# visualization
import seaborn as sns #Seaborn 建立在 Matplotlib 之上，提供了更高級的功能和更美觀的預設樣式。
plt.style.use('fivethirtyeight') #設定了 Matplotlib 圖表的風格，使用了 'fivethirtyeight' 風格
sns.set_style("white") #設定了 Seaborn 庫的風格，使圖表使用白色背景

plt.rcParams['figure.figsize'] = (8,4) #設定了 Matplotlib 的全局參數，將圖表的尺寸設定為寬 8 單位，高 4 單位。
#plt.rcParams['axes.titlesize'] = 'large'
data = pd.read_csv('C:\\Users\\user\\Desktop\\modledata\\SVM2data.csv', index_col=False) #讀取檔案
data.drop('Unnamed: 0',axis=1, inplace=True)
#'Unnamed: 0'：這是要刪除的列的名稱。在 Pandas 中，CSV 文件的列通常會自動分配列名，如果 CSV 文件中的第一列不包含列名，則 Pandas 會將其命名為 'Unnamed: 0' 或類似的名稱。
#'inplace=True'：當它設置為 True 時，表示要修改原始的 DataFrame，而不是返回一個新的 DataFrame。
#axis=1：這是 drop 函数的一個參數，它指示要刪除的是列（列是沿著軸 1 方向的），而不是行（軸 0 方向）。在這裡，我們告訴 Pandas 刪除指定的列。
#data.head()
#Assign predictors to a variable of ndarray (matrix) type
array = data.values # 使data 轉換為 NumPy'array'，分別存入X(特徵數據)、y(目標變數)
X = array[:,1:31] # features
y = array[:,0]

#transform the class labels from their original string representation (M and B) into integers
#LabelEncoder 用於將類別標籤（在這個情境中是 'M' 和 'B'）轉換為整數表示。
le = LabelEncoder()
y = le.fit_transform(y) #用於目標變數y

# Normalize the  data (center around 0 and scale to remove the variance).
#標準化數據。標準化的目標是將數據的分佈調整為均值為 0，標準差為 1。
scaler =StandardScaler()
Xs = scaler.fit_transform(X)
# 5. Divide records in training and testing sets.
#將數據分為訓練集（70%）和測試集（30%），stratify=y 確保分割後的類別比例與原始數據一致。
X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.3, random_state=2, stratify=y)

# 6. Create an SVM classifier and train it on 70% of the data set.
#創建了一個支持向量機（SVM），probability=True 參數允許分類器計算每個類別的概率。
clf = SVC(probability=True)
clf.fit(X_train, y_train)

#7. Analyze accuracy of predictions on 30% of the holdout test sample.
#classifier_score = clf.score(X_test, y_test)：這一行使用已經訓練好的 SVM 分類器 clf 來預測測試集 X_test 的類別，然後使用 score 方法計算分類器的準確度。
classifier_score = clf.score(X_test, y_test)
print ('\nThe classifier accuracy score is {:03.2f}\n'.format(classifier_score))
# Get average of 3-fold cross-validation score using an SVC estimator.
#設置交叉驗證的折數為3次，等同於將數據分成幾個子集進行驗證。
n_folds = 3
#使用交叉驗證評估模型的性能
#使用 cross_val_score 函數，並傳給新的 SVM 分類器 SVC()，並計算在 n_folds 折交叉驗證下的平均性能，結果將被存在 cv_error
cv_error = np.average(cross_val_score(SVC(), Xs, y, cv=n_folds))
print ('\nThe {}-fold cross-validation accuracy score for this classifier is {:.2f}\n'.format(n_folds, cv_error))
from sklearn.feature_selection import SelectKBest, f_regression #SelectKBest 用於特徵選擇，f_regression 用於回歸分析
#使用 SelectKBest 來選擇特徵，並再次分類，以提高性能
clf2 = make_pipeline(SelectKBest(f_regression, k=3),SVC(probability=True))

scores = cross_val_score(clf2, Xs, y, cv=3)

# Get average of 3-fold cross-validation score using an SVC estimator.
#n_folds 為交叉驗證的折數
n_folds = 3
#評估經特徵篩選之SVM性能
cv_error = np.average(cross_val_score(SVC(), Xs, y, cv=n_folds))
print ('\nThe {}-fold cross-validation accuracy score for this classifier is {:.2f}\n'.format(n_folds, cv_error))
print (scores)
avg = (100*np.mean(scores), 100*np.std(scores)/np.sqrt(scores.shape[0]))
print ("Average score and uncertainty: (%.2f +- %.3f)%%"%avg)
# The confusion matrix helps visualize the performance of the algorithm.
#將新的SVM套用在測試集
y_pred = clf.fit(X_train, y_train).predict(X_test)
cm = metrics.confusion_matrix(y_test, y_pred)
#print(cm)
#%matplotlib inline
import matplotlib.pyplot as plt

#from IPython.display import Image, display

fig, ax = plt.subplots(figsize=(5, 5)) #創建一個 5x5 的畫布和軸，畫布存在fig、軸存在ax
ax.matshow(cm, cmap=plt.cm.Reds, alpha=0.3) #matshow 將混淆矩陣以顏色圖的形式顯示在軸(ax)上，顏色地(cmap 設置為紅色，透明度(alpha)為 0.3
#在混淆矩陣的可視化圖表上標記每個細胞（cell）的數值，以便更清楚地理解混淆矩陣的內容
for i in range(cm.shape[0]):
     for j in range(cm.shape[1]):
         ax.text(x=j, y=i,
                s=cm[i, j], 
                va='center', ha='center')
#設置 x 軸和 y 軸的標籤
plt.xlabel('Predicted Values', )
plt.ylabel('Actual Values')
plt.show()
print(classification_report(y_test, y_pred ))
from sklearn.metrics import roc_curve, auc
# Plot the receiver operating characteristic curve (ROC).
plt.figure(figsize=(10,8)) #創建一個大小為 10x8 的圖形
probas_ = clf.predict_proba(X_test) #probas_為已訓練的 SVM-clf 對測試集 X_test 的預測正確機率
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1]) #這一行計算了 ROC 曲線的假正例率（FPR）、真正例率（TPR）以及閾值。
#（False Positive Rate，FPR）表示被錯誤分類為正例的負例樣本的比例，FPR = FP / (FP + TN)，其中 FP 表示假正例（False Positives），TN 表示真負例（True Negatives）。
#（True Positive Rate，TPR）表示被正確分類為正例的正例樣本的比例，也稱為召回率，TPR = TP / (TP + FN)，其中 TP 表示真正例（True Positives），FN 表示假負例（False Negatives）。
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=1, label='ROC fold (area = %0.2f)' % (roc_auc)) #繪制 ROC 曲線
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random') #添加對角線
#設置x和y軸的範圍
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate') #設置x軸的標籤
plt.ylabel('True Positive Rate')  #設置了y軸的標籤
plt.title('Receiver operating characteristic example') #設置標題
#plt.axes().set_aspect(1) #設置坐標軸的等比例縮放，確保 ROC 曲線比例正確
plt.gca().set_aspect('equal', adjustable='box')  #plt.gca()用於獲得當前的坐標軸，equal表示坐標軸的寬高比是相等的，adjustable='box'：表示可以自動調整坐標軸的大小來實現寬高比為1
plt.show() # 顯示ROC圖表。
#Optimizing the SVM classifier
# We can tune two key parameters of the SVM algorithm:
# 1. the value of C (how much to relax the margin)小於1表示容許bias小、大於1表示容許bias大
# 2. the type of kernel:Radial Basis Function(rbf)
# C values with less bias and more bias (less than and more than 1.0 respectively).
# Python scikit-learn provides two simple methods for algorithm parameter tuning:
# 1. Grid Search Parameter Tuning.系統性地探索以找到最佳的超參數組合，以提高模型的性能。
# 2. Random Search Parameter Tuning. 
# Random Search 與 Grid Search 不同，它不是遍歷所有可能的超參數組合，而是在超參數的可能範圍內隨機選擇一組組合進行訓練和評估。在超參數的隨機樣本上運行多次來尋找最佳的超參數組合。
from sklearn.model_selection import GridSearchCV 
# Train classifiers.
#使用 Grid Search Parameter Tuning 來調整支持向量機（SVM）模型的超參數
kernel_values = [ 'linear' ,  'poly' ,  'rbf' ,  'sigmoid' ]#kernel_values 是一個包含不同核函數的列表，包括 'linear'（線性核函數）、'poly'（多項式核函數）、'rbf'（高斯徑向基函數）、'sigmoid'（S型核函數）。
param_grid = {'C': np.logspace(-3, 2, 6), 'gamma': np.logspace(-3, 2, 6),'kernel': kernel_values}
#定義要嘗試的超參數組合
# C（正則化參數）:用於控制模型的複雜度，
# gamma（核函數的係數）:核函數的係數，它影響了核函數的形狀。
# kernel（核函數的類型）
#np.logspace(-3, 2, 6) 是一個 NumPy 函數，生成一個在對數刻度上均勻分佈的數組，
# -3 是開始的指數。它表示我們希望生成的第一個數值為 10 的負 3 次方，即 0.001
# 2 是結束的指數。它表示我們希望生成的最後一個數值為 10 的 2 次方，即 100
#6 是生成的數值的總數。在這種情況下，我們希望生成 6 個數值，均勻分佈在指定的對數刻度範圍內。
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=5)#GridSearchCV 是 Scikit-Learn 中的一個用於網格搜索超參數的工具，透過交叉驗證來評估不同超參數組合的性能。cv=5 表示使用 5 折交叉驗證。
grid.fit(X_train, y_train)#開始進行網格搜索，試驗不同的超參數組合，並在訓練集 X_train 上訓練模型。
#grid.best_params_ 包含網格搜索中找到的最佳超參數組合，而 grid.best_score_ 則包含對應的最佳分數。
print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))
#grid.best_estimator_ 會返回經過Grid Search所找到的最佳模型，.probability = True可獲得模型每個類別的預測機率
grid.best_estimator_.probability = True
clf = grid.best_estimator_  # 以最佳超參數訓練模型，比較第6步驟的clf = SVC(probability=True)模型，並比較預測結果
y_pred = clf.fit(X_train, y_train).predict(X_test)#用找到的最佳模型clf對測試集 X_test 進行預測

cm = confusion_matrix(y_test, y_pred)
#print(cm)
print(classification_report(y_test, y_pred ))  # 輸出分類報告，包括精確度、召回率、F1 分數等性能指標。

fig, ax = plt.subplots(figsize=(5, 5))#創建了一個新的 Matplotlib 圖形（Figure）和一個軸（Axes），大小為 5x5。
ax.matshow(cm, cmap=plt.cm.Reds, alpha=0.3)    # 用熱圖可視化混淆矩陣，cmap 指定顏色，alpha設定透明度
#在混淆矩陣的可視化圖表上標記每個細胞（cell）的數值，以便更清楚地理解混淆矩陣的內容
for i in range(cm.shape[0]):
     for j in range(cm.shape[1]):
         ax.text(x=j, y=i,
                s=cm[i, j],
                va='center', ha='center')
#設置 x 軸、y 軸的標籤
plt.xlabel('Predicted Values', )
plt.ylabel('Actual Values')
plt.show()

## Decision boundaries of different classifiers -> see the decision boundaries produced by the linear, rbf and polynomial classifiers.
import matplotlib.pyplot as plt #matplotlib 的 pyplot 模組，用於繪製圖表和視覺化數據
from matplotlib.colors import ListedColormap #用於生成顏色映射
#decision_plot用於繪製支持向量機（SVM）的決策邊界
def decision_plot(X_train, y_train, n_neighbors, weights):
    h = .02  # step size in the mesh
    Xtrain = X_train[:, :2] #從訓練數據 X_train 中選擇前兩個特徵 

    #================================================================
    # Create color maps
    #================================================================
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])  #淺色的顏色映射
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])   #深色的顏色映射


    #================================================================
    # we create an instance of SVM and fit out data.
    # We do not scale ourdata since we want to plot the support vectors??
    #================================================================

    C = 1.0  # SVM 正則化參數的值
    #創建三種不同核函數的 SVM 分類器
    svm = SVC(kernel='linear', random_state=0, gamma=0.1, C=C).fit(Xtrain, y_train)
    rbf_svc = SVC(kernel='rbf', gamma=0.7, C=C).fit(Xtrain, y_train)
    poly_svc = SVC(kernel='poly', degree=3, C=C).fit(Xtrain, y_train)
    #設置圖表的尺寸和標題樣式
    plt.rcParams['figure.figsize'] = (15, 9)
    plt.rcParams['axes.titlesize'] = 'large'

    #創建一個網格，以便在二維特徵空間中繪製 SVM 決策邊界的輪廓。。
    x_min, x_max = Xtrain[:, 0].min() - 1, Xtrain[:, 0].max() + 1 #X的範圍為Xtrain 第一個特徵中最小值減去1，到找到最大值加上1。
    y_min, y_max = Xtrain[:, 1].min() - 1, Xtrain[:, 1].max() + 1 #同上
    #使用 np.meshgrid 函數創建了一個網格，xx,yy包含了所有可能的坐標點
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1))  # np.meshgrid-> Return coordinate matrices from coordinate vectors.意思是根據座標向量創建坐標矩陣，可應用於x、y坐標向量를網格向量對應成不同的圖??

    #創建一個標題列表 
    titles = ['SVC with linear kernel',
            'SVC with RBF kernel',
            'SVC with polynomial (degree 3) kernel']

    for i, clf in enumerate((svm, rbf_svc, poly_svc)):#設置迴圈，經過三個不同的 SVM 模型
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        plt.subplot(2, 2, i + 1)  #設定子圖的位置
        plt.subplots_adjust(wspace=0.4, hspace=0.4) #調整子圖的間距，wspace 和 hspace 分別表示寬度和高度的空白間距。
        # 用 clf（當前 SVM 模型）對網格中的所有點進行預測。
        #xx.ravel() 和 yy.ravel() 是網格中所有點的坐標，np.c_ 函數用於將這些坐標合併成一個二維數組
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) 

        # Put the result into a color plot
        #將預測結果 Z 轉換回與網格相同的形狀
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)  # plt.contourf繪製等高線圖

        # Plot also the training points
        plt.scatter(Xtrain[:, 0], Xtrain[:, 1], c=y_train, cmap=plt.cm.coolwarm)#繪製訓練數據集的散點圖
        plt.xlabel('radius_mean')
        plt.ylabel('texture_mean')
        #設定X和Y軸的範圍
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        #清除X和Y軸上的刻度標籤
        plt.xticks(())
        plt.yticks(())
        #設定子圖的標題
        plt.title(titles[i])

    plt.show()
decision_plot(X_train, y_train, 5, 'uniform')