import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.linear_model import SGDOneClassSVM
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.tree import ExtraTreeRegressor
from pathlib import Path
p=Path(__file__).parent
X=np.array([[4,6],[8,9],[7,8],[11,13],[5,6]])
lof=LocalOutlierFactor(n_neighbors=2);lof.fit_predict(X)
print('Task 1 LOF:',dict(zip('ABCDE',np.round(-lof.negative_outlier_factor_,4))))
Y=np.array([[4,6,8,8],[8,6,10,2],[4,8,10,10],[16,18,12,14]])
# three actual isolation trees with deterministic independent random splits
rng=np.random.default_rng(42)
def tree(points,depth=0):
 if len(points)<=1:return depth,[]
 choices=[j for j in range(4) if np.ptp(Y[points,j])>0]
 if not choices:return depth,[]
 j=int(rng.choice(choices));low=Y[points,j].min();high=Y[points,j].max();t=float(rng.uniform(low,high))
 left=[i for i in points if Y[i,j]<t];right=[i for i in points if Y[i,j]>=t]
 dl,sl=tree(left,depth+1);dr,sr=tree(right,depth+1)
 return None,[(depth,j,t,''.join('ABCD'[i] for i in points),''.join('ABCD'[i] for i in left),''.join('ABCD'[i] for i in right))]+sl+sr
paths=[]
for z in range(3):
 # generate one shared tree, record each point's depth via splits
 def rec(ids,d=0):
  if len(ids)<=1 or all(np.all(Y[i]==Y[ids[0]]) for i in ids):
   for i in ids: dep[i]=d
   return
  dims=[j for j in range(4) if np.ptp(Y[ids,j])>0]
  j=int(rng.choice(dims));t=float(rng.uniform(Y[ids,j].min(),Y[ids,j].max()))
  l=[i for i in ids if Y[i,j]<t];r=[i for i in ids if Y[i,j]>=t]
  splits.append((d,j,round(t,3),''.join('ABCD'[i] for i in l),''.join('ABCD'[i] for i in r)))
  rec(l,d+1);rec(r,d+1)
 dep={};splits=[];rec(list(range(4)));paths.append([dep[i] for i in range(4)])
 print('Tree',z+1,'splits:',splits,'path lengths:',paths[-1])
print('Average path lengths',np.mean(paths,axis=0))
iris=load_iris();raw=iris.data[:,[2,3]];x=StandardScaler().fit_transform(raw)
models=[('One-Class SVM',OneClassSVM(nu=.15,kernel='rbf',gamma=.5)),('SGD One-Class SVM',make_pipeline(Nystroem(gamma=.5,n_components=100,random_state=42),SGDOneClassSVM(nu=.15,random_state=42,max_iter=3000,tol=1e-5))),('Isolation Forest',IsolationForest(contamination=.15,random_state=42,n_estimators=150)),('Local Outlier Factor',LocalOutlierFactor(n_neighbors=20,contamination=.15,novelty=True))]
xx,yy=np.meshgrid(np.linspace(x[:,0].min()-.7,x[:,0].max()+.7,250),np.linspace(x[:,1].min()-.7,x[:,1].max()+.7,250));grid=np.c_[xx.ravel(),yy.ravel()]
fig,axs=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
for ax,(name,model) in zip(axs.ravel(),models):
 model.fit(x);scores=model.decision_function(grid).reshape(xx.shape);pred=model.predict(x)
 ax.contourf(xx,yy,scores,levels=20,cmap='RdYlBu',alpha=.45)
 if scores.min()<0<scores.max():ax.contour(xx,yy,scores,levels=[0],colors='black',linewidths=1.5)
 ax.scatter(x[pred==1,0],x[pred==1,1],c='steelblue',edgecolors='white',s=27,label='Inlier')
 ax.scatter(x[pred==-1,0],x[pred==-1,1],c='crimson',edgecolors='black',s=45,label='Outlier')
 ax.set_title(f'{name} — {sum(pred==-1)} flagged');ax.set_xlabel('Petal length (standardized)');ax.set_ylabel('Petal width (standardized)');ax.legend(loc='best',fontsize=8)
fig.suptitle('Iris: 2D anomaly detection algorithm comparison',fontsize=16)
fig.savefig(p/'iris_anomaly_comparison.png',dpi=180);plt.close(fig)
