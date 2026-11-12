# -*- coding: utf-8 -*-
"""《模板GPT版(2)》标签同源融合、加权分组优化、W+RSGBD best-prefix（固定参数实验版）。

仅消费 1paper1_ball_fixed.py 产生的 GB_TEMPLATE_GPT2_V1 状态，
不兼容旧 *_gb_state.joblib（旧状态不含三通道独立来源）。
本文件不执行 RF/SVM：只为每一组 (alpha, delta, lambda) 输出原始属性索引列表；
分类和交叉验证由用户的独立分类器代码执行。只保存一个原始属性子集字典 joblib。
上游条件属性缺失值采用用户确认的众数/均值补全；标签缺失保留为无标记样本。
"""
from __future__ import annotations
from pathlib import Path
from itertools import combinations
from typing import Any
import datetime
import numpy as np
from joblib import load, dump
from scipy.linalg import solve
from scipy.optimize import minimize


def _label_key(x: Any):
    return (type(x).__name__, repr(x))


def _project_simplex(t):
    """Euclidean projection of a small vector onto Δ: numerical fallback."""
    x=np.asarray(t,dtype=float)
    u=np.sort(x)[::-1]
    rho=np.flatnonzero(u-(np.cumsum(u)-1)/(np.arange(len(x))+1)>0)
    theta=(np.sum(u[:rho[-1]+1])-1)/(rho[-1]+1)
    return np.maximum(x-theta,0)


def _canonical_weights(M, s, tol=1e-8):
    """≤3通道，遍历支持集求和同一个来源向量对应的最小 L2 范数权重。"""
    v=len(M)
    best=None
    for k in range(1,v+1):
        for subset in combinations(range(v),k):
            A=np.vstack([M[list(subset)].T, np.ones(k)])
            y=np.r_[s,1]
            weight=np.linalg.lstsq(A,y,rcond=None)[0]
            if np.min(weight)<-tol or np.linalg.norm(A@weight-y)>tol:
                continue
            t=np.zeros(v)
            t[list(subset)]=np.clip(weight,0,None)
            t/=t.sum()
            err=np.linalg.norm(M.T@t-s)
            if err>tol:
                continue
            val=np.dot(t,t)
            if best is None or val<best[0]:
                best=(val,t)
    if best is None:
        raise RuntimeError('来源融合 canonical simplex representation 数值失败')
    return best[1]


def _fusion_one(M, bias, ctheta, L, tol=1e-8, gap_tol=1e-6):
    """(3.16)(3.17) 的凸目标；计算中不读取 Y_L 的值。"""
    v,N=M.shape
    if v==1:
        s=M[0].copy()
        return np.ones(1),s,0.,0.
    def cost(t):
        s=M.T@t
        norm=np.linalg.norm(s)
        return float(L*(bias@t) + ctheta*norm)
    def grad(t):
        s=M.T@t
        return L*bias+ctheta*(M@s)/np.linalg.norm(s)
    starts=[np.ones(v)/v,*list(np.eye(v))]
    results=[]
    for t0 in starts:
        r=minimize(cost,t0,jac=grad,method='SLSQP',
                   bounds=[(0,1)]*v,
                   constraints=[{'type':'eq','fun':lambda t:np.sum(t)-1,
                                 'jac':lambda t:np.ones(v)}],
                   options={'maxiter':500,'ftol':1e-13})
        if r.success or (np.min(r.x)>=-tol and abs(r.x.sum()-1)<=tol):
            results.append((_project_simplex(r.x),float(cost(_project_simplex(r.x)))))
    if not results:
        raise RuntimeError('标签同源融合求解失败，未获得 simplex 上的可行点')
    t=min(results,key=lambda z:z[1])[0]
    s=M.T@t
    # 同一个源向量的最小L2权重，不改变主目标最优的来源配置。
    t=_canonical_weights(M,s,tol=1e-7)
    s=M.T@t
    g=grad(t)
    convex_gap=float(t@g-np.min(g))
    if convex_gap>gap_tol:
        # 做一轮切实的可行梯度细化；若仍不满足则报告而不是默默通过。
        for k in range(3000):
            next_t=_project_simplex(t-(0.2/np.sqrt(k+1))*grad(t))
            if cost(next_t)<=cost(t): t=next_t
            g=grad(t)
            if float(t@g-np.min(g))<=gap_tol: break
        s=M.T@t
        t=_canonical_weights(M,s,tol=1e-7)
        g=grad(t)
        convex_gap=float(t@g-np.min(g))
    if convex_gap>gap_tol or not np.isclose(t.sum(),1,atol=1e-8):
        raise RuntimeError(f'标签融合未达到凸间隙要求: {convex_gap:.5g}')
    return t,M.T@t,cost(t),convex_gap


def _bounded_simplex_project(p, lower, upper, tol=1e-10):
    """(3.21)(3.22)：对区间与 simplex 的交集求精确二次投影。"""
    p=np.asarray(p,dtype=float)
    lo=np.asarray(lower,dtype=float);up=np.asarray(upper,dtype=float)
    if np.any(lo>up+tol) or np.sum(lo)>1+tol or np.sum(up)<1-tol:
        raise ValueError('粗糙集概率约束确实不可行；禁止人为扩张边界')
    if abs(np.sum(lo)-1)<=tol: return lo.copy()
    if abs(np.sum(up)-1)<=tol: return up.copy()
    low=np.min(p-up);high=np.max(p-lo)
    for _ in range(100):
        mid=(low+high)/2
        q=np.clip(p-mid,lo,up)
        if q.sum()>1:low=mid
        else:high=mid
    q=np.clip(p-(low+high)/2,lo,up)
    if abs(q.sum()-1)>tol*10 or np.any(q<lo-tol) or np.any(q>up+tol):
        raise RuntimeError('有界 simplex 投影的残差超过容差')
    return q


class GroupedOriginalAttributeSelector:
    def __init__(self,name_ds,*,lip_L=1.0,vartheta=0.05,
                 alpha=1.,delta=1.,lambda_=1.,tau=1e-8,
                 optimization_tol=1e-7,optimization_max_iter=200,
                 fusion_gap_tol=1e-6, class_universe=None, run_tag=None):
        self.name_ds=name_ds
        self.run_tag=run_tag
        self.output_stem = name_ds if run_tag is None else f'{name_ds}__{run_tag}'
        self.lip_L=float(lip_L)
        self.vartheta=float(vartheta)  # 实验固定为0.05；L=1仅属经验设置，不当作有证书的概率界
        self.class_universe=None if class_universe is None else list(class_universe)
        self.alpha=float(alpha);self.delta=float(delta);self.lambda_=float(lambda_)
        self.tau=float(tau)
        self.optimization_tol=float(optimization_tol)
        self.optimization_max_iter=int(optimization_max_iter)
        self.fusion_gap_tol=float(fusion_gap_tol)
        self.project_dir=Path(__file__).resolve().parent.parent
        self.ball_dir=self.project_dir/'GranularBalls'
        self.result_dir=self.project_dir/'BO01paperRed'

    def load_granular_ball_state(self):
        path=self.ball_dir/f'{self.output_stem}_gb01.joblib'
        state=load(path)
        if state.get('version')!='GB_TEMPLATE_GPT2_V1':
            raise ValueError('检测到旧分球状态；请先运行新的 1paper1_ball_fixed.py')
        expected={'nmin':2,'kL':5,'kg':5,'kappa_g':1.0,'kappa_e':1.0,'beta':0.8}
        saved=state.get('parameters',{})
        if any(saved.get(k)!=v for k,v in expected.items()) or any(float(v)!=1.0 for v in saved.get('bandwidth',{}).values()):
            raise ValueError('分球状态的固定参数与新实验约定不符；请重新运行 1paper1_ball_fixed.py')
        return state

    @staticmethod
    def _classes(y,L,class_universe=None):
        classes=[];seen=set()
        values = [y[i] for i in L] if class_universe is None else class_universe
        for v in values:
            kv=_label_key(v)
            if kv not in seen:
                seen.add(kv);classes.append(v)
        observed={_label_key(y[i]) for i in L}
        if observed - seen:
            raise ValueError('已标记行含 class_universe 未列出的标签')
        if len(classes)<2:
            raise ValueError('论文设定 d >= 2，已标记训练行至少包含两个类别')
        ids={_label_key(x):j for j,x in enumerate(classes)}
        YL=np.zeros((len(L),len(classes)),dtype=float)
        for k,i in enumerate(L): YL[k,ids[_label_key(y[i])]]=1.
        return classes,YL

    def build_upstream(self, state):
        """仅构造一次，与 α,δ,λ 无关；所有通道的行列保持独立。"""
        if self.vartheta != 0.05 or self.lip_L != 1.0 or self.tau != 1e-8:
            raise ValueError('实验固定 vartheta=0.05, empirical_L=1, tau=1e-8，不能作为额外搜索参数')
        if not (0<self.vartheta<1 and self.lip_L>=0):
            raise ValueError('vartheta 或 L 不合法')
        Z=np.asarray(state['X'],dtype=float)
        n,D=Z.shape
        groups=state['feature_mapping']
        Lidx=np.asarray(state['labeled_indices'],dtype=int)
        Uidx=np.asarray(state['unlabeled_indices'],dtype=int)
        labels,YL=self._classes(state['y_raw'],Lidx,self.class_universe)
        d=len(labels)
        channels=state['channels']
        # channel output: Lambda 是独立样本来源系数，不因共享标签重复计算独立证据。
        ordered_channel_names=[x for x in ('mix','num','cat') if x in channels]
        Lambdas=np.stack([channels[v]['Lambda'] for v in ordered_channel_names],axis=0)
        Pstar={v:channels[v]['Kbeta'] @ channels[v]['B'] @ YL for v in ordered_channel_names}
        joint=channels['mix']
        # (3.16): 全部用联合几何评估到观测标签来源的距离。
        d_joint=np.zeros((n,len(Lidx)),dtype=float)
        for group in groups:
            cols=group['processed_indices']
            # 内存控制：按列组累计 n×N 的表，避免 n×N×D 张量。
            x=Z[:,cols];l=Z[Lidx][:,cols]
            if len(cols)==1:
                d_joint+=np.abs(x[:,None,0]-l[None,:,0])/len(groups)
            else:
                for i in range(n):
                    d_joint[i]+=np.linalg.norm(x[i][None,:]-l,axis=1)/len(groups)
        ctheta=float(np.sqrt(0.5*np.log(2*n*d/self.vartheta)))
        weights=np.zeros((n,len(ordered_channel_names)))
        S=np.zeros((n,len(Lidx)))
        convex_gap=np.zeros(n)
        fusion_objective=np.zeros(n)
        for i in range(n):
            M=Lambdas[:,i,:]
            biases=M@d_joint[i]
            ti,si,ei,gap=_fusion_one(M,biases,ctheta,self.lip_L,gap_tol=self.fusion_gap_tol)
            weights[i]=ti;S[i]=si;convex_gap[i]=gap;fusion_objective[i]=ei
        P=S@YL
        neff=1/np.sum(S**2,axis=1)
        R=joint['R'];Pm=Pstar['mix']
        # max-min / min-max supports (3.19): 不对窗口修补或重新缩放。
        lower=np.min(np.maximum(1-R[:,:,None],Pm[None,:,:]),axis=1)
        upper=np.max(np.minimum(R[:,:,None],Pm[None,:,:]),axis=1)
        Yproject=np.vstack([_bounded_simplex_project(P[i],lower[i],upper[i]) for i in range(n)])
        conflict=np.linalg.norm(P-Yproject,axis=1)
        Ytilde=Yproject.copy()
        Ytilde[Lidx]=YL
        # mixed channel evidence and entropy separations (3.26)--(3.29)
        p_safe=np.where(Pm>0,Pm,1)
        h=-np.sum(np.where(Pm>0,Pm*np.log(p_safe),0),axis=1)/np.log(d)
        rho=np.clip(1-h,0,1)
        e_star=joint['e_star']
        theta=e_star*rho
        BND=np.mean(upper-lower,axis=1)
        HBL=joint['Omega']@h
        xi=joint['Omega']@theta
        psi=np.maximum(0,1-conflict/np.sqrt(2.))
        gamma=(1-BND)*xi*psi
        gamma[Lidx]=1.
        if np.any(gamma<0) or np.any(gamma>1+1e-9):
            raise RuntimeError('gamma 离开 [0,1]')
        weighted_R=R*theta[None,:]
        row_sum=weighted_R.sum(axis=1,keepdims=True)
        eta=np.divide(weighted_R,row_sum,out=np.zeros_like(weighted_R),where=row_sum>0)
        # (3.30): 加权中心化，带截距。所有模型的 Γ 相同并在网格前冻结。
        mass=gamma.sum()
        Zbar=(gamma@Z)/mass
        Ybar=(gamma@Ytilde)/mass
        Zc=Z-Zbar
        Yc=Ytilde-Ybar
        gram=Zc.T@(gamma[:,None]*Zc)
        rhs=Zc.T@(gamma[:,None]*Yc)
        # H (3.32): joint mean centers, η; H PSD by construction.
        H=np.zeros((D,D))
        c_joint=np.stack([b['center'] for b in joint['balls']])
        for j in range(len(c_joint)):
            if not np.any(eta[:,j]):continue
            diff=Z-c_joint[j]
            H+=diff.T@(eta[:,j,None]*diff)
        H=(H+H.T)/2
        # group redundancy (3.31): weighted centered RV, ζ_aa=0.
        p=len(groups); zeta=np.zeros((p,p))
        whitened=[(np.sqrt(gamma)[:,None]*Zc[:,g['processed_indices']]) for g in groups]
        normG=np.asarray([np.linalg.norm(A.T@A,'fro') for A in whitened])
        for a in range(p):
            for b in range(a+1,p):
                if normG[a]>0 and normG[b]>0:
                    ab=np.linalg.norm(whitened[a].T@whitened[b],'fro')**2/(normG[a]*normG[b])
                    zeta[a,b]=zeta[b,a]=min(1.,max(0.,float(ab)))
        return dict(Z=Z,groups=groups,n=n,d=d,classes=labels,YL=YL,
                    labeled_indices=Lidx,unlabeled_indices=Uidx,
                    channel_names=ordered_channel_names,Pstar=Pstar,
                    source_matrix_by_channel=Lambdas,fusion_weights=weights,
                    source_weights=S,source_effective_n=neff,
                    fusion_objective=fusion_objective,fusion_convex_gap=convex_gap,
                    P_fused=P,Y_projected=Yproject,Y_tilde=Ytilde,
                    lower=lower,upper=upper,conflict=conflict,
                    rho=rho,e_star=e_star,theta=theta,BND=BND,HBL=HBL,
                    xi=xi,psi=psi,gamma=gamma,eta=eta,
                    Zc=Zc,Yc=Yc,Zbar=Zbar,Ybar=Ybar,
                    gram=gram,rhs=rhs,H=H,zeta=zeta)

    @staticmethod
    def _norms(W,groups):
        return np.asarray([np.linalg.norm(W[g['processed_indices'],:],'fro') for g in groups])

    def _objective(self,W,pre):
        norm=self._norms(W,pre['groups'])
        smooth=np.sqrt(norm**2+self.tau)
        residual=pre['Zc']@W-pre['Yc']
        loss=np.sum(pre['gamma'][:,None]*residual**2)
        struct=self.alpha*np.sum(W*(pre['H']@W))
        zeta=pre['zeta']
        redundancy=self.delta*0.5*np.sum(zeta*(smooth[:,None]+smooth[None,:])**2)
        sparse=self.lambda_*np.sum(smooth)  # default nu_a=1
        return float(loss+struct+redundancy+sparse)

    def _gradient(self,W,pre):
        groups=pre['groups']
        norm=self._norms(W,groups)
        s=np.sqrt(norm**2+self.tau)
        factor=self.lambda_/s+2*self.delta*(pre['zeta']@(s)+s*pre['zeta'].sum(axis=1))/s
        grad=2*((pre['gram']+self.alpha*pre['H'])@W-pre['rhs'])
        for k,g in enumerate(groups):
            grad[g['processed_indices'],:]+=factor[k]*W[g['processed_indices'],:]
        return grad

    def optimize_W(self,pre):
        """(3.37)--(3.38) 固定平滑 MM；梯度判据，记录迭代上限。"""
        if self.lambda_<=0 or self.alpha<0 or self.delta<0 or self.tau<=0:
            raise ValueError('要求 alpha,delta>=0；lambda,tau>0')
        groups=pre['groups'];D=pre['Z'].shape[1];d=pre['d']
        W=np.zeros((D,d));history=[]
        denominator=max(1,2*np.linalg.norm(pre['rhs'],'fro'))
        base=pre['gram']+self.alpha*pre['H']
        prev=self._objective(W,pre)
        for it in range(1,self.optimization_max_iter+1):
            s=np.sqrt(self._norms(W,groups)**2+self.tau)
            E=(pre['zeta']@(s)+s*pre['zeta'].sum(axis=1))/s
            Dv=1/(2*s)
            diagonals=np.zeros(D)
            for k,g in enumerate(groups):
                diagonals[g['processed_indices']]=self.delta*E[k]+self.lambda_*Dv[k]
            A=base+np.diag(diagonals)
            Wnext=solve(A,pre['rhs'],assume_a='pos',check_finite=True)
            current=self._objective(Wnext,pre)
            if current>prev+1e-8*max(1,abs(prev)):
                raise RuntimeError(f'MM 目标函数上升，迭代 {it}: {prev}→{current}')
            residual=float(np.linalg.norm(self._gradient(Wnext,pre),'fro')/denominator)
            step=float(np.linalg.norm(Wnext-W,'fro'))
            history.append(dict(iteration=it,objective=current,gradient_residual=residual,step_norm=step))
            W=Wnext;prev=current
            if residual<=self.optimization_tol:break
        result={'W':W,'intercept':pre['Ybar']-pre['Zbar']@W,
                'history':history,'converged':bool(history[-1]['gradient_residual']<=self.optimization_tol)}
        return result

    @staticmethod
    def _mass_weights(pre,ball_state):
        """(3.44)--(3.45) main reliable class mass, never legacy alternatives."""
        balls=ball_state['channels']['mix']['balls']
        m=len(balls);d=pre['d']
        mass=np.zeros((m,d))
        for j,b in enumerate(balls):
            ids=b['indices']
            mass[j]=pre['rho'][j]*np.sum(pre['gamma'][ids,None]*pre['Y_tilde'][ids],axis=0)
        m_l=pre['Y_tilde'].sum(axis=0)
        q_l=mass.sum(axis=0)
        kappa=np.divide(q_l,m_l,out=np.zeros_like(q_l),where=m_l>0)
        pi=np.divide(mass,q_l[None,:],out=np.zeros_like(mass),where=q_l[None,:]>0)
        return dict(w=mass,m_l=m_l,q_l=q_l,kappa=kappa,pi=pi)

    @staticmethod
    def _prefix_scan(pre,state,W,mass):
        """(3.43)--(3.49)：不限制候选比例，不贪心重排，不提前终止。"""
        groups=pre['groups']
        scores=GroupedOriginalAttributeSelector._norms(W,groups)
        # 全部属性按原始索引升序存储；先按分数降序再按原始索引升序。
        order=np.lexsort((np.array([g['original_feature_index'] for g in groups]),-scores))
        ordered=[groups[k] for k in order]
        m=len(state['channels']['mix']['balls']);d=pre['d']
        centers=np.stack([b['center'] for b in state['channels']['mix']['balls']])
        sum_center=np.zeros((m,m));sum_radius=np.zeros(m)
        CDD=[]
        kap=mass['kappa'];pi=mass['pi']
        # π_{jℓ}π_{qℓ}与不同类别对，仅与可靠类权重相关，可复用。
        same=np.zeros((m,m));cross=np.zeros((m,m))
        for ell in range(d):
            same+=kap[ell]*np.outer(pi[:,ell],pi[:,ell])/d
            for r in range(ell+1,d):
                cross+=2*kap[ell]*kap[r]*np.outer(pi[:,ell],pi[:,r])/(d*(d-1))
        for k,g in enumerate(ordered,1):
            cols=g['processed_indices']
            cg=centers[:,cols]
            if len(cols)==1:
                Dc=np.abs(cg[:,None,0]-cg[None,:,0])
            else:
                Dc=np.linalg.norm(cg[:,None,:]-cg[None,:,:],axis=2)
            rj=np.asarray([np.linalg.norm(pre['Z'][b['indices']][:,cols]-cg[j],axis=1).mean()
                           for j,b in enumerate(state['channels']['mix']['balls'])])
            sum_center+=Dc;sum_radius+=rj
            distance=sum_center/k;rad=sum_radius/k
            kernelC=1/(1+distance+rad[:,None]+rad[None,:])
            gap=np.maximum(0,distance-rad[:,None]-rad[None,:])
            kernelD=gap/(1+gap)
            C=float(np.sum(kernelC*same))
            D=float(np.sum(kernelD*cross))
            F=C*D
            if not (0<=C<=1+1e-8 and 0<=D<1+1e-8 and 0<=F<1+1e-8):
                raise RuntimeError('RSGBD C/D/F 超出理论界')
            CDD.append(dict(k=k,original_feature_index=int(g['original_feature_index']),
                            W_group_norm=float(scores[order[k-1]]),C=C,D=D,F=F))
        kmax=int(np.argmax([row['F'] for row in CDD]))+1
        maxscore=CDD[kmax-1]['F']
        reason=None
        if np.all(scores==scores[0]):
            reason='all_W_group_norms_exactly_equal'
        elif maxscore==0:
            reason='all_prefix_RSGBD_scores_zero'
        chosen=[int(g['original_feature_index']) for g in (groups if reason else ordered[:kmax])]
        # 完整集 fallback 返回的索引使用已保留原属性的原始顺序。
        return dict(selected_original_indices=tuple(chosen),
                    selected_processed_indices=tuple(c for g in (groups if reason else ordered[:kmax])
                                                     for c in g['processed_indices']),
                    W_ordered_original_indices=tuple(int(g['original_feature_index']) for g in ordered),
                    sorted_group_norms=tuple(float(scores[i]) for i in order),
                    k_star=kmax,returned_count=len(chosen),
                    fallback_reason=reason,best_RSGBD_score=maxscore,
                    full_prefix_score=CDD[-1]['F'],prefix_curve=CDD)

    def run_parameter_grid(self,grid_values=(.01,.1,1.,10.,100.)):
        """运行全部参数组合，只保存一个纯字典 joblib，不运行分类器。

        文件内容：{(alpha, delta, lambda): [原始属性索引, ...]}
        参数顺序：lambda 最慢、delta 次之、alpha 最快，均按 grid_values 升序。
        约简集按原始属性索引升序存放，不按分类精度或 RSGBD 分数重排。
        """
        state=self.load_granular_ball_state()
        pre=self.build_upstream(state)
        masses=self._mass_weights(pre,state)
        values=tuple(map(float,grid_values))
        if len(values)==0 or any(a<=0 for a in values):
            raise ValueError('参数网格需提供正数')
        if len(set(values))!=len(values):
            raise ValueError('参数网格包含重复值，会导致字典键被覆盖')
        if tuple(sorted(values))!=values:
            raise ValueError('参数网格应按升序提供，保证组合排列一致')
        # 循环顺序 lambda -> delta -> alpha；alpha 在最内层变化最快。
        combos=((alpha,delta,lam)
                for lam in values for delta in values for alpha in values)
        total=len(values)**3
        parameter_to_subset={}
        n_cols=pre['Z'].shape[1]
        group_by_original={int(g['original_feature_index']):g for g in pre['groups']}
        print(f'开始 {self.name_ds}: 保留原始属性数 p={len(pre["groups"])}，参数组数 {total}')

        for count,(alpha,delta,lam) in enumerate(combos,1):
            self.alpha=alpha
            self.delta=delta
            self.lambda_=lam
            opt=self.optimize_W(pre)
            selection=self._prefix_scan(pre,state,opt['W'],masses)
            key=(alpha,delta,lam)
            original_indices=tuple(map(int,selection['selected_original_indices']))
            processed_indices=tuple(map(int,selection['selected_processed_indices']))

            # 保留原有的映射一致性检查，不允许编码后的 One-Hot 列被部分选中。
            if not original_indices or len(set(original_indices)) != len(original_indices):
                raise ValueError(f'参数 {key} 的原始属性索引不合法：{original_indices}')
            if not processed_indices or len(set(processed_indices)) != len(processed_indices):
                raise ValueError(f'参数 {key} 的内部列索引不合法：{processed_indices}')
            if min(processed_indices)<0 or max(processed_indices)>=n_cols:
                raise ValueError(f'参数 {key} 的内部列索引越界（X列数={n_cols}）')
            try:
                expected=tuple(int(c) for a in original_indices
                               for c in group_by_original[a]['processed_indices'])
            except KeyError as exc:
                raise ValueError(f'参数 {key} 包含未经保留的原始属性：{exc}') from exc
            if processed_indices != expected:
                raise ValueError(f'参数 {key} 编码列与原始属性映射不一致')

            # 只保存原始属性编号；排序仅影响输出展示，不影响选中哪些特征。
            parameter_to_subset[key]=sorted(original_indices)
            if count==1 or count%10==0 or count==total:
                print(f'{count}/{total}，{key} -> {parameter_to_subset[key]}')

        self.result_dir.mkdir(parents=True,exist_ok=True)
        output_path=self.result_dir/f'{self.output_stem}_TheReduction01.joblib'
        dump(parameter_to_subset,output_path,compress=3)
        print(f'已保存唯一的原始属性约简集 joblib：{output_path}')
        return parameter_to_subset


if __name__=='__main__':
    # 当前实验约定：只有 alpha/delta/lambda 做网格搜索。
    VARTTHETA = 0.05
    EMPIRICAL_L = 1.0  # 论文4.1节的实验性设置；不当作有证书的概率误差界
    GRID_VALUES = (0.01,0.1,1.0,10.0,100.0)
    TAU = 1e-8  # 固定数值平滑常数，非理论超参数敏感性分析对象
    OPTIMIZATION_TOL = 1e-7
    OPTIMIZATION_MAX_ITER = 200
    # 例如已知所有类别为 [0,1,2]，可填入，否则只能根据已标记训练行推断类别。
    CLASS_UNIVERSE_BY_DATASET = {}

    dic_ds = {

        'Musk (Version 1)': ['Musk (Version 1)', 476, 166, 0],
        'Breast_Cancer': ['Breast_Cancer', 569, 30, 0],
        'Connectionist bench': ['Connectionist bench', 208, 60, 0],
        '191_DARWIN': ['191_DARWIN', 174, 450, 0],
        'Gastrointestinal lesions': ['Gastrointestinal lesions', 152, 698, 0],
        'Parkinsons_Classification_Complete': ['Parkinsons_Classification_Complete', 756, 753, 1],
        'Semeion handwritten': ['Semeion handwritten', 1593, 256, 256],
        'qsar': ['qsar', 1687, 1024, 1024],
        '180_Toxicity处理后的数据文件': ['180_Toxicity处理后的数据文件', 171, 1203, 163],




        ######属性大的
        # '179_Period Changer处理后的数据文件': ['179_Period Changer处理后的数据文件', 90, 1177, 151],#?
        # '180_Toxicity处理后的数据文件': ['180_Toxicity处理后的数据文件', 171, 1203, 163],#?
        # '181_MicroMass': ['181_MicroMass', 931, 1300, 0],
        # '13_SCADI': ['13_SCADI', 70, 205, 204,],
        # '191_DARWIN': ['191_DARWIN', 174, 450, 0],
        # '192_Myocardial': ['192_Myocardial', 1700, 122, 0]

        ##########样本太大的
        # '182_MAGIC Gamma Telescope': ['182_MAGIC Gamma Telescope', 19020, 10, 0],
        # '162_Localization Data for Person Activity': ['162_Localization Data for Person Activity', 164860, 7, 2],


    }
    for name_ds,n_samples,n_cond,n_cat in dic_ds.values():
        print('\n运行特征选择：',name_ds)
        selector=GroupedOriginalAttributeSelector(name_ds,
            lip_L=EMPIRICAL_L,vartheta=VARTTHETA,tau=TAU,
            optimization_tol=OPTIMIZATION_TOL,
            optimization_max_iter=OPTIMIZATION_MAX_ITER,
            class_universe=CLASS_UNIVERSE_BY_DATASET.get(name_ds))
        selector.run_parameter_grid(GRID_VALUES)
