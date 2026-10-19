import json,re
from pathlib import Path
import numpy as np
from tests.transport_audit.report import table
from tests.bauer_p2_20260923.common import dump

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def main():
    a=json.loads((HERE/'diagnosis/report.json').read_text());s=json.loads((HERE/'diagnosis/summary.json').read_text())
    rows=[[x['arm'],x['lag'],x['covariance_rmse'],x['covariance_squared_bias_crossfold'],x['reference_squared_bias_crossfold'],x['signed_trace_difference']] for x in s]
    boundary=[]
    for arm in ('control','reversal'):
        rr=[x for x in a['conditional'] if x['task']['repair']==arm]
        boundary.append([arm,len(rr),*np.mean([[x['metrics']['endpoint_mean_rmse'],x['metrics']['interior_mean_rmse']] for x in rr],0)])
    body='<h1>下一轮前置诊断：跨格点涨落关联</h1><p>读取上一轮6个冻结模型的已有生成HDF，不重训、不新生成数据。上一轮链反转修复改善短时均值和协方差，但完整1ps联合U距离平均0.00354070→0.00403991，未闭环。该整体描述并非三seed置信区间或物理通过门槛。</p>'
    body+='<h2>局部协方差漏掉了什么</h2><p>旧指标是同一格点内部3x3协方差。这里对条件增量ΔS减去集合均值，再计算Cov(ΔS_i,ΔS_j)完整3x3矩阵。相同单格点分布可以有不同的格点间联合分布；新增回归用逐格点独立打乱分支验证这种情况。物理上它描述共同涨落及交换传播关联，不是平衡态认证。</p>'
    body+=table(['arm','distance','tensor RMSE','crossfold squared bias','base/fine squared bias','signed trace bias'],rows)
    body+='<p>Crossfold squared bias将生成及参考各分两个独立半集合，使用Cov差值的交叉积。独立分支假设下估计真实协方差差的Frobenius平方，而不是直接把样本协方差误差平方当总体偏差；估计可负，不裁零。不提供95%区间或硬性通过，单次拆分和参考有限样本限制保留。参考base/fine不是容差。每臂216行是72状态×3seed，状态相关且参考重复，不是216独立父状态。距离8的短时信号弱，不据此定义失败。</p>'
    body+='<h2>边界还是链内部</h2>'+table(['arm','rows including related states','two-site endpoints meanRMSE','interior meanRMSE'],boundary)
    body+='<p>均值偏差并非仅边界问题，反转修复后内部误差仍存在。当前描述性证据支持进一步检查距离1/2/4协方差，但不能证明这是全部误差唯一来源，时间接续和更高阶分布仍待检查。</p>'
    body+='<h2>下一轮受控修复</h2><p>三个seed分别从各自reversal120144模型开始，同期control/local_moments/spatial_moments。后两者共同约束局部均值和自身协方差，空间臂增加间隔1/2/4的全3x3涨落关联。只使用原TRAIN状态独立未来，未改Hamiltonian、温度、边界、时间或split。6000更新到126144，完整评估非目标能量、时间/空间、更高阶联合分布和成本，不通过增加flow步数消除模型偏差。</p>'
    body+='<p>保留已知联合距离v2近似区间未认证的限制；本页不做新的多重显著性筛选，不设事后容差，不以没有事件判失败。L16/25/32已训练，单温度63.824849668K开放链；两帧0.1ps模型10次接续1ps，参考.25/.125fs，保存.1ps，flow128/512严格分开。</p><a href="report.json">每状态/时间/格点详细矩阵与元数据</a> | <a href="../smoke_assets/index.html">修复预检（完成后生成）</a> | <a href="../assets/index.html">正式对照（完成后生成）</a>'
    (HERE/'diagnosis/index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><body style="max-width:1500px;margin:30px auto;font:16px/1.6 sans-serif">'+body+'</body></html>')
    path=ROOT/'tests/index.html';text=path.read_text();start='<!-- spatial-covariance-diagnosis:start -->';end='<!-- spatial-covariance-diagnosis:end -->'
    block=start+'<section id="spatial-covariance-diagnosis"><h2>Bauer跨格点涨落协方差诊断</h2><a href="spatial_covariance_repair/diagnosis/index.html">冻结路径诊断、边界/内部误差、下一轮对照</a></section>'+end
    text=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:block,text,flags=re.S) if start in text else text.replace('</body>',block+'</body>')
    tmp=path.with_suffix('.html.tmp');tmp.write_text(text);tmp.replace(path)


if __name__=='__main__':main()
