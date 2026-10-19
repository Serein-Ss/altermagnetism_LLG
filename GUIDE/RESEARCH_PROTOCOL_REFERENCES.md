# 研究协议参考文献与证据登记

## 📚 修订9新增证据入口

2026-09-16物理调研及视频／世界模型讨论整理于[物理目标与路径模型计划](PATH_MODEL_AND_PHYSICS_PLAN.md)：10篇2025—2026年物理研究与6项方法来源。分别标注直接随机LLG、实验、Thiele及过渡态理论，不把相关物理动机当成全自旋模拟验证。Dreamer与Diffusion Forcing仅作方法借鉴；不据此宣称本项目首次提出随机转移或尺寸迁移。核查层级、已发表／预印本来源和未核实SI均在新表披露。下列历史登记保持原核查日期。

版本：2026-09-14，配套主协议修订6；保留历史来源核验范围。[返回主协议](PUBLICATION_GRADE_RESEARCH_PROTOCOL_20260911.md)

本文集中保存来源，不把被引用、摘要核验或代码内部一致性当成实验认证。启用某项文献结论前，证据表至少填写：论断、paper_id、正式版/预印本版本、公式/图号、SI位置、参数与几何、原条件/扩展、对照文件hash、容差、当前状态和未核实项。原图按实际许可再分发；否则保留链接、元数据和可分享派生图。

## 与主线的关系

| 来源 | 用途和边界 |
|---|---|
| Bauer2011 | 当前开放链的随机LL、反转机制和寿命；原文Eq.(2)、附录A及选定案例，相关长度/畴壁宽度条件见主协议2.2 |
| Gomonay2024 | 后续交错磁Hamiltonian、色散与纹理；细节见[交错磁附录](RESEARCH_PROTOCOL_AM_APPENDIX.md) |
| Weissenhofer2024 | 有限温sLLG方法及其自身三维模型对照；不为二维Gomonay热路径提供同参数真值 |
| ChenLipman2023 / Lipman2023 | RFM/FM方法来源；球面构造本身不作为本项目首创 |
| Timewarp2023 / TITO2026 | 最近邻方法比较：登记状态、lag、训练信息、路径/事件指标及成本。Timewarp的MCMC链不能当真实时间轨迹；TITO需逐方法核查 |
| Nishino2015 | 独立平衡校验；显式midpoint与几何隐式midpoint分别登记，保留勘误和接受偏差的原始记录 |
| Hirst2022 | 独立ASD/LLB/AFMR和热梯度壁复现；LLB允许纵向变化，不能使用单位球面模长证书 |
| Laliena2020/2022 | 独立BVP、临界分支和电流复现；残差小不等于参考值正确 |
| Cao2026 | [后续SOT规划](RESEARCH_PROTOCOL_EXTENSIONS.md#sot)候选来源；全文条件仍需核查 |
| GenCOMPAS2026 | 借鉴中间态未来到达概率、真实动力学分支验证与定向补数据；不当作完整动力学或尺寸推广的既成证明 |
| ITO2023 / gEDMD2020 | 借鉴多时间分辨率转移和样本驱动生成元／漂移扩散识别；仅作方法启发，不能替代本项目的真实LLG标签、CK三方检验或长时事件认证 |
| 其余条目 | 相关工作和材料证据候选；未用于主张前不自动成为主线前置任务 |

Nishino、Hirst、Laliena的详细原条件与验收继续见[严格文献计划](STRICT_LITERATURE_REPRODUCTION_PLAN.md)。各体系独立保留未完成项，共用内核错误按实际依赖传播。

RuO2模型与真实样品磁序确认分开；支持/质疑的完整证据表仍待核对，不能因采用模型就声称争议已解决。仅有外部审查者名称或不可访问路径不构成审查证据，实际报告须有日期、版本与hash。

## 来源清单

新增 **[GenCOMPAS2026]**：Tang, Pandey, Chen, Megías, Dehez and Chipot, *Breaking timescales with generative sampling of conformational transitions*, Nature (2026), DOI: [10.1038/s41586-026-11025-1](https://www.nature.com/articles/s41586-026-11025-1)；[作者代码与说明](https://github.com/Tangcyu/Gen-COMPAS)。本次依据此前实际读取的出版社公开摘要与作者README；出版社正文需订阅，补充材料未取得，未声称全文复现。

可借鉴范围：生成候选中间构型、committor筛选、真实短程模拟、概率权重与机制发现分开处理。作者仓库含定向MD、无偏MD及RiteWeight；不能把定向路径或扩散步直接看作物理时间，不能无推导移植平衡/稳态权重到本项目非平衡路径。小体系到大体系、等待时间与纯模型独立生成能力需本项目自身验证。当前落地为[分支先导配置](../tests/conf/committor_pilot.yaml)，不是移植蛋白质软件或改做生物体系研究。

1. **[Gomonay2024]** Gomonay et al., *Structure, control, and dynamics of altermagnetic textures*, npj Spintronics 2,35 (2024). https://www.nature.com/articles/s44306-024-00042-3
2. **[Bauer2011]** Bauer et al., *Thermally activated magnetization reversal in monoatomic magnetic chains on surfaces studied by classical atomistic spin-dynamics simulations*, JPCM 23,394204 (2011). https://doi.org/10.1088/0953-8984/23/39/394204 ; https://arxiv.org/abs/1010.4730
3. **[Nishino2015]** Nishino and Miyashita, *Realization of the thermal equilibrium in inhomogeneous magnetic systems by the Landau-Lifshitz-Gilbert equation with stochastic noise, and its dynamical aspects*, PRB 91,134411 (2015). https://arxiv.org/abs/1507.03075 ; erratum https://doi.org/10.1103/PhysRevB.97.019904
4. **[Hirst2022]** Hirst et al., *Temperature-dependent micromagnetic model of the antiferromagnet Mn2Au: A multiscale approach*, PRB 106,094402 (2022). https://doi.org/10.1103/PhysRevB.106.094402
5. **[Laliena2020]** Laliena et al., *Current-driven dynamics of chiral magnetic solitons in CrNb3S6*, Scientific Reports 10,20430 (2020). https://doi.org/10.1038/s41598-020-76903-8 ; correction https://doi.org/10.1038/s41598-022-06147-1
6. **[Rozsa2019]** Rózsa et al., *Reduced thermal stability of antiferromagnetic nanostructures*, PRB 100,064422 (2019). https://doi.org/10.1103/PhysRevB.100.064422 ; https://arxiv.org/abs/1808.07665
7. **[ChenLipman2023]** Chen and Lipman, *Flow Matching on General Geometries*, ICLR2024. https://arxiv.org/abs/2302.03660
8. **[Lipman2023]** Lipman et al., *Flow Matching for Generative Modeling*, ICLR2023. https://arxiv.org/abs/2210.02747
9. **[Timewarp2023]** Klein et al., *Timewarp: Transferable Acceleration of Molecular Dynamics by Learning Time-Coarsened Dynamics*, NeurIPS2023. https://arxiv.org/abs/2302.01170
10. **[TITO2026]** Viguera Diez et al., *Transferable generative models bridge femtosecond to nanosecond time-step molecular dynamics*, Science Advances 12,eaed2333 (2026). https://doi.org/10.1126/sciadv.aed2333 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC13060594/
11. **[ML_micromagnetics2021]** *Machine learning methods for the prediction of micromagnetic magnetization dynamics*. https://arxiv.org/abs/2103.09079
12. **[ML_magnetoelastic2021]** *Data-driven magneto-elastic predictions with scalable classical spin-lattice dynamics*, npj Computational Materials (2021). https://doi.org/10.1038/s41524-021-00617-2
13. **[ML_exchange2026]** Gao, Bokdam and Kelly, *Smooth overlap of spin orientations: Machine learning exchange fields for ab initio spin dynamics*, PRB 113,144413 (2026). https://doi.org/10.1103/kknv-7ypx （修订3记录已核验Crossref出版元数据；修订4未重复核验，逐方法比较仍待完成。）
14. **[RuO2_challenge2024]** Plouff et al., *Revisiting altermagnetism in RuO2: a study of laser-pulse induced charge dynamics by time-domain terahertz spectroscopy*. https://arxiv.org/abs/2412.11240
15. **[MerminWagner1966]** Mermin and Wagner, *Absence of Ferromagnetism or Antiferromagnetism in One- or Two-Dimensional Isotropic Heisenberg Models*, PRL17,1133 (1966). https://doi.org/10.1103/PhysRevLett.17.1133

16. **[Weissenhofer2024]** Weißenhofer and Marmodoro, *Atomistic spin dynamics simulations of magnonic spin Seebeck and spin Nernst effects in altermagnets*, PRB 110,094427 (2024). https://doi.org/10.1103/PhysRevB.110.094427 ; https://arxiv.org/abs/2405.20921
17. **[Cao2026]** Cao et al., *Magnetization dynamics of altermagnet driven by spin-orbit torque*, Acta Physica Sinica 75,060709 (2026). https://doi.org/10.7498/aps.75.20251628 ; https://wulixb.iphy.ac.cn/en/article/doi/10.7498/aps.75.20251628
18. **[Mentink2010]** Mentink et al., *Stable and fast semi-implicit integration of the stochastic Landau-Lifshitz equation*, JPCM 22,176001 (2010). [原文](https://arxiv.org/abs/1002.1801)。
19. **[ITO2023]** Schreiner, Winther and Olsson, *Implicit Transfer Operator Learning: Multiple Time-Resolution Surrogates for Molecular Dynamics*. [原文](https://arxiv.org/abs/2305.18046)。用于多时间分辨率转移的概念借鉴，不作为磁性动力学已验证实现。
20. **[gEDMD2020]** Klus et al., *Data-driven approximation of the Koopman generator: Model reduction, system identification, and control*. [原文](https://arxiv.org/abs/1909.10638)。用于从随机轨迹估计生成元、漂移和扩散的概念借鉴；有限样本、流形几何及稀有事件仍需本项目独立验证。

这是来源登记，不声称修订6重读所有论文/SI。修订3记录曾核查Bauer/Gomonay相关原文、Weissenhofer附录C及相关结果，Cao仅核查发表信息与摘要；完整路径包可用性均未据此认证。投稿前补全逐论断证据及元数据。

## 修订4公式核对入口

- [Bauer原文Eq.(2)、机制条件及附录A](https://arxiv.org/html/1010.4730v2)：特殊LL、时间单位、噪声和文献寿命定义。
- [Gomonay原文及SI](https://arxiv.org/html/2403.10218v3)：S.8–S.16与参数表；附录保留正幅值到有符号键的对应。
- [RFM原文](https://arxiv.org/abs/2302.03660)：几何概率路径和流匹配构造。

上述是审查依据和定位入口，不代表修订4运行了复现实验或确认了公开完整路径数据包。
