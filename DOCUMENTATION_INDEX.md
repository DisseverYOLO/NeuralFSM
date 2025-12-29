# 📚 NeuralFSM 文档索引

**快速找到你需要的文档！**

---

## 🎯 根据需求查找文档

### **✨ 最新更新（2024-11-13 V3集成）**
👉 **强烈推荐首先阅读**：
1. **`COMPLETE_INTEGRATION_SUMMARY.md`** ⭐⭐⭐⭐⭐ - **V3完整集成总结** 🆕🔥
   - 所有新功能详细说明（四目标损失、FSM验证、转移匹配等）
   - 关键问题全面解答
   - 核心代码示例
2. **`EXPERIMENT_SCRIPTS_CONSOLIDATED.md`** ⭐⭐⭐⭐⭐ - **实验脚本整合指南** 🆕🔥
   - 脚本详细对比（run_experiment_1_fsm.py vs run_experiment_1_fsm_complete.py）
   - 使用建议和迁移指南
   - 功能矩阵表格
3. **`neural_fsm_mas/core_integration.py`** - **核心集成模块** 🆕 (456行代码)

### **我是新手，想快速开始**
👉 阅读顺序：
1. `QUICK_START.md` ⭐⭐⭐ - **5分钟快速开始！**
2. **`COMPLETE_INTEGRATION_SUMMARY.md`** ⭐⭐⭐⭐⭐ - **V3完整功能** 🆕
3. `FSM_MODE_COMPLETE_GUIDE.md` ⭐⭐⭐ - **FSM模式完整指南** 🆕
4. `DATASET_USAGE_GUIDE.md` ⭐⭐⭐ - **数据集使用指南**
5. `CHANGES_AT_A_GLANCE.md` - 快速了解项目状态
6. `FINAL_PROJECT_STATUS.md` - 全面了解项目

### **我想理解实验一 (FSM优化)**
👉 阅读顺序：
1. **`TGN_MATHEMATICAL_FORMULATION.md`** ⭐⭐⭐⭐⭐ - **TGN数学公式与实现详解** 🆕🔥 **[论文核心！]**
   - 为什么选择TGN
   - TGN核心数学公式推导
   - TGN与FSM、智能体通信图的结合
   - 三目标损失函数完整推导
   - **监听路径预测与FSM智能体通信机制** ✅扩展
   - 论文撰写建议
2. **`STATE_COMPLETION_MECHANISM.md`** ⭐⭐⭐⭐⭐ - **状态完成判断机制详解** 🆕🔥 (必读！)
   - 状态完成判断的实现方式
   - LLM基于结构化标记判断
   - 增强版状态完成条件
   - 完整的状态转移流程
3. `FSM_LOSS_FUNCTIONS_EXPLAINED.md` ⭐⭐⭐⭐⭐ - **损失函数详解 (原版 vs V2版)** 🆕🔥 (必读！)
4. `EXPERIMENT_COMPARISON_GUIDE.md` ⭐⭐⭐⭐ - **实验对比指南** 🆕🔥 (实验设计)
5. `FSM_TRAINING_MODES_COMPARISON.md` ⭐⭐⭐ - **训练模式对比** 🆕 (架构对比)
6. `FSM_MODE_COMPLETE_GUIDE.md` ⭐⭐⭐ - **FSM模式完整指南** 🆕 (推荐)
7. `FSM_ARCHITECTURE_REFACTORING.md` ⭐⭐ - **FSM架构重构报告** 🆕
8. `EXPERIMENT1_FSM_OPTIMIZATION_GUIDE.md` ⭐⭐⭐ - **完整技术指南** (工程版)
9. `EXPERIMENT1_ACADEMIC_PAPER.md` ⭐⭐⭐ - **学术论文版** (理论+实验详解)
10. `neural_fsm_mas/README.md` - 模块说明
11. 代码: `train_fsm_mas_v2.py` (V2版) 🆕, `train_fsm_mas.py` (原版), `run_experiment_1_fsm.py` (简化版)
12. **FSM自动生成**: `baseclass/Enhanced_FSM_Gen.py` 🆕 (增强版LLM生成器)

### **我想理解实验二 (保护机制)**
👉 阅读顺序：
1. **`ATTACK_DEFENSE_INTEGRATION_COMPLETE.md`** ⭐⭐⭐⭐⭐ - **攻击-防御完整集成** 🆕🔥 (必读！)
2. **`DEFENSE_MECHANISMS_ANALYSIS.md`** ⭐⭐⭐⭐ - **文件结构分析** 🆕🔥 (重要！)
3. `DEFENSE_SIMPLIFIED_DESIGN.md` ⭐⭐⭐ - **核心设计**
4. `SIMPLIFIED_IMPLEMENTATION_GUIDE.md` ⭐⭐ - 实现细节
5. `MESSAGE_ATTENUATION_EXPLAINED.md` ⭐⭐ - 消息衰减
6. `PROTECTION_MECHANISM_FINAL_REPORT.md` - 完整报告
7. 代码: `advanced_attack_injector.py` (6种攻击) 🆕, `simplified_anomaly.py` (5种检测) ✅扩展

### **我想运行实验**
👉 阅读：
1. `QUICK_START.md` ⭐⭐⭐ - **快速开始** (数据集验证+运行)
2. `MODEL_CONFIG_UPDATE.md` ⭐ **NEW** - **LLM模型配置** (默认gpt-4o-mini)
3. `DATASET_USAGE_GUIDE.md` ⭐⭐⭐ - **数据集完整指南**
4. `UNIFIED_DATASET_GUIDE.md` ⭐⭐ - **统一数据集集成**
5. `EXPERIMENT_CHECKLIST.md` - 实验清单
6. `EVALUATION_SYSTEM_EXPLAINED.md` - 评估系统

### **我想集成到自己的项目**
👉 阅读：
1. `INTEGRATION_GUIDE.md` - 集成指南
2. `MULTI_DATASET_TRAINING_GUIDE.md` - 多数据集
3. `LLM_API_GUIDE.md` - LLM API

### **我遇到了问题**
👉 查看：
1. `QUICK_START_GUIDE.md` 的故障排除部分
2. `CODE_AUDIT_REPORT.md` - 已知问题
3. 运行 `python verify_imports.py` 验证环境

### **我想了解项目修改历史**
👉 阅读：
1. `CHANGES_AT_A_GLANCE.md` - 快速一览
2. `MODIFICATIONS_SUMMARY.md` - 详细总结
3. `CODE_AUDIT_REPORT.md` - 审计报告

---

## 📂 按文档类型分类

### **🌟 入门文档** (必读)

| 文档 | 优先级 | 用途 | 阅读时间 |
|------|--------|------|----------|
| `QUICK_START_GUIDE.md` | ⭐⭐⭐ | 快速开始，环境配置 | 20分钟 |
| `FINAL_PROJECT_STATUS.md` | ⭐⭐⭐ | 项目整体状态 | 15分钟 |
| `CHANGES_AT_A_GLANCE.md` | ⭐⭐ | 快速了解修改 | 5分钟 |

### **🔬 技术文档** (实验相关)

| 文档 | 实验 | 用途 | 阅读时间 |
|------|------|------|----------|
| **`TGN_MATHEMATICAL_FORMULATION.md`** 🆕 | 实验一 | **TGN数学公式与实现详解 [论文核心]** 🎓🔥 | 120分钟 |
| `EXPERIMENT1_FSM_OPTIMIZATION_GUIDE.md` | 实验一 | FSM/拓扑优化指南 (工程版) | 60分钟 |
| `EXPERIMENT1_ACADEMIC_PAPER.md` | 实验一 | **学术论文版 (理论+实验)** 🎓 | 90分钟 |
| `DEFENSE_SIMPLIFIED_DESIGN.md` | 实验二 | 保护机制设计 | 45分钟 |
| `SIMPLIFIED_IMPLEMENTATION_GUIDE.md` | 实验二 | 保护机制实现 | 40分钟 |
| `MESSAGE_ATTENUATION_EXPLAINED.md` | 实验二 | 消息衰减详解 | 30分钟 |
| `MESSAGE_WEIGHT_TRAINING_EXPLAINED.md` | 实验二 | **消息权重训练详解** 🆕 | 40分钟 |
| `PROTECTION_MECHANISM_FINAL_REPORT.md` | 实验二 | 完整技术报告 (含FAQ) | 50分钟 |

### **🧪 实验和评估**

| 文档 | 用途 | 阅读时间 |
|------|------|----------|
| `EXPERIMENT_CHECKLIST.md` | 实验验证清单 | 10分钟 |
| `EVALUATION_SYSTEM_EXPLAINED.md` | 评估系统详解 | 25分钟 |

### **🔧 使用和集成**

| 文档 | 用途 | 阅读时间 |
|------|------|----------|
| `INTEGRATION_GUIDE.md` | 如何集成到项目 | 30分钟 |
| `MULTI_DATASET_TRAINING_GUIDE.md` | 多数据集训练 | 25分钟 |
| `LLM_API_GUIDE.md` | LLM API使用 | 20分钟 |

### **📊 审计和总结**

| 文档 | 用途 | 阅读时间 |
|------|------|----------|
| `CODE_AUDIT_REPORT.md` | 代码审计报告 | 15分钟 |
| `MODIFICATIONS_SUMMARY.md` | 修改详细总结 | 20分钟 |
| `FINAL_IMPLEMENTATION_SUMMARY.md` | 实现总结 | 30分钟 |

### **📁 子模块文档**

| 文档 | 模块 | 用途 |
|------|------|------|
| `neural_fsm_mas/README.md` | neural_fsm_mas | 核心模块说明 |
| `baselines/README.md` | baselines | Baseline实现 |

---

## 🗺️ 学习路径推荐

### **路径 1: 快速上手** (1小时)
```
1. QUICK_START_GUIDE.md (20分钟)
   ↓
2. 运行 verify_imports.py (2分钟)
   ↓
3. 运行快速测试实验 (30分钟)
   ↓
4. CHANGES_AT_A_GLANCE.md (5分钟)
```

### **路径 2: 深入实验一** (3小时 - 工程视角)
```
1. QUICK_START_GUIDE.md (20分钟)
   ↓
2. EXPERIMENT1_FSM_OPTIMIZATION_GUIDE.md (60分钟) ⭐
   ↓
3. 阅读代码:
   - neural_fsm_mas/train_neural_mas.py (30分钟)
   - neural_fsm_mas/temporal_networks/neural_temporal_graph.py (30分钟)
   - neural_fsm_mas/agent_topology/multi_agent_topology.py (30分钟)
   ↓
4. 运行实验一 (30分钟)
```

### **路径 2b: 深入实验一** (4小时 - 学术视角) 🎓
```
1. QUICK_START_GUIDE.md (20分钟)
   ↓
2. EXPERIMENT1_ACADEMIC_PAPER.md (90分钟) ⭐⭐⭐
   - 重点: 第3节(问题形式化), 第4节(方法论), 第6节(实验结果)
   ↓
3. EXPERIMENT1_FSM_OPTIMIZATION_GUIDE.md (40分钟)
   - 对比工程实现
   ↓
4. 阅读代码验证理论 (60分钟)
   ↓
5. 运行实验并分析结果 (30分钟)
```

### **路径 3: 深入实验二** (4小时)
```
1. QUICK_START_GUIDE.md (20分钟)
   ↓
2. DEFENSE_SIMPLIFIED_DESIGN.md (45分钟) ⭐
   ↓
3. SIMPLIFIED_IMPLEMENTATION_GUIDE.md (40分钟)
   ↓
4. MESSAGE_ATTENUATION_EXPLAINED.md (30分钟)
   ↓
5. 阅读代码:
   - neural_fsm_mas/defense_mechanisms/protected_tgn.py (30分钟)
   - neural_fsm_mas/defense_mechanisms/simplified_centrality.py (20分钟)
   - neural_fsm_mas/defense_mechanisms/simplified_anomaly.py (20分钟)
   ↓
6. 运行实验二 (45分钟)
```

### **路径 4: 完整掌握** (8小时)
```
路径1 (1小时) → 路径2 (3小时) → 路径3 (4小时)
```

---

## 🔍 按关键词搜索

### **TGN / Neural Temporal Graph**
- `EXPERIMENT1_FSM_OPTIMIZATION_GUIDE.md` - 第2, 4.1节
- `neural_fsm_mas/temporal_networks/neural_temporal_graph.py` - 源码

### **FSM / 有限状态机**
- `EXPERIMENT1_FSM_OPTIMIZATION_GUIDE.md` - 第2.2, 4.2节
- `STATE_COMPLETION_MECHANISM.md` - **状态完成判断机制** 🆕🔥
- `baseclass/FSM_Gen.py` - FSM生成 (原版)
- `baseclass/Enhanced_FSM_Gen.py` - **增强版FSM生成器** 🆕 (针对数据集定制)

### **多智能体通信**
- `EXPERIMENT1_FSM_OPTIMIZATION_GUIDE.md` - 第4.2节
- `neural_fsm_mas/agent_topology/multi_agent_topology.py` - 源码

### **策略梯度 / REINFORCE**
- `EXPERIMENT1_FSM_OPTIMIZATION_GUIDE.md` - 第2.3, 5节
- `neural_fsm_mas/train_neural_mas.py` - 训练循环

### **图中心性 / Centrality**
- `DEFENSE_SIMPLIFIED_DESIGN.md` - 第3.1节
- `neural_fsm_mas/defense_mechanisms/simplified_centrality.py` - 源码

### **异常检测 / Anomaly Detection**
- `DEFENSE_SIMPLIFIED_DESIGN.md` - 第3.2节
- `neural_fsm_mas/defense_mechanisms/simplified_anomaly.py` - 源码

### **消息衰减 / Message Attenuation**
- `MESSAGE_ATTENUATION_EXPLAINED.md` - 完整说明
- `neural_fsm_mas/defense_mechanisms/protected_tgn.py` - 实现

### **保护损失 / Protection Loss**
- `DEFENSE_SIMPLIFIED_DESIGN.md` - 第3.5节
- `neural_fsm_mas/defense_mechanisms/protection_loss.py` - 源码

### **MMLU 数据集**
- `EXPERIMENT1_FSM_OPTIMIZATION_GUIDE.md` - 第7.1节
- `neural_fsm_mas/training_data/mmlu_data_processor.py` - 数据处理

---

## 📖 文档完整列表

### ⭐⭐⭐ 最高优先级 (6个)
1. `QUICK_START_GUIDE.md` - 快速开始
2. `FSM_MODE_COMPLETE_GUIDE.md` - **FSM模式完整指南** 🆕
3. `TGN_MATHEMATICAL_FORMULATION.md` - **TGN数学公式详解 [论文核心]** 🆕🔥
4. `STATE_COMPLETION_MECHANISM.md` - **状态完成判断机制** 🆕🔥
5. `FINAL_PROJECT_STATUS.md` - 项目状态
6. `DEFENSE_SIMPLIFIED_DESIGN.md` - 保护机制设计 ✅扩展

### ⭐⭐ 高优先级 (5个)
5. `FSM_ARCHITECTURE_REFACTORING.md` - **FSM架构重构报告** 🆕
6. `EXPERIMENT1_FSM_OPTIMIZATION_GUIDE.md` - 实验一技术指南
7. `SIMPLIFIED_IMPLEMENTATION_GUIDE.md` - 实现指南
8. `MESSAGE_ATTENUATION_EXPLAINED.md` - 消息衰减详解
9. `PROTECTION_MECHANISM_FINAL_REPORT.md` - 技术报告

### ⭐ 中优先级 (5个)
8. `EXPERIMENT_CHECKLIST.md` - 实验清单
9. `EVALUATION_SYSTEM_EXPLAINED.md` - 评估系统
10. `INTEGRATION_GUIDE.md` - 集成指南
11. `MULTI_DATASET_TRAINING_GUIDE.md` - 多数据集训练
12. `LLM_API_GUIDE.md` - LLM API指南

### ✅ 标准文档 (3个)
13. `FINAL_IMPLEMENTATION_SUMMARY.md` - 实现总结
14. `neural_fsm_mas/README.md` - 模块说明
15. `baselines/README.md` - Baseline说明

### 📝 审计文档 (4个)
16. `CODE_AUDIT_REPORT.md` - 审计报告
17. `MODIFICATIONS_SUMMARY.md` - 修改总结
18. `CHANGES_AT_A_GLANCE.md` - 快速一览
19. `DOCUMENTATION_INDEX.md` - **本文档**

---

## 🚀 快速命令参考

### **验证环境**
```bash
python verify_imports.py
```

### **查看实验帮助**
```bash
python run_experiment_1_baseline.py --help
python run_experiment_2_protected.py --help
python run_robustness_test.py --help
```

### **运行快速测试**
```bash
# 实验一
python run_experiment_1_baseline.py --num_epochs 1 --batch_size 4

# 实验二
python run_experiment_2_protected.py --num_epochs 1 --batch_size 4
```

### **可视化结果**
```bash
python visualize_results.py
```

---

## 💡 文档阅读建议

### **文档符号说明**
- ⭐⭐⭐ 最高优先级，必读
- ⭐⭐ 高优先级，推荐阅读
- ⭐ 中优先级，按需阅读
- ✅ 标准文档，参考查阅

### **如何高效阅读**

1. **第一次接触项目**
   - 先读 `QUICK_START_GUIDE.md`
   - 然后读 `CHANGES_AT_A_GLANCE.md`
   - 运行验证脚本确认环境

2. **准备运行实验**
   - 根据实验类型选择对应的技术指南
   - 实验一: `EXPERIMENT1_FSM_OPTIMIZATION_GUIDE.md`
   - 实验二: `DEFENSE_SIMPLIFIED_DESIGN.md`

3. **深入理解代码**
   - 先读技术文档理解原理
   - 再看实现指南了解细节
   - 最后阅读源码验证理解

4. **遇到问题**
   - 查看对应文档的"常见问题"部分
   - 参考审计报告了解已知问题
   - 运行验证脚本检查环境

---

## 📞 需要帮助？

### **找不到想要的信息？**
1. 使用本文档的"按关键词搜索"功能
2. 查看"根据需求查找文档"部分
3. 浏览完整文档列表

### **文档内容有疑问？**
1. 查看相关的其他文档交叉验证
2. 查看源码注释
3. 运行示例代码验证理解

### **发现文档错误？**
欢迎提交Issue或Pull Request！

---

## 📊 文档统计

| 类型 | 数量 | 总页数估计 |
|------|------|-----------|
| 入门文档 | 3 | ~50页 |
| 技术文档 | 5 | ~220页 |
| 实验文档 | 2 | ~40页 |
| 使用文档 | 3 | ~75页 |
| 审计文档 | 4 | ~60页 |
| 子模块文档 | 2 | ~20页 |
| **总计** | **19** | **~465页** |

---

## 🎯 文档质量保证

所有文档都经过：
- ✅ 内容与代码一致性检查
- ✅ 技术准确性验证
- ✅ 可读性审核
- ✅ 示例代码测试

**文档版本**: 1.0  
**最后更新**: 2025-11-03  
**维护者**: Neural FSM Team

---

**🎉 开始探索NeuralFSM的文档世界吧！** 🎉

