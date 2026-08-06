# CUDA Kernel Academy 路线图

> 定位：五仓学习路径的**总入口与 CUDA 系统教学仓**，当前处于维护模式。
> 旗舰开发投入在 [tiny-llm](https://github.com/AICL-Lab/tiny-llm)（运行时）与
> [cuflash-attn](https://github.com/AICL-Lab/cuflash-attn)（kernel 深度）。

## 面试前必做（数据可信度）

- [ ] **复核 SGEMM 优化阶梯的全部 TFLOPS 数字**（01-sgemm-tutorial 与 03-hpc-advanced 文档）：
      在自己可支配的 GPU 上重跑每一步并记录硬件与复现命令；凡是引自教学材料而非自测的数字，
      要么重测覆盖，要么在文档中明确标注来源
- [ ] 为优化阶梯补一份 nsys/ncu 实操示例（真实 trace 截图 + 瓶颈解读）
- [ ] 检查 docs 站点构建无死链（Pages workflow 为绿色）

## 结构维护

- [ ] 保持 LEARNING_PATH.md 作为五仓导航的唯一权威入口，新增仓库或改名时同步更新
- [ ] 保持 04-inference-engine 与 tiny-llm 的边界说明：前者是"kernel/内存/流如何组装进小系统"的教学演示，
      后者是面向真实模型权重的运行时（避免面试官问"这两个什么关系"时答不清）
- [ ] 每个模块保持"一个教学角色只留一份实现"的既有收敛原则

## 明确不做

- 不再新增模块或扩大范围；新想法优先进入对应专项仓库
- 不在文档中写入未经本机测量的性能数字
