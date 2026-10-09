# 多模态相册（Multimodal Album）

**LUMA · 光屿相册** —— 基于 CLIP 的本地智能相册。将照片管理、中英双语图文检索和 LoRA 微调整合在一个项目中，照片、收藏和相册数据保存在本机。

## 快速开始

需要 **Python 3.10+**（推荐 3.12）。先安装轻量依赖即可使用图库，不需要下载模型或安装 PyTorch：

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-web.txt
python main.py web
```

打开 **http://127.0.0.1:8000**，导入照片后即可浏览、收藏、编辑说明和标签、创建相册及关键词搜索。API 文档位于 **http://127.0.0.1:8000/docs**。界面资源均随项目提供，不依赖字体或脚本 CDN。

仓库不包含照片和模型权重，首次启动显示真实的空图库；不会用模拟数据冒充识别结果。也可将已有照片放入 `data/images/`，重启服务或点击图库刷新按钮即可扫描。

### 启用语义检索与自动打标

```bash
pip install -r requirements.txt
# 从 Hugging Face 下载模型（需要联网，有较大的模型文件）
python scripts/download_models.py --lang all
# 也可以先只下载英文模型：--lang en
python main.py web
```

在界面的「智能引擎」中选择已经安装的语言模型，更新索引。模型只在首次使用 AI 时加载，推理使用本地文件；设备自动选择 CUDA → Apple MPS → CPU，设置 `FORCE_CPU=1` 可强制 CPU。

中英文模型需要独立索引。只安装英文模型时可使用英文语义搜索、相似图和自动标签；中文语义搜索需要 Chinese-CLIP。没有模型或索引时，混合搜索会明确提示并退回关键词检索；纯语义模式会显示需要准备的内容。

## 本次增强

- **相册工作台**：响应式图库、横竖方图筛选、排序、分页、照片详情和键盘操作。
- **批量导入**：支持 JPEG / PNG / WEBP / BMP，每批最多 50 张、每张最多 25 MiB / 4000 万像素；内容哈希识别完全重复文件，坏图逐张反馈。
- **收藏与主题相册**：收藏、说明、标签、相册归属写入 SQLite，重启保留。同一照片可以出现在多个相册中。
- **混合检索**：文件名、说明和标签的 BM25 关键词结果与 CLIP 结果通过 RRF 排名融合；中文使用字与双字词组切分。支持收藏 / 相册 / 方向筛选。
- **照片探索**：详情内寻找相似照片、生成标签、保存个人说明；浏览器本地保留最近的搜索词。
- **增量索引**：未变照片复用向量，只计算新增或修改照片；移除已删除文件；模型文件变化时使缓存失效。后台显示进度，索引发布失败时保留旧版本。
- **轻量启动**：基础功能与模型依赖分离；修复 Mac / CPU 设备选择、不同 Transformers 特征输出和手机 EXIF 图片方向兼容。

关键词分数、语义余弦相似度和混合排序分数是不同的排序指标，**不是识别准确率或概率**。重复检测指文件内容完全相同，不会把经过压缩、裁剪的相似照片自动当作重复。

## 功能

| 功能 | 说明 |
|---|---|
| 🔍 文字搜图 | 用自然语言在相册里找照片，**中英文自动识别**（英文用 CLIP，中文自动切换 Chinese-CLIP）|
| 🖼️ 以图搜图 | 上传一张照片，找出视觉上/语义上最相似的其它照片 |
| 🏷️ 零样本自动打标 | 不用训练，自动给照片贴标签（人像/风景/美食/宠物…）并给出置信度 |
| 🎓 LoRA 微调 | 在自有照片上微调 CLIP，检索/分类效果显著提升 |
| 🌐 接口服务 | FastAPI 提供 REST 接口，方便对接其它系统 |
| 🖥️ 图形界面 | LUMA 相册工作台；保留原 Gradio 算法演示界面 |

## 技术栈

- **CLIP**（openai/clip-vit-base-patch32，经 Transformers 加载，本地 safetensors 离线运行）
- **LoRA 微调**（peft，冻结主干、只训练 0.65% 参数）
- **FAISS** 向量检索（余弦相似度，IndexFlatIP）
- **PyTorch / CUDA**（RTX 3050 4GB 可跑）
- **FastAPI + 原生 HTML/CSS/JavaScript** 相册工作台，SQLite 保存用户数据
- **Gradio** 保留独立算法演示界面

## 核心原理

1. CLIP 把图片和文字映射到同一个 512 维"语义向量空间"：相似的图文，向量靠得近。
2. 建索引时，把相册每张照片都变成向量，存入 FAISS（**英文、中文各建一套索引**）。
3. 搜图时，把查询文字/图片也变成向量，用 FAISS 找最近的 N 个向量，对应照片就是结果。
4. **中文检索**：英文 CLIP 的分词器读不懂中文，故额外接入 Chinese-CLIP（OFA-Sys/chinese-clip-vit-base-patch16，本地离线）。文字搜索自动检测查询语言，中文走中文索引、英文走英文索引，互不干扰。
5. **微调**：在自有照片上，用"图片 ↔ 类别描述"的对比学习损失（InfoNCE）训练 LoRA 适配器，让 CLIP 更懂你的相册。

### 命令行方式

```bash
# 增量更新中英文索引（需先下载两套模型）
python main.py index
python main.py index --force  # 强制重新编码全部照片

# 命令行搜图
python main.py search "a dog running in the park"

# 启动相册 / 原 Gradio 演示
python main.py web
python main.py ui
```

## 第二阶段：CLIP LoRA 微调

### 原项目实验记录（8 类相册检索任务，测试集 23 张）

以下为原项目记录的数据，本次产品功能更新未重新训练模型或复测这些指标；小测试集上的结果不代表通用场景效果。

| 指标 | 微调前（零样本） | 微调后（CLIP+LoRA） | 提升 |
|---|---|---|---|
| 分类准确率 | 86.96% | **91.30%** | **+4.3 个百分点** |
| 宏平均准确率 | 90.62% | **93.75%** | +3.1 |
| 检索召回率@3 | 79.17% | **90.62%** | **+11.5 个百分点** |

![效果对比](docs/效果对比.png)

弱类别改善：猫分类 50%→75%、运动类检索召回率 33%→100%。

### 复现

```bash
python main.py split      # 划分训练/测试集
python main.py baseline   # 微调前评估
python main.py finetune   # LoRA 微调（默认 25 epochs，约 1 分钟）
python main.py eval_ft    # 微调后评估
# 或一键跑完：
python main.py phase2
```

微调参数：LoRA r=16, alpha=32, dropout=0.1，只训练注意力层的 q/k/v/out 投影（0.65% 参数），
对比损失 InfoNCE，lr=5e-5，batch=8，25 epochs，fp16 混合精度，4GB 显存可跑。

## 项目结构

```
多模态相册/
├── app/
│   ├── clip_service.py   # CLIP 模型封装（图片/文字 -> 向量）
│   ├── indexer.py        # 建索引 / 读索引
│   ├── retriever.py      # 文字搜图 / 以图搜图
│   ├── tagger.py         # 零样本自动打标
│   ├── dataset.py        # 数据集加载与训练/测试划分
│   ├── finetune.py       # CLIP LoRA 微调
│   └── evaluate.py       # 分类准确率 + 检索 Recall@K 评估
├── api/server.py         # FastAPI 接口
├── app/library.py        # SQLite 图库、导入校验、缩略图与持久化
├── app/search_service.py # 元数据 BM25 与 RRF 混合检索
├── web/                  # LUMA 相册工作台（无需前端构建）
├── ui/app.py             # Gradio 界面
├── scripts/
│   ├── download_demo_data.py      # 从维基共享资源下载带标签演示照片
│   ├── download_picsum.py         # 从 picsum 补充通用照片
│   ├── download_chinese_clip.sh   # 下载中文 CLIP 模型（hf-mirror）
│   ├── download_models.py         # 下载官方中英文模型快照
│   └── plot_comparison.py         # 生成微调前后对比图
├── data/
│   ├── images/           # 相册照片（约 200 张，含 8 类带标签照片）
│   ├── index/            # FAISS 索引 + 元数据
│   ├── library.sqlite3   # 本机用户数据（不提交 Git）
│   ├── thumbnails/       # 本机缩略图缓存（不提交 Git）
│   ├── split.json        # 训练/测试划分
│   └── manifest.json     # 照片清单
├── models/
│   ├── clip-vit-base-patch32-local/        # 英文 CLIP 本地模型（离线可用）
│   ├── chinese-clip-vit-base-patch16-local # 中文 CLIP 本地模型（中文检索用）
│   └── clip-lora-album/                    # 微调后的 LoRA 适配器
├── docs/
│   └──效果对比.png
└── README.md
```

## 数据说明

- 带标签演示照片来自 **维基共享资源（Wikimedia Commons）**，通用照片来自 **picsum.photos**，均为可自由使用的公开图片。
- `manifest.json` 记录了每张照片的类别标签（dog/cat/...），用于微调与效果评估。

## 验证

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

测试使用临时照片、临时数据库和真实 FAISS 索引，编码器使用测试替身，不下载模型。覆盖导入校验、精确去重、持久化、相册关系、筛选检索、路径边界、增量向量复用、故障时保留旧快照及设备选择。真实 CLIP 推理需要额外准备权重后验证。

## 本地数据与使用边界

- 服务默认只监听 `127.0.0.1`，按单用户本地应用设计，没有登录或多用户隔离；并非可直接公开暴露的托管服务。
- 照片原件在 `data/images/`，个人整理数据在 `data/library.sqlite3`；备份时停止服务并一并保存这两部分。照片导入后不会修改原始内容，移出相册不会删除原件。
- 缩略图在 `data/thumbnails/`，可在停止服务后清理；搜索历史存于当前浏览器。
- 索引以不可变版本文件与原子 metadata 指针发布。为保证并发读取，旧版本暂不自动清理；大量更新后，可在停止所有服务时删除不再被 `meta.json` / `meta_zh.json` 的 `index_file` 引用的 `.index` 文件。
- 本版提供单进程后台索引任务，重启会结束任务；已发布索引仍保留，重新点击更新即可继续。多用户协作、视觉近似去重和更大数据集评估留待后续扩展。
