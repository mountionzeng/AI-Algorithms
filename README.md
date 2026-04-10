# 📚 仓库结构

```plaintext
AI-Algorithms-Journey/
│
├── README.md                          # 本文件
├── LICENSE                            # MIT License
├── .gitignore                         # Git 忽略文件
│
├── 01-Mathematical-Foundations/       # 数学基础
│   ├── README.md
│   ├── linear-algebra/                # 线性代数
│   │   ├── notes/
│   │   │   ├── 01-vectors-matrices.md
│   │   │   ├── 02-eigenvalues.md
│   │   │   └── 03-svd.md
│   │   └── implementations/
│   │       ├── matrix_operations.py
│   │       └── pca_from_scratch.py
│   │
│   ├── probability-theory/            # 概率论
│   │   ├── notes/
│   │   │   ├── 01-bayes-theorem.md
│   │   │   ├── 02-distributions.md
│   │   │   └── 03-mle-map.md
│   │   └── implementations/
│   │       └── bayesian_inference.py
│   │
│   └── optimization/                  # 优化理论
│       ├── notes/
│       │   ├── 01-gradient-descent.md
│       │   ├── 02-convex-optimization.md
│       │   └── 03-lagrange-multipliers.md
│       └── implementations/
│           ├── gradient_descent.py
│           └── adam_optimizer.py
│
├── 02-Machine-Learning/               # 机器学习原理
│   ├── README.md
│   ├── CS586-Course-Notes/            # CS586 课程笔记
│   │   ├── lectures/
│   │   ├── assignments/
│   │   └── final-project/
│   │
│   ├── supervised-learning/           # 监督学习
│   │   ├── linear-regression/
│   │   │   ├── theory.md
│   │   │   ├── from_scratch.py
│   │   │   └── sklearn_comparison.py
│   │   ├── logistic-regression/
│   │   └── svm/
│   │
│   ├── unsupervised-learning/         # 无监督学习
│   │   ├── kmeans/
│   │   ├── pca/
│   │   └── clustering/
│   │
│   ├── optimization-algorithms/       # 优化算法
│   │   ├── sgd.py
│   │   ├── adam.py
│   │   ├── rmsprop.py
│   │   └── comparison.ipynb
│   │
│   ├── regularization/                # 正则化
│   │   ├── l1_l2_regularization.md
│   │   ├── dropout.py
│   │   └── early_stopping.py
│   │
│   └── projects/
│       └── ml-pipeline-from-scratch/  # 从零实现 ML 管线
│
├── 03-Deep-Learning/                  # 深度学习架构
│   ├── README.md
│   ├── CS583-Course-Notes/            # CS583 课程笔记
│   │   ├── lectures/
│   │   ├── assignments/
│   │   └── final-project/
│   │
│   ├── neural-networks/               # 神经网络基础
│   │   ├── backpropagation/
│   │   │   ├── theory.md
│   │   │   ├── manual_backprop.py
│   │   │   └── visualization.ipynb
│   │   ├── activation-functions/
│   │   └── initialization/
│   │
│   ├── cnn/                           # 卷积神经网络
│   │   ├── architectures/
│   │   │   ├── lenet.py
│   │   │   ├── alexnet.py
│   │   │   ├── vgg.py
│   │   │   ├── resnet.py
│   │   │   └── architecture_comparison.md
│   │   ├── convolution-operations/
│   │   │   ├── conv2d_from_scratch.py
│   │   │   └── pooling_layers.py
│   │   └── projects/
│   │       └── image-classification/
│   │
│   ├── transformer/                   # Transformer
│   │   ├── attention-mechanism/
│   │   │   ├── self_attention.py
│   │   │   ├── multi_head_attention.py
│   │   │   └── attention_visualization.ipynb
│   │   ├── positional-encoding/
│   │   ├── transformer_from_scratch.py
│   │   └── projects/
│   │       └── mini-gpt/
│   │
│   ├── diffusion-models/              # 扩散模型
│   │   ├── theory/
│   │   │   ├── ddpm.md
│   │   │   ├── ddim.md
│   │   │   └── stable_diffusion.md
│   │   ├── implementations/
│   │   │   ├── simple_diffusion.py
│   │   │   └── unet.py
│   │   └── projects/
│   │       └── image-generation/
│   │
│   └── training-techniques/           # 训练技巧
│       ├── batch_normalization.py
│       ├── learning_rate_scheduling.py
│       └── transfer_learning.py
│
├── 04-Computer-Vision/                # 计算机视觉 - AI 美术
│   ├── README.md
│   ├── image-processing/              # 图像处理算法
│   │   ├── basic-operations/
│   │   │   ├── filtering.py
│   │   │   ├── edge_detection.py
│   │   │   └── morphology.py
│   │   ├── color-spaces/
│   │   │   ├── rgb_hsv_lab.py
│   │   │   └── color_matching.py
│   │   └── advanced/
│   │       ├── image_segmentation.py
│   │       ├── super_resolution.py
│   │       └── style_transfer.py
│   │
│   ├── object-detection/              # 目标检测
│   │   ├── yolo/
│   │   ├── faster_rcnn/
│   │   └── projects/
│   │
│   ├── image-generation/              # 图像生成
│   │   ├── gan/
│   │   ├── vae/
│   │   └── stable-diffusion/
│   │       ├── controlnet_integration.py
│   │       ├── lora_training.py
│   │       └── inpainting.py
│   │
│   └── projects/
│       ├── matte-painting-tools/      # MP 工具链
│       │   ├── README.md
│       │   ├── auto_projection.py
│       │   ├── intelligent_stitching.py
│       │   ├── ai_assisted_repair.py
│       │   └── nuke_integration/
│       │
│       └── ai-art-workflow/           # AI 美术工作流
│           ├── concept_to_final.md
│           └── production_pipeline/
│
├── 05-Natural-Language-Processing/    # 自然语言处理
│   ├── README.md
│   ├── CS584-Course-Notes/            # CS584 课程笔记
│   │   ├── lectures/
│   │   └── assignments/
│   │
│   ├── text-preprocessing/            # 文本预处理
│   │   ├── tokenization.py
│   │   ├── normalization.py
│   │   └── stopwords.py
│   │
│   ├── word-embeddings/               # 词嵌入
│   │   ├── word2vec/
│   │   │   ├── skip_gram.py
│   │   │   └── cbow.py
│   │   ├── glove/
│   │   └── visualization.ipynb
│   │
│   ├── language-models/               # 语言模型
│   │   ├── rnn_lstm/
│   │   ├── gpt/
│   │   └── bert/
│   │
│   └── projects/
│       └── ai-social-product/         # AI 社交产品
│           ├── README.md
│           ├── llm_integration.py
│           ├── prompt_engineering/
│           └── personality_analysis/
│
├── 06-Computer-Graphics/              # 计算机图形学 - 传统美术
│   ├── README.md
│   ├── rendering-pipeline/            # 渲染管线
│   │   ├── rasterization.md
│   │   ├── pbr_materials.md
│   │   └── lighting_models.py
│   │
│   ├── shaders/                       # 着色器
│   │   ├── glsl/
│   │   ├── hlsl/
│   │   └── procedural_textures/
│   │
│   └── visual-effects/                # 视觉特效
│       ├── particle_systems.py
│       ├── fluid_simulation.md
│       └── shadow_mapping.py
│
├── 07-AI-Tools-Mastery/               # AI 工具精通
│   ├── README.md
│   ├── midjourney/                    # Midjourney
│   │   ├── prompt-engineering/
│   │   │   ├── systematic_approach.md
│   │   │   ├── parameter_guide.md
│   │   │   └── examples/
│   │   ├── style-transfer/
│   │   └── projects/
│   │       └── changan-style-concepts/  # 长安风格概念图
│   │
│   ├── comfyui/                       # ComfyUI
│   │   ├── workflow-design/
│   │   │   ├── node_logic.md
│   │   │   ├── controlnet_workflows.json
│   │   │   └── lora_workflows.json
│   │   ├── automation/
│   │   │   ├── batch_generation.py
│   │   │   └── api_integration.py
│   │   └── projects/
│   │       └── film-art-pipeline/     # 影视美术流程
│   │
│   └── post-processing/               # 后期优化
│       ├── common_issues.md
│       ├── inpainting_techniques.py
│       └── color_correction.py
│
├── 08-Integration-Projects/           # 综合项目
│   ├── README.md
│   ├── nuke-plugin-development/       # NUKE 插件开发
│   │   ├── opencv_integration/
│   │   ├── ai_algorithms/
│   │   └── performance_optimization/
│   │
│   └── end-to-end-workflows/          # 端到端工作流
│       ├── concept-to-production/
│       └── ai-traditional-hybrid/
│
├── papers/                            # 论文阅读
│   ├── README.md
│   ├── computer-vision/
│   │   ├── resnet.md
│   │   ├── stable_diffusion.md
│   │   └── controlnet.md
│   ├── nlp/
│   │   ├── attention_is_all_you_need.md
│   │   └── gpt.md
│   └── reading-notes-template.md
│
├── blog-posts/                        # 技术博客
│   ├── transformer-for-artists.md     # 从艺术家视角理解 Transformer
│   ├── ai-tools-efficiency.md         # AI 工具提升美术效率
│   └── nuke-plugin-ai-integration.md  # NUKE 插件开发
│
└── resources/                         # 学习资源
    ├── books.md                       # 推荐书籍
    ├── courses.md                     # 在线课程
    ├── datasets.md                    # 常用数据集
    └── tools.md                       # 开发工具
```
