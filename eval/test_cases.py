"""Test cases for evaluating analysis quality."""

test_cases = [
    {
        "id": "perfect_match_01",
        "category": "perfect_match",
        "resume_text": """
个人信息：张三 | zhang.san@email.com | 138-0000-0001

个人总结：3年AI算法经验，主要研究方向为自然语言处理和大型语言模型，熟悉深度学习框架。

技能：Python, PyTorch, TensorFlow, LangChain, Transformers, RAG, BERT, GPT系列

工作经验：
公司：字节跳动 AI Lab (2023-2025) | 职位：AI算法工程师
- 设计和实现了基于RAG的问答系统，召回率达到92%
- 使用PyTorch训练和部署了BERT-based文本分类模型
- 使用LangChain构建Agent工作流，实现了多步骤推理

项目经验：
项目：智能客服知识库系统
- 基于LangChain + ChromaDB构建RAG pipeline
- 使用DeepSeek API进行文本生成和推理
- 实现了多Agent协作的工作流编排

教育背景：
学校：北京大学 | 学位：硕士 | 专业：计算机科学与技术
        """,
        "jd_text": """职位：AI算法工程师
我们正在寻找有经验的AI算法工程师加入团队。

岗位职责：
- 设计和优化RAG系统，提升检索和生成质量
- 基于大语言模型构建Agent应用
- 使用LangChain或类似框架实现AI工作流
- 优化模型推理性能和效果

任职要求：
- 精通Python，熟悉PyTorch等深度学习框架
- 3年以上AI相关工作经验
- 深入理解大语言模型原理和应用
- 有RAG系统实际落地经验
- 熟悉LangChain、ChromaDB等技术栈

加分项：
- 有Agent/Multi-Agent系统开发经验
- 熟悉NLP相关技术
        """,
        "expected_score_min": 75,
        "expected_score_max": 100,
        "expected_skills": ["Python", "PyTorch", "LangChain", "RAG"],
        "expected_suggestion_count_min": 1,
    },
    {
        "id": "partial_match_01",
        "category": "partial_match",
        "resume_text": """
个人信息：李四 | li.si@email.com | 139-0000-0002

个人总结：2年后端开发经验，主要在电商领域，熟悉Java生态，对AI有学习意愿。

技能：Java, Spring Boot, MySQL, Redis, Docker, Kafka, Python（基础）

工作经验：
公司：美团 (2022-2024) | 职位：Java后端开发
- 负责订单系统的设计和开发
- 使用Spring Boot构建微服务
- 使用Redis做缓存优化，QPS提升40%
- 维护Kafka消息队列

项目经验：
项目：电商订单管理系统
- 基于Spring Cloud的微服务架构
- 使用Docker进行容器化部署

教育背景：
学校：华中科技大学 | 学位：本科 | 专业：软件工程
        """,
        "jd_text": """职位：AI算法工程师
我们正在寻找有经验的AI算法工程师加入团队。

岗位职责：
- 设计和优化RAG系统，提升检索和生成质量
- 基于大语言模型构建Agent应用
- 优化模型推理性能和效果

任职要求：
- 精通Python，熟悉PyTorch/TensorFlow
- 3年以上AI相关工作经验
- 深入理解大语言模型原理
- 有RAG系统实际落地经验
- 熟悉机器学习经典算法

加分项：
- 有NLP/CV相关经验
- 有论文发表经历
        """,
        "expected_score_min": 15,
        "expected_score_max": 55,
        "expected_skills": ["Python"],
        "expected_suggestion_count_min": 2,
    },
    {
        "id": "no_match_01",
        "category": "no_match",
        "resume_text": """
个人信息：王五 | wang.wu@email.com | 136-0000-0003

个人总结：5年财务会计经验，熟悉企业财务管理和审计流程。

技能：Excel, 用友, SAP, 财务报表分析, 税务筹划

工作经验：
公司：普华永道 (2019-2024) | 职位：高级审计师
- 负责多家上市公司的年度审计
- 编制合并财务报表
- 税务筹划和合规审查

教育背景：
学校：上海财经大学 | 学位：本科 | 专业：会计学
        """,
        "jd_text": """职位：AI算法工程师
我们正在寻找有经验的AI算法工程师加入团队。

岗位职责：
- 设计和优化RAG系统，提升检索和生成质量
- 基于大语言模型构建Agent应用
- 优化模型推理性能和效果

任职要求：
- 精通Python，熟悉PyTorch/TensorFlow
- 3年以上AI相关工作经验
- 深入理解大语言模型原理
- 有RAG系统实际落地经验
- 熟悉机器学习经典算法
        """,
        "expected_score_min": 0,
        "expected_score_max": 25,
        "expected_skills": [],
        "expected_suggestion_count_min": 1,
    },
    {
        "id": "boundary_empty_01",
        "category": "boundary",
        "resume_text": " ",
        "jd_text": "AI工程师，需要精通Python和深度学习",
        "expected_score_min": 0,
        "expected_score_max": 100,
        "expected_skills": [],
        "expected_suggestion_count_min": 0,
    },
    {
        "id": "english_mixed_01",
        "category": "mixed",
        "resume_text": """
Name: Chen Wei | chen.wei@email.com

Summary: 5 years of ML/AI experience at Google and Microsoft.

Skills: Python, TensorFlow, JAX, Kubernetes, MLflow, GCP, BERT, Transformer

Experience:
Google (2021-2024) | ML Engineer
- Built large-scale recommendation systems serving 100M+ users
- Optimized model training pipelines reducing cost by 40%
- Deployed ML models using TFX and Kubernetes

Education:
CMU | Master | Machine Learning
        """,
        "jd_text": """职位：高级机器学习工程师
岗位职责：
- 设计和优化推荐系统
- 构建和部署大规模ML模型
- 优化训练和推理pipeline

任职要求：
- 精通Python和TensorFlow/PyTorch
- 5年以上ML经验
- 熟悉Kubernetes和模型部署
- 有大规模推荐系统经验
        """,
        "expected_score_min": 65,
        "expected_score_max": 100,
        "expected_skills": ["Python", "TensorFlow", "Kubernetes"],
        "expected_suggestion_count_min": 0,
    },
]
