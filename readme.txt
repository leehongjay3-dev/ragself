测试环境python3.12 postgres 18 + vetor ***************************


--下载脚本
git clone https://github.com/leehongjay3-dev/ragself.git


--下载大模型  https://gitee.com/hf-models/bge-large-zh-v1.5
mkdir models
cd   models

# 从 HF 镜像下载关键rag文件
wget https://hf-mirror.com/BAAI/bge-large-zh-v1.5/resolve/main/pytorch_model.bin
wget https://hf-mirror.com/BAAI/bge-large-zh-v1.5/resolve/main/config.json
wget https://hf-mirror.com/BAAI/bge-large-zh-v1.5/resolve/main/tokenizer.json
wget https://hf-mirror.com/BAAI/bge-large-zh-v1.5/resolve/main/vocab.txt


--上传文件，见demo.log
python vector_upload.py  -c ./config.yaml  -d ./buginfo/

--查询执行,直接调用返回
python AskRag.py -q "ORA-00600: internal error code, arguments: [kghfrempty:ds"

--交互式调用返回
(oracleBUG) [root@redhatos zisk]# python AskRag.py -i
⚠️  无法导入vector_upload模块，使用内置配置: nofile 预期没有

============================================================
🤖 RAG查询系统 - 交互模式
当前模型: glm-4-7-251222
Base URL: https://ark.cn-beijing.volces.com/api/v3
输入问题进行查询，输入 'quit' 或 'exit' 退出
输入 'rag off' 关闭向量检索，输入 'rag on' 开启
输入 'config' 显示当前配置
============================================================


❓ 请输入问题: ORA-00600: internal error code, arguments: [kghfrempty:ds

🔍 正在查询...
2026-03-04 23:12:29,602 - INFO - 📦 加载向量模型: ./models
2026-03-04 23:12:29,604 - INFO - Use pytorch device_name: cpu
2026-03-04 23:12:29,604 - INFO - Load pretrained SentenceTransformer: ./models
2026-03-04 23:12:29,607 - WARNING - No sentence-transformers model found with name ./models. Creating a new one with mean pooling.
Loading weights: 100%|██████████████████████████████████████████| 391/391 [00:00<00:00, 2086.05it/s, Materializing param=pooler.dense.weight]
BertModel LOAD REPORT from: ./models
Key                     | Status     |  |
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  |

Notes:
- UNEXPECTED    :can be ignored when loading from different task/architecture; not ok if you expect identical arch.
2026-03-04 23:12:30,461 - INFO - ✅ RAG系统初始化完成，使用模型: glm-4-7-251222
Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.12s/it]
🔍 向量检索: ORA-00600: internal error code, arguments: [kghfrempty:ds
2026-03-04 23:12:31,625 - INFO - 🔍 向量检索完成，找到 5 个相关文档
2026-03-04 23:12:31,626 - INFO - 📝 构建上下文完成，长度: 2730
2026-03-04 23:13:00,584 - INFO - HTTP Request: POST https://ark.cn-beijing.volces.com/api/v3/chat/completions "HTTP/1.1 200 OK"

============================================================
💡 答案:
根据提供的参考信息，错误代码 `ORA-00600: internal error code, arguments: [kghfrempty:ds]` 对应于 **Bug 37383342**。

以下是该问题的详细说明：

### **Bug 编号：** 37383342

### **影响版本：**
*   **Oracle Database 19c**

### **核心问题描述：**
该 Bug 与 **Data Guard** 和 **Redo Apply** 有关。具体表现为，在 **Active Data Guard** 物理备库上，当主库执行某些特定的 **在线表重定义（Online Table Redefinition, 简称 OTR）** 操作时，备库的 **Redo Apply** 进程（MRP 进程）可能会意外终止。

### **错误详情：**
*   **错误代码：** `ORA-00600: internal error code, arguments: [kghfrempty:ds], [], [], [], [], [], [], [], [], [], [], []`
*   **后果：** 该错误会导致备库停止应用重做日志，与主库的数据同步中断，从而影响高可用性和报表查询能力。

### **来源：**
*   以上信息来源于 **【文档3】bug_37383342.txt**。

**注意：** 参考信息中未提及该 Bug 的具体解决方案或补丁编号，建议查阅 Oracle 官方支持文档获取修复方案。

📚 参考文档:
  1. bug_37389579.txt (相关度: 0.543)
  2. bug_37383977.txt (相关度: 0.539)
  3. bug_37383342.txt (相关度: 0.530)
  4. bug_37393792.txt (相关度: 0.529)
  5. bug_37381614.txt (相关度: 0.521)
============================================================

❓ 请输入问题:

❓ 请输入问题: quit
👋 退出系统
2026-03-04 23:13:29,117 - INFO - RAG系统已关闭
(oracleBUG) [root@redhatos zisk]#



其他：********************************************************
pip freeze > requirements.txt

python 安装办法：
sudo wget https://www.python.org/ftp/python/3.12.3/Python-3.12.3.tgz
ll
tar -xvf Python-3.12.3.tgz
sudo tar xzf Python-3.12.3.tgz
cd Python-3.12.3/
sudo ./configure --enable-optimizations
sudo make altinstall
ll /usr/local/bin/python3.12
ln -s /usr/local/bin/python3.12 /usr/bin/python
ln -s /usr/local/bin/pip3.12 /usr/bin/pip

--设置环境
cd /app
python -m venv oracleBUG
source oracleBUG/bin/activate
cd dev

--数据库初始化办法

--PG 18环境初始化
sudo dnf install -y https://download.postgresql.org/pub/repos/yum/reporpms/EL-8-x86_64/pgdg-redhat-repo-latest.noarch.rpm
sudo dnf -qy module disable postgresql
sudo dnf install -y postgresql18-server postgresql18
yum install pgvector_18
sudo /usr/pgsql-18/bin/postgresql-18-setup initdb
sudo systemctl ennalbe postgresql-18
sudo systemctl start   postgresql-18
echo "host    all             all             0.0.0.0/0               scram-sha-256" >> pg_hba.conf
echo "listen_addresses = '*'"                                                        >> postgresql.conf
CREATE EXTENSION IF NOT EXISTS pgvector;
CREATE USER pgdba WITH SUPERUSER LOGIN PASSWORD 'dbpassw0rd';


sudo yum install -y mysql-server
sudo systemctl start mysqld
sudo systemctl enable mysqld
CREATE USER 'mysqldba'@'%' IDENTIFIED BY 'dbpassw0rd';
GRANT ALL PRIVILEGES ON *.* TO 'mysqldba'@'%' WITH GRANT OPTION;
FLUSH PRIVILEGES;
systemctl stop firewalld
systemctl disable firewalld



--文件扩容

pvcreate /dev/sdc
vgdisplay
vgextend  rhel /dev/sdc
lvextend -L 50G /dev/mapper/rhel-root
xfs_growfs /dev/mapper/rhel-root


--其他：模型下载方法！
https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/
https://www.modelscope.cn/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
