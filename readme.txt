python vector_upload.py -d ./buginfo/
unzip paraphrase-multilingual-MiniLM-L12-v2.zip -d models




python AskRag.py -q "Creating an Index Explicitly"
python AskRag.py -h
pip freeze > requirements.txt
python vector_upload.py  -c ./config.yaml  -d ../file



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



cd /app
python -m venv oracleBUG
source oracleBUG/bin/activate
cd dev


--数据库初始化办法


--mysql 8 环境初始化
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
CREATE USER pgdba WITH SUPERUSER LOGIN PASSWORD 'Lihj@sz2019';



sudo yum install -y mysql-server
sudo systemctl start mysqld
sudo systemctl enable mysqld


CREATE USER 'mysqldba'@'%' IDENTIFIED BY 'Lihj@sz2019';
GRANT ALL PRIVILEGES ON *.* TO 'mysqldba'@'%' WITH GRANT OPTION;
FLUSH PRIVILEGES;


systemctl stop firewalld
systemctl disable firewalld



# 使用清华镜像（没完全搞清楚）
pip3 install sentence-transformers -i https://pypi.tuna.tsinghua.edu.cn/simple






--文件扩容

pvcreate /dev/sdc
vgdisplay
vgextend  rhel /dev/sdc
lvextend -L 50G /dev/mapper/rhel-root
xfs_growfs /dev/mapper/rhel-root







--其他：模型下载方法！


下载模型：
https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/
https://www.modelscope.cn/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2


模型下载学习！
https://zhuanlan.zhihu.com/p/23847461291


# 使用清华镜像
pip3 install sentence-transformers -i https://pypi.tuna.tsinghua.edu.cn/simple

https://hf-mirror.com/sentence-transformers/all-MiniLM-L6-v2/tree/main

git clone https://hf-mirror.com/sentence-transformers/all-MiniLM-L6-v2

from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer('D:/notebook/BERTopic/all-MiniLM-L6-v2')

mkdir -p /data/pretrained_model/all-MiniLM-L6-v2
wget -P /data/pretrained_model/all-MiniLM-L6-v2 https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json
wget -P /data/pretrained_model/all-MiniLM-L6-v2 https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/pytorch_model.bin
wget -P /data/pretrained_model/all-MiniLM-L6-v2 https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/data_config.json
wget -P /data/pretrained_model/all-MiniLM-L6-v2 https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config_sentence_transformers.json
wget -P /data/pretrained_model/all-MiniLM-L6-v2 https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/modules.json
wget -P /data/pretrained_model/all-MiniLM-L6-v2 https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/sentence_bert_config.json
wget -P /data/pretrained_model/all-MiniLM-L6-v2 https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/special_tokens_map.json
wget -P /data/pretrained_model/all-MiniLM-L6-v2 https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json
wget -P /data/pretrained_model/all-MiniLM-L6-v2 https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer_config.json
wget -P /data/pretrained_model/all-MiniLM-L6-v2 https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/train_script.py
wget -P /data/pretrained_model/all-MiniLM-L6-v2 https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt
mkdir /data/pretrained_model/all-MiniLM-L6-v2/1_Pooling
wget -P /data/pretrained_model/all-MiniLM-L6-v2/1_Pooling https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/1_Pooling/config.json
