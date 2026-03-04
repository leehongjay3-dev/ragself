#!/usr/bin/env python3
"""
RAG查询系统 - 结合向量数据库和大模型
功能：先从向量库检索相关信息，再结合问题调用大模型生成答案
优化：从配置文件读取API配置，支持多种模型切换
"""

import os
import sys
import json
import argparse
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# OpenAI客户端
from openai import OpenAI

# 数据库和向量处理
import psycopg2
from psycopg2 import extras
from sentence_transformers import SentenceTransformer

# 尝试导入配置模块（从vector_upload.py）
try:
    from nofile import Config, DatabaseManager, TextCleaner
except ImportError:
    # 如果导入失败，定义必要的类
    print("⚠️  无法导入vector_upload模块，使用内置配置: nofile 预期没有")
    
    import yaml
    
    class Config:
        """内置配置类"""
        def __init__(self, config_path: str = "config.yaml"):
            self.config_path = config_path
            self.config = self._load_config()
            self.model_path = self.config['model']['path']
            self.vector_dim = self.config['model']['vector_dim']
            
        def _load_config(self) -> Dict:
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
    
    class DatabaseManager:
        """内置数据库管理类"""
        def __init__(self, db_config: Dict):
            self.db_config = db_config
            self.conn = None
        
        def connect(self):
            try:
                self.conn = psycopg2.connect(**self.db_config)
                return True
            except Exception as e:
                print(f"❌ 数据库连接失败: {e}")
                return False
        
        def close(self):
            if self.conn:
                self.conn.close()
    
    class TextCleaner:
        """内置文本清理类"""
        @staticmethod
        def clean_text(text: str) -> str:
            if not text:
                return ""
            # 移除控制字符
            text = ''.join(char for char in text if char >= ' ' or char in '\n\r\t')
            return text.strip()


# ------------------- RAG查询系统 -------------------
class RAGQuerySystem:
    """RAG查询系统：向量检索 + 大模型生成"""
    
    def __init__(self, config_path: str = "config.yaml", api_key: str = None):
        # 初始化配置
        self.config_path = config_path
        self.config = self._load_config()
        
        # 从配置文件获取API配置
        self._init_openai_config(api_key)
        
        # 初始化模型和数据库
        self.model = None
        self.db_manager = None
        self.is_initialized = False
        
        # 日志设置
        self._setup_logging()
    
    def _init_openai_config(self, api_key: str = None):
        """初始化OpenAI配置，从配置文件读取参数"""
        # 从配置文件获取OpenAI配置
        openai_config = self.config.get('openai', {})
        
        # 获取base_url，优先使用配置文件，其次使用默认值
        self.base_url = openai_config.get('base_url', "https://ark.cn-beijing.volces.com/api/v3")
        
        # 获取模型名称，优先使用配置文件，其次使用默认值
        self.model_name = openai_config.get('model', 'deepseek-v3-2-251201')
        
        # 获取API密钥，优先级：命令行参数 > 环境变量 > 配置文件
        self.api_key = api_key or os.getenv("ARK_API_KEY") or openai_config.get('api_key')
        
        if not self.api_key:
            raise ValueError("未配置API密钥，请在配置文件中设置openai.api_key或通过环境变量ARK_API_KEY提供")
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
    
    def _load_config(self) -> Dict:
        """加载配置"""
        try:
            import yaml
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"⚠️  配置加载失败，使用默认配置: {e}")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'user': 'postgres',
                'password': 'password',
                'dbname': 'vectordb'
            },
            'model': {
                'path': '/app/mu',
                'vector_dim': 384
            },
            'rag': {
                'top_k': 5,
                'max_context_length': 4000,
                'similarity_threshold': 0.7,
                'max_history': 3
            },
            'openai': {
                'base_url': "https://ark.cn-beijing.volces.com/api/v3",
                'model': 'deepseek-v3-2-251201',
                'api_key': '',
                'temperature': 0.7,
                'max_tokens': 2000
            }
        }
    
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> bool:
        """初始化系统"""
        try:
            # 加载向量模型
            self.logger.info(f"📦 加载向量模型: {self.config['model']['path']}")
            self.model = SentenceTransformer(self.config['model']['path'])
            
            # 连接数据库
            db_config = self.config['database']
            self.db_manager = DatabaseManager(db_config)
            
            if not self.db_manager.connect():
                self.logger.warning("⚠️  数据库连接失败，将仅使用大模型模式")
                self.db_manager = None
            
            self.is_initialized = True
            self.logger.info(f"✅ RAG系统初始化完成，使用模型: {self.model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 初始化失败: {e}")
            return False
    
    def query_vector_db(self, question: str, top_k: int = None) -> List[Dict]:
        """查询向量数据库"""
        if not self.db_manager:
            return []
        
        try:
            # 生成问题向量
            question_vector = self.model.encode(question, convert_to_numpy=True).tolist()
            print(f"🔍 向量检索: {question}")
            
            # 查询SQL
            rag_config = self.config.get('rag', {})
            k = top_k or rag_config.get('top_k', 5)
            
            query_sql = """
                SELECT 
                    content, 
                    file_name, 
                    file_type,
                    chunk_id,
                    page_range,
                    line_range,
                    metadata,
                    1 - (vector <=> %s::vector) as similarity
                FROM file_vectors
                WHERE 1 - (vector <=> %s::vector) > %s
                ORDER BY vector <=> %s::vector
                LIMIT %s;
            """
            
            similarity_threshold = rag_config.get('similarity_threshold', 0.6)
            
            cur = self.db_manager.conn.cursor()
            cur.execute(query_sql, (
                question_vector, 
                question_vector, 
                similarity_threshold,
                question_vector,
                k
            ))
            
            results = cur.fetchall()
            cur.close()
            
            # 格式化结果
            documents = []
            for row in results:
                doc = {
                    'content': row[0],
                    'file_name': row[1],
                    'file_type': row[2],
                    'chunk_id': row[3],
                    'page_range': row[4],
                    'line_range': row[5],
                    'metadata': row[6] if isinstance(row[6], dict) else json.loads(row[6]) if row[6] else {},
                    'similarity': float(row[7])
                }
                documents.append(doc)
            
            self.logger.info(f"🔍 向量检索完成，找到 {len(documents)} 个相关文档")
            return documents
            
        except Exception as e:
            self.logger.error(f"❌ 向量查询失败: {e}")
            return []
    
    def build_context(self, question: str, documents: List[Dict]) -> str:
        """构建上下文"""
        if not documents:
            return ""
        
        context_parts = []
        current_length = 0
        max_context = self.config.get('rag', {}).get('max_context_length', 4000)
        
        for i, doc in enumerate(documents, 1):
            # 构建文档信息
            doc_info = f"【文档{i}】来源: {doc['file_name']}"
            if doc['page_range']:
                doc_info += f" (页码: {doc['page_range']})"
            elif doc['line_range']:
                doc_info += f" (行号: {doc['line_range']})"
            
            doc_info += f"\n相关度: {doc['similarity']:.3f}\n"
            doc_info += f"内容: {doc['content'][:500]}...\n\n"
            
            # 检查长度限制
            if current_length + len(doc_info) > max_context:
                break
            
            context_parts.append(doc_info)
            current_length += len(doc_info)
        
        context = "".join(context_parts)
        self.logger.info(f"📝 构建上下文完成，长度: {len(context)}")
        return context
    
    def generate_answer(self, question: str, context: str, history: List[Dict] = None) -> str:
        """生成答案"""
        try:
            # 构建提示词
            if context:
                prompt = f"""你是专业的数据库和操作系统专家。请根据以下参考信息回答问题。

参考信息：
{context}

用户问题: {question}

请基于参考信息回答问题，如果参考信息不足，请明确说明。回答要准确、详细，并引用相关文档来源。"""
            else:
                prompt = f"你是专业的数据库和操作系统专家。请回答以下问题：\n\n{question}"
            
            # 构建消息列表
            messages = [
                {"role": "system", "content": "你是人工智能助手,精通数据库操作系统"}
            ]
            
            # 添加历史对话
            if history:
                messages.extend(history[-self.config.get('rag', {}).get('max_history', 3):])
            
            messages.append({"role": "user", "content": prompt})
            
            # 调用大模型
            openai_config = self.config.get('openai', {})
            completion = self.client.chat.completions.create(
                model=openai_config.get('model', self.model_name),
                messages=messages,
                temperature=openai_config.get('temperature', 0.5),
                max_tokens=openai_config.get('max_tokens', 2000)
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"❌ 答案生成失败: {e}")
            return f"抱歉，生成答案时出现错误: {str(e)}"
    
    def query(self, question: str, use_rag: bool = True) -> Dict:
        """执行查询"""
        if not self.is_initialized:
            if not self.initialize():
                return {
                    'answer': '系统初始化失败',
                    'sources': [],
                    'success': False
                }
        
        try:
            documents = []
            context = ""
            
            # 使用RAG模式
            if use_rag and self.db_manager:
                # 1. 查询向量数据库
                documents = self.query_vector_db(question)
                
                # 2. 构建上下文
                context = self.build_context(question, documents)
            
            # 3. 生成答案
            answer = self.generate_answer(question, context)
            
            return {
                'answer': answer,
                'sources': documents,
                'context_used': bool(context),
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"❌ 查询失败: {e}")
            return {
                'answer': f"查询过程中出现错误: {str(e)}",
                'sources': [],
                'success': False
            }
    
    def interactive_mode(self):
        """交互模式"""
        print("\n" + "="*60)
        print("🤖 RAG查询系统 - 交互模式")
        print(f"当前模型: {self.model_name}")
        print(f"Base URL: {self.base_url}")
        print("输入问题进行查询，输入 'quit' 或 'exit' 退出")
        print("输入 'rag off' 关闭向量检索，输入 'rag on' 开启")
        print("输入 'config' 显示当前配置")
        print("="*60 + "\n")
        
        use_rag = True
        
        while True:
            try:
                question = input("\n❓ 请输入问题: ").strip()
                
                if not question:
                    continue
                
                # 处理特殊命令
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 退出系统")
                    break
                
                if question.lower() == 'rag off':
                    use_rag = False
                    print("🔄 已关闭向量检索模式")
                    continue
                
                if question.lower() == 'rag on':
                    use_rag = True
                    print("🔄 已开启向量检索模式")
                    continue
                
                if question.lower() == 'config':
                    self._show_config()
                    continue
                
                # 执行查询
                print("\n🔍 正在查询...")
                result = self.query(question, use_rag=use_rag)
                
                # 显示结果
                print("\n" + "="*60)
                print("💡 答案:")
                print(result['answer'])
                
                if result['sources']:
                    print("\n📚 参考文档:")
                    for i, doc in enumerate(result['sources'], 1):
                        print(f"  {i}. {doc['file_name']} (相关度: {doc['similarity']:.3f})")
                        if doc['page_range']:
                            print(f"     页码: {doc['page_range']}")
                
                print("="*60)
                
            except KeyboardInterrupt:
                print("\n\n👋 用户中断，退出系统")
                break
            except Exception as e:
                print(f"\n❌ 错误: {e}")
    
    def _show_config(self):
        """显示当前配置"""
        print("\n📋 当前配置:")
        print(f"  模型名称: {self.model_name}")
        print(f"  Base URL: {self.base_url}")
        print(f"  API密钥: {'***' + self.api_key[-4:] if self.api_key else '未设置'}")
        print(f"  向量模型路径: {self.config['model']['path']}")
        print(f"  数据库: {self.config['database']['host']}:{self.config['database']['port']}")
        print(f"  RAG配置: top_k={self.config['rag']['top_k']}, 阈值={self.config['rag']['similarity_threshold']}")
    
    def close(self):
        """关闭系统"""
        if self.db_manager:
            self.db_manager.close()
        self.logger.info("RAG系统已关闭")


# ------------------- 命令行接口 -------------------
def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RAG查询系统')
    parser.add_argument('-q', '--question', type=str, help='查询问题')
    parser.add_argument('-c', '--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--no-rag', action='store_true', help='禁用向量检索')
    parser.add_argument('-i', '--interactive', action='store_true', help='交互模式')
    parser.add_argument('--api-key', type=str, default='', help='API密钥（覆盖配置文件）')
    parser.add_argument('--base-url', type=str, help='API Base URL（覆盖配置文件）')
    parser.add_argument('--model', type=str, help='模型名称（覆盖配置文件）')
    parser.add_argument('--top-k', type=int, default=5, help='检索文档数量')
    parser.add_argument('--max-context', type=int, default=4000, help='最大上下文长度')
    
    args = parser.parse_args()
    
    # 创建RAG系统
    rag_system = RAGQuerySystem(
        config_path=args.config,
        api_key=args.api_key
    )
    
    # 覆盖配置（命令行参数优先级最高）
    if args.base_url:
        rag_system.base_url = args.base_url
        rag_system.client = OpenAI(
            base_url=args.base_url,
            api_key=rag_system.api_key
        )
        rag_system.logger.info(f"已更新Base URL: {args.base_url}")
    
    if args.model:
        rag_system.model_name = args.model
        rag_system.logger.info(f"已更新模型: {args.model}")
    
    try:
        if args.interactive:
            # 交互模式
            rag_system.interactive_mode()
        elif args.question:
            # 单次查询
            result = rag_system.query(
                args.question, 
                use_rag=not args.no_rag
            )
            
            print("\n" + "="*100)
            print("💡 问题:")
            print(args.question)
            print()
            print("💡 答案:")
            print(result['answer'])
            
            if result['sources']:
                print("\n📚 参考文档:")
                for i, doc in enumerate(result['sources'], 1):
                    print(f"  {i}. {doc['file_name']} (相关度: {doc['similarity']:.3f})")
            
            print("="*60)
        else:
            # 默认交互模式
            rag_system.interactive_mode()
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)
    finally:
        rag_system.close()


if __name__ == '__main__':
    main()

