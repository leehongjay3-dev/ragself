#!/usr/bin/env python3
"""
统一文件向量化存储工具（支持PDF、TXT等多种格式）
功能：读取配置，自动识别文件类型，智能分片，生成向量并存储到PostgreSQL
依赖：pymupdf, psycopg2, sentence-transformers, pgvector, pyyaml
"""

import os
import sys
import json
import re
import logging
import hashlib
from datetime import datetime
from typing import Optional, Dict, Tuple, List, Any
from pathlib import Path

import yaml
import psycopg2
from psycopg2 import extras
from sentence_transformers import SentenceTransformer

# 尝试导入pymupdf（用于PDF处理）
try:
    import fitz  # pymupdf
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logging.warning("⚠️  pymupdf未安装，PDF处理功能不可用")


# ------------------- 配置管理类 -------------------
class Config:
    """配置管理类，从YAML文件加载配置"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        
        # 数据库配置
        self.db_host = self.config['database']['host']
        self.db_port = self.config['database']['port']
        self.db_user = self.config['database']['user']
        self.db_password = self.config['database']['password']
        self.db_name = self.config['database']['dbname']
        
        # 模型配置
        self.model_path = self.config['model']['path']
        self.vector_dim = self.config['model']['vector_dim']
        
        # 文件处理配置
        self.target_dir = self.config['processing']['target_dir']
        self.max_file_size = self.config['processing']['max_file_size_mb'] * 1024 * 1024
        
        # 分片配置
        self.chunk_size = self.config['chunking']['chunk_size']
        self.chunk_overlap = self.config['chunking']['chunk_overlap']
        self.min_chunk_size = self.config['chunking']['min_chunk_size']
        self.max_text_for_vector = self.config['chunking']['max_text_for_vector']
        self.txt_lines_per_chunk = self.config['chunking']['txt_lines_per_chunk']
        self.txt_overlap_lines = self.config['chunking']['txt_overlap_lines']
        
        # 支持的文件类型
        self.supported_extensions = []
        if self.config['processing']['supported_types']['pdf']['enabled']:
            self.supported_extensions.extend(
                self.config['processing']['supported_types']['pdf']['extensions']
            )
        if self.config['processing']['supported_types']['txt']['enabled']:
            self.supported_extensions.extend(
                self.config['processing']['supported_types']['txt']['extensions']
            )
        
        # TXT编码支持
        self.txt_encodings = self.config['processing']['supported_types']['txt']['encodings']
    
    def _load_config(self) -> Dict:
        """加载YAML配置文件"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get_db_config(self) -> Dict:
        """获取数据库连接配置"""
        return {
            'host': self.db_host,
            'port': self.db_port,
            'user': self.db_user,
            'password': self.db_password,
            'dbname': self.db_name
        }
    
    def setup_logging(self):
        """配置日志系统"""
        log_config = self.config['logging']
        
        logging.basicConfig(
            level=getattr(logging, log_config['level']),
            format=log_config['format'],
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_config['file'], encoding='utf-8')
            ]
        )
        
        return logging.getLogger(__name__)


# ------------------- 文本清理工具 -------------------
class TextCleaner:
    """文本清理工具类"""
    
    # PostgreSQL TEXT类型不允许的控制字符
    INVALID_CHARS = [
        '\x00', '\x01', '\x02', '\x03', '\x04', '\x05', '\x06', '\x07',
        '\x08', '\x0B', '\x0C', '\x0E', '\x0F', '\x10', '\x11', '\x12', 
        '\x13', '\x14', '\x15', '\x16', '\x17', '\x18', '\x19', '\x1A', 
        '\x1B', '\x1C', '\x1D', '\x1E', '\x1F'
    ]
    
    @classmethod
    def clean_text(cls, text: str, aggressive: bool = False) -> str:
        """清理文本中的无效字符"""
        if not text:
            return ""
        
        # 移除NUL字符
        text = text.replace('\x00', '')
        
        if aggressive:
            text = ''.join(char for char in text if char >= ' ' or char in '\n\r\t')
        else:
            for char in cls.INVALID_CHARS:
                text = text.replace(char, '')
        
        # 清理多余空白
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        return text.strip()
    
    @classmethod
    def is_valid_text(cls, text: str) -> bool:
        """检查文本是否有效"""
        for char in cls.INVALID_CHARS:
            if char in text:
                return False
        return True


# ------------------- 文本分片器 -------------------
class TextChunker:
    """文本分片器"""
    
    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int,
        overlap: int,
        min_chunk_size: int
    ) -> List[Dict]:
        """将长文本分片"""
        if not text:
            return []
        
        chunks = []
        text_length = len(text)
        start = 0
        chunk_id = 0
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            
            # 尊重段落边界
            if end < text_length:
                max_extend = int(chunk_size * 0.2)
                search_end = min(end + max_extend, text_length)
                
                for i in range(end, search_end):
                    if text[i:i+2] == '\n\n' or (i > 0 and text[i] == '\n' and text[i-1] in '。！？.!?'):
                        end = i + 1
                        break
            
            chunk_text = text[start:end].strip()
            
            if chunk_text and len(chunk_text) >= min_chunk_size:
                chunks.append({
                    'text': chunk_text,
                    'start': start,
                    'end': end,
                    'chunk_id': chunk_id,
                    'length': len(chunk_text)
                })
                chunk_id += 1
            
            start = end - overlap if end < text_length else end
            
            if chunks and start <= chunks[-1]['start']:
                start = end
        
        return chunks
    
    @staticmethod
    def chunk_by_lines(
        text: str,
        lines_per_chunk: int,
        overlap_lines: int
    ) -> List[Dict]:
        """按行分片"""
        lines = text.split('\n')
        if not lines:
            return []
        
        chunks = []
        chunk_id = 0
        start_line = 0
        
        while start_line < len(lines):
            end_line = min(start_line + lines_per_chunk, len(lines))
            chunk_lines = lines[start_line:end_line]
            chunk_text = '\n'.join(chunk_lines)
            
            if chunk_text.strip():
                chunks.append({
                    'text': chunk_text,
                    'start': start_line,
                    'end': end_line,
                    'chunk_id': chunk_id,
                    'length': len(chunk_text),
                    'line_range': f"{start_line+1}-{end_line}"
                })
                chunk_id += 1
            
            start_line = end_line - overlap_lines if end_line < len(lines) else end_line
        
        return chunks


# ------------------- 数据库管理类 -------------------
class DatabaseManager:
    """数据库连接和操作管理"""
    
    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.conn = None
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def connect(self):
        """建立数据库连接"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            logger.info("✅ 数据库连接成功")
        except Exception as e:
            logger.error(f"❌ 数据库连接失败: {e}")
            raise
    
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            logger.info("数据库连接已关闭")
    
    def init_vector_table(self):
        """创建向量存储表（支持PDF和TXT）"""
        create_table_sql = """
            -- 安装pgvector扩展
            CREATE EXTENSION IF NOT EXISTS vector;
            
            -- 创建表（如果不存在）
            CREATE TABLE IF NOT EXISTS file_vectors (
                id SERIAL PRIMARY KEY,
                file_hash VARCHAR(64) NOT NULL,
                file_name VARCHAR(255) NOT NULL,
                file_type VARCHAR(50) NOT NULL,
                chunk_id INTEGER NOT NULL,
                total_chunks INTEGER NOT NULL,
                content TEXT,
                vector vector(384),
                create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_path VARCHAR(512),
                page_count INTEGER,
                file_size BIGINT,
                chunk_size INTEGER,
                page_range VARCHAR(100),
                line_range VARCHAR(100),
                metadata JSONB,
                CONSTRAINT unique_file_chunk UNIQUE (file_hash, chunk_id)
            );
            
            -- 创建索引
            CREATE INDEX IF NOT EXISTS idx_file_vectors_file_hash ON file_vectors(file_hash);
            CREATE INDEX IF NOT EXISTS idx_file_vectors_file_name ON file_vectors(file_name);
            CREATE INDEX IF NOT EXISTS idx_file_vectors_file_type ON file_vectors(file_type);
            CREATE INDEX IF NOT EXISTS idx_file_vectors_create_time ON file_vectors(create_time);
            
            -- 创建向量索引
            CREATE INDEX IF NOT EXISTS idx_file_vectors_vector 
            ON file_vectors USING ivfflat (vector vector_cosine_ops)
            WITH (lists = 100);
        """
        
        try:
            cur = self.conn.cursor()
            cur.execute(create_table_sql)
            self.conn.commit()
            logger.info("✅ 向量表初始化成功（支持PDF和TXT）")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"❌ 向量表创建失败: {e}")
            raise
        finally:
            cur.close()
    
    def insert_chunks_batch(self, chunks_data: List[Dict]):
        """批量插入分片数据"""
        insert_sql = """
            INSERT INTO file_vectors 
            (file_hash, file_name, file_type, chunk_id, total_chunks, content, vector, 
             file_path, page_count, file_size, chunk_size, page_range, line_range, metadata)
            VALUES %s
            ON CONFLICT (file_hash, chunk_id) 
            DO UPDATE SET
                content = EXCLUDED.content,
                vector = EXCLUDED.vector,
                file_path = EXCLUDED.file_path,
                page_count = EXCLUDED.page_count,
                file_size = EXCLUDED.file_size,
                chunk_size = EXCLUDED.chunk_size,
                page_range = EXCLUDED.page_range,
                line_range = EXCLUDED.line_range,
                metadata = EXCLUDED.metadata,
                create_time = CURRENT_TIMESTAMP
        """
        
        try:
            cur = self.conn.cursor()
            data = [
                (
                    item['file_hash'],
                    item['file_name'],
                    item['file_type'],
                    item['chunk_id'],
                    item['total_chunks'],
                    item['content'],
                    item['vector'],
                    item['file_path'],
                    item.get('page_count'),
                    item['file_size'],
                    item['chunk_size'],
                    item.get('page_range', ''),
                    item.get('line_range', ''),
                    json.dumps(item['metadata'])
                )
                for item in chunks_data
            ]
            extras.execute_values(cur, insert_sql, data)
            self.conn.commit()
            logger.info(f"✅ 成功插入 {len(chunks_data)} 个分片")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"❌ 批量插入失败: {e}")
            raise
        finally:
            cur.close()
    
    def delete_file_chunks(self, file_hash: str):
        """删除指定文件的所有分片"""
        delete_sql = "DELETE FROM file_vectors WHERE file_hash = %s"
        try:
            cur = self.conn.cursor()
            cur.execute(delete_sql, (file_hash,))
            deleted_count = cur.rowcount
            self.conn.commit()
            if deleted_count > 0:
                logger.info(f"🗑️  删除旧数据: {deleted_count} 个分片")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"❌ 删除失败: {e}")
            raise
        finally:
            cur.close()


# ------------------- 文件处理器基类 -------------------
class FileProcessor:
    """文件处理器基类"""
    
    def __init__(self, model: SentenceTransformer, config: Config):
        self.model = model
        self.config = config
    
    @staticmethod
    def calculate_file_hash(file_path: str) -> str:
        """计算文件MD5哈希值"""
        hash_md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def text_to_vector(self, text: str) -> List[float]:
        """将文本转为向量"""
        if len(text) > self.config.max_text_for_vector:
            text = text[:self.config.max_text_for_vector]
        
        vector = self.model.encode(text, convert_to_numpy=True).tolist()
        return vector


# ------------------- PDF处理器 -------------------
class PDFProcessor(FileProcessor):
    """PDF文件处理器"""
    
    def process_pdf(self, file_path: str) -> Tuple[str, int, List[Dict]]:
        """处理PDF文件并分片"""
        if not PDF_SUPPORT:
            raise ImportError("pymupdf未安装，无法处理PDF文件")
        
        logger.info(f"📄 开始处理PDF: {os.path.basename(file_path)}")
        
        # 计算文件哈希
        file_hash = self.calculate_file_hash(file_path)
        logger.info(f"🔑 文件哈希: {file_hash}")
        
        # 获取文件大小
        file_size = os.path.getsize(file_path)
        
        if file_size > self.config.max_file_size:
            raise ValueError(f"文件过大: {file_size} bytes")
        
        # 提取PDF内容
        doc = None
        try:
            doc = fitz.open(file_path)
            page_count = len(doc)
            
            # 提取元数据
            metadata = {
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'creator': doc.metadata.get('creator', ''),
                'producer': doc.metadata.get('producer', ''),
                'creation_date': doc.metadata.get('creationDate', ''),
                'modification_date': doc.metadata.get('modDate', ''),
                'format': 'PDF'
            }
            
            # 提取所有页面的文本
            pages_text = []
            all_text_parts = []
            
            for page_num, page in enumerate(doc, 1):
                try:
                    text = page.get_text()
                    if text and text.strip():
                        text = TextCleaner.clean_text(text, aggressive=False)
                        
                        if text:
                            pages_text.append((page_num, text))
                            all_text_parts.append(f"[第{page_num}页]\n{text}")
                except Exception as e:
                    logger.warning(f"⚠️  第{page_num}页提取失败: {e}")
            
            full_text = '\n\n'.join(all_text_parts)
            full_text = TextCleaner.clean_text(full_text, aggressive=False)
            
            logger.info(f"📊 提取完成: {page_count}页, {len(full_text)}字符, {file_size}字节")
            
        finally:
            if doc:
                doc.close()
        
        # 分片处理
        if pages_text:
            logger.info("📦 使用页面分片策略")
            chunks = self._chunk_by_pages(pages_text)
        else:
            logger.info("📦 使用文本分片策略")
            chunks = TextChunker.chunk_text(
                full_text,
                self.config.chunk_size,
                self.config.chunk_overlap,
                self.config.min_chunk_size
            )
        
        # 为每个分片生成向量
        chunks_data = []
        for chunk in chunks:
            vector = self.text_to_vector(chunk['text'])
            
            chunk_data = {
                'file_hash': file_hash,
                'file_name': os.path.basename(file_path),
                'file_type': 'pdf',
                'chunk_id': chunk['chunk_id'],
                'total_chunks': len(chunks),
                'content': chunk['text'],
                'vector': vector,
                'file_path': os.path.abspath(file_path),
                'page_count': page_count,
                'file_size': file_size,
                'chunk_size': chunk['length'],
                'page_range': chunk.get('page_range', ''),
                'line_range': '',
                'metadata': {
                    **metadata,
                    'text_start': chunk.get('start', 0),
                    'text_end': chunk.get('end', 0)
                }
            }
            chunks_data.append(chunk_data)
        
        logger.info(f"✅ 分片处理完成: {len(chunks_data)} 个分片")
        return file_hash, len(chunks_data), chunks_data
    
    def _chunk_by_pages(self, pages_text: List[Tuple[int, str]]) -> List[Dict]:
        """按页面分片"""
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_id = 0
        
        for page_num, page_text in pages_text:
            if not page_text or not page_text.strip():
                continue
            
            page_size = len(page_text)
            
            # 单页超过chunk_size，需要拆分
            if page_size > self.config.chunk_size:
                # 先保存当前累积的分片
                if current_chunk:
                    chunks.append({
                        'text': '\n\n'.join([t for _, t in current_chunk]),
                        'pages': [p for p, _ in current_chunk],
                        'chunk_id': chunk_id,
                        'length': current_size
                    })
                    chunk_id += 1
                    current_chunk = []
                    current_size = 0
                
                # 拆分大页面
                page_chunks = TextChunker.chunk_text(
                    page_text,
                    self.config.chunk_size,
                    overlap=0,
                    min_chunk_size=self.config.min_chunk_size
                )
                for pc in page_chunks:
                    pc['pages'] = [page_num]
                    pc['chunk_id'] = chunk_id
                    chunks.append(pc)
                    chunk_id += 1
            else:
                # 判断是否需要创建新分片
                if current_size + page_size > self.config.chunk_size and current_chunk:
                    chunks.append({
                        'text': '\n\n'.join([t for _, t in current_chunk]),
                        'pages': [p for p, _ in current_chunk],
                        'chunk_id': chunk_id,
                        'length': current_size
                    })
                    chunk_id += 1
                    current_chunk = []
                    current_size = 0
                
                current_chunk.append((page_num, page_text))
                current_size += page_size
        
        # 保存最后一个分片
        if current_chunk:
            chunks.append({
                'text': '\n\n'.join([t for _, t in current_chunk]),
                'pages': [p for p, _ in current_chunk],
                'chunk_id': chunk_id,
                'length': current_size
            })
        
        # 格式化页面范围
        for chunk in chunks:
            if 'pages' in chunk:
                chunk['page_range'] = self._format_page_range(chunk['pages'])
            else:
                chunk['page_range'] = ''
        
        logger.info(f"📦 页面分片完成: {len(pages_text)} 页 -> {len(chunks)} 个分片")
        return chunks
    
    @staticmethod
    def _format_page_range(pages: List[int]) -> str:
        """格式化页面范围"""
        if not pages:
            return ""
        
        pages = sorted(set(pages))
        if len(pages) == 1:
            return str(pages[0])
        
        ranges = []
        start = pages[0]
        end = pages[0]
        
        for page in pages[1:]:
            if page == end + 1:
                end = page
            else:
                ranges.append(f"{start}-{end}" if start != end else str(start))
                start = page
                end = page
        
        ranges.append(f"{start}-{end}" if start != end else str(start))
        return ",".join(ranges)


# ------------------- TXT处理器 -------------------
class TXTProcessor(FileProcessor):
    """TXT文件处理器"""
    
    def process_txt(self, file_path: str) -> Tuple[str, int, List[Dict]]:
        """处理TXT文件并分片"""
        logger.info(f"📄 开始处理TXT: {os.path.basename(file_path)}")
        
        # 计算文件哈希
        file_hash = self.calculate_file_hash(file_path)
        logger.info(f"🔑 文件哈希: {file_hash}")
        
        # 获取文件大小
        file_size = os.path.getsize(file_path)
        
        if file_size > self.config.max_file_size:
            raise ValueError(f"文件过大: {file_size} bytes")
        
        # 读取文件内容
        content = self._read_file_with_encoding(file_path)
        
        # 清理文本
        content = TextCleaner.clean_text(content, aggressive=False)
        
        if not content:
            raise ValueError("文件内容为空")
        
        logger.info(f"📊 文件大小: {file_size} 字节, 内容长度: {len(content)} 字符")
        
        # 判断分片策略
        file_ext = os.path.splitext(file_path)[1].lower()
        is_code_file = file_ext in ['.py', '.js', '.java', '.c', '.cpp', '.log', '.sh', '.sql']
        
        if len(content) <= self.config.chunk_size:
            # 小文件不分片
            logger.info("📦 文件较小，不分片")
            chunks = [{
                'text': content,
                'start': 0,
                'end': len(content),
                'chunk_id': 0,
                'length': len(content)
            }]
        elif is_code_file or '\n' in content[:1000]:
            # 按行分片
            logger.info("📦 使用按行分片策略")
            chunks = TextChunker.chunk_by_lines(
                content,
                self.config.txt_lines_per_chunk,
                self.config.txt_overlap_lines
            )
        else:
            # 按文本分片
            logger.info("📦 使用文本分片策略")
            chunks = TextChunker.chunk_text(
                content,
                self.config.chunk_size,
                self.config.chunk_overlap,
                self.config.min_chunk_size
            )
        
        # 为每个分片生成向量
        chunks_data = []
        for chunk in chunks:
            vector = self.text_to_vector(chunk['text'])
            
            chunk_data = {
                'file_hash': file_hash,
                'file_name': os.path.basename(file_path),
                'file_type': 'txt',
                'chunk_id': chunk['chunk_id'],
                'total_chunks': len(chunks),
                'content': chunk['text'],
                'vector': vector,
                'file_path': os.path.abspath(file_path),
                'page_count': None,
                'file_size': file_size,
                'chunk_size': chunk['length'],
                'page_range': '',
                'line_range': chunk.get('line_range', ''),
                'metadata': {
                    'text_start': chunk.get('start', 0),
                    'text_end': chunk.get('end', 0),
                    'extension': file_ext,
                    'encoding': 'utf-8'
                }
            }
            chunks_data.append(chunk_data)
        
        logger.info(f"✅ 分片处理完成: {len(chunks_data)} 个分片")
        return file_hash, len(chunks_data), chunks_data
    
    def _read_file_with_encoding(self, file_path: str) -> str:
        """尝试多种编码读取文件"""
        for encoding in self.config.txt_encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                logger.debug(f"使用编码 {encoding} 成功读取文件")
                return content
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        raise ValueError(f"无法识别文件编码: {file_path}")


# ------------------- 主处理器 -------------------
class VectorUploader:
    """统一向量化上传主处理器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = Config(config_path)
        self.logger = self.config.setup_logging()
        self.model = None
        self.db_manager = None
        self.pdf_processor = None
        self.txt_processor = None
    
    def initialize(self):
        """初始化：加载模型、连接数据库、创建表"""
        global logger
        logger = self.logger
        
        logger.info("="*60)
        logger.info("开始初始化...")
        
        # 加载模型
        logger.info(f"📦 加载本地模型: {self.config.model_path}")
        try:
            self.model = SentenceTransformer(self.config.model_path)
            logger.info("✅ 模型加载成功")
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            raise
        
        # 连接数据库
        self.db_manager = DatabaseManager(self.config.get_db_config())
        self.db_manager.connect()
        
        # 初始化表
        self.db_manager.init_vector_table()
        
        # 创建处理器
        self.pdf_processor = PDFProcessor(self.model, self.config)
        self.txt_processor = TXTProcessor(self.model, self.config)
        
        logger.info("="*60)
    
    def get_files(self, directory: str) -> List[str]:
        """获取目录下所有支持的文件"""
        files = []
        target_dir = Path(directory)
        
        if not target_dir.exists():
            logger.error(f"目录不存在: {directory}")
            return files
        
        for ext in self.config.supported_extensions:
            for file_path in target_dir.rglob(f'*{ext}'):
                if file_path.is_file():
                    files.append(str(file_path))
        
        logger.info(f"📂 发现 {len(files)} 个文件")
        return sorted(files)
    
    def process_single_file(self, file_path: str) -> bool:
        """处理单个文件"""
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"处理文件: {os.path.basename(file_path)}")
            
            # 判断文件类型
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.pdf':
                file_hash, total_chunks, chunks_data = self.pdf_processor.process_pdf(file_path)
            else:
                file_hash, total_chunks, chunks_data = self.txt_processor.process_txt(file_path)
            
            # 删除旧数据
            self.db_manager.delete_file_chunks(file_hash)
            
            # 批量插入分片
            self.db_manager.insert_chunks_batch(chunks_data)
            
            logger.info(f"✅ 文件处理完成: {total_chunks} 个分片已存储")
            return True
            
        except Exception as e:
            logger.error(f"❌ 处理失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    def process_batch(self, file_paths: List[str]):
        """批量处理文件"""
        total = len(file_paths)
        success_count = 0
        failed_count = 0
        
        logger.info(f"\n开始批量处理 {total} 个文件...")
        logger.info(f"分片配置: 大小={self.config.chunk_size}, 重叠={self.config.chunk_overlap}")
        
        for idx, file_path in enumerate(file_paths, 1):
            logger.info(f"\n[{idx}/{total}] 处理: {os.path.basename(file_path)}")
            
            if self.process_single_file(file_path):
                success_count += 1
            else:
                failed_count += 1
        
        # 统计报告
        logger.info("\n" + "="*60)
        logger.info("📊 处理完成统计:")
        logger.info(f"  ✅ 成功: {success_count}")
        logger.info(f"  ❌ 失败: {failed_count}")
        if total > 0:
            logger.info(f"  📈 成功率: {success_count/total*100:.1f}%")
        logger.info("="*60)
    
    def run(self, target_dir: Optional[str] = None):
        """运行主流程"""
        try:
            # 初始化
            self.initialize()
            
            # 获取文件列表
            directory = target_dir or self.config.target_dir
            files = self.get_files(directory)
            
            if not files:
                logger.warning("⚠️  未发现支持的文件")
                return
            
            # 批量处理
            self.process_batch(files)
            
        except Exception as e:
            logger.error(f"❌ 处理流程异常: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        finally:
            if self.db_manager:
                self.db_manager.close()


# ------------------- 主函数 -------------------
def main():
    """主函数入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='统一文件向量化存储工具（支持PDF、TXT等）')
    parser.add_argument('-c', '--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('-d', '--dir', type=str, help='文件目录路径')
    parser.add_argument('-f', '--file', type=str, help='单个文件路径')
    parser.add_argument('--init-only', action='store_true', help='仅初始化表结构')
    
    args = parser.parse_args()
    
    uploader = VectorUploader(args.config)
    
    try:
        if args.init_only:
            uploader.initialize()
            logger.info("✅ 初始化完成")
        elif args.file:
            uploader.initialize()
            uploader.process_single_file(args.file)
        elif args.dir:
            uploader.run(args.dir)
        else:
            uploader.run()
    except KeyboardInterrupt:
        logger.info("\n⚠️  用户中断操作")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ 程序异常退出: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
