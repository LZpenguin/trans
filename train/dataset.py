from datasets import load_dataset, Dataset, concatenate_datasets, Features, Value
import pandas as pd
from typing import Dict, List, Optional
from tqdm import tqdm
import os
import hashlib

class TranslationDataset:
    def __init__(self, data_type: str = "csv", double: bool = False, cache_dir: str = None, target_lang: str = None):
        """
        初始化翻译数据集加载器
        Args:
            data_type: 数据集类型，'csv'、'parquet' 或 'x'
            double: 是否生成双向翻译数据
            cache_dir: 缓存目录，默认为 ~/.cache/huggingface/datasets
            target_lang: 对于x数据类型，指定目标语言 ('英语', '泰语', '马来语')
        """
        self.data_type = data_type
        self.double = double
        self.cache_dir = cache_dir
        self.target_lang = target_lang
        self.lang_map = {
            "马来语": "Malay",
            "泰语": "Thai",
            "英语": "English",
            '中文': 'Chinese',
            'ms': 'Malay',
            'th': 'Thai',
            'en': 'English',
            'zh': 'Chinese'
        }
        # 定义数据集特征
        self.features = Features({
            'source_lang': Value('string'),
            'source_text': Value('string'),
            'target_lang': Value('string'),
            'target_text': Value('string'),
            'prompt': Value('string')
        })
    
    def _get_cache_path(self, data_path: str) -> str:
        """
        获取缓存文件路径
        Args:
            data_path: 原始数据路径
        Returns:
            缓存文件路径
        """
        # 使用数据路径、数据类型、double参数和target_lang参数生成唯一的缓存标识
        cache_id = f"{data_path}_{self.data_type}_{self.double}_{self.target_lang}"
        # 计算哈希值作为缓存文件名
        cache_name = hashlib.md5(cache_id.encode()).hexdigest()
        
        # 确定缓存目录
        if self.cache_dir:
            cache_dir = self.cache_dir
        else:
            cache_dir = os.path.expanduser("~/.cache/huggingface/datasets/translation_cache")
        
        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)
        
        return os.path.join(cache_dir, f"{cache_name}.cache")

    def _create_prompt(self, source_lang: str, target_lang: str, source_text: str, target_text: str) -> str:
        """
        创建提示文本
        Args:
            source_lang: 源语言
            target_lang: 目标语言
            source_text: 源文本
            target_text: 目标文本
        Returns:
            提示文本
        """
        return f"Translate this from {source_lang} to {target_lang}:\n{source_lang}: {source_text}\n{target_lang}: {target_text}<eos>"
    
    def _process_ft_sample(self, example: Dict) -> Dict:
        """
        处理单个微调数据样本
        Args:
            example: 原始样本
        Returns:
            处理后的样本
        """
        result = {
            'source_lang': 'Chinese',
            'source_text': example['中文'],
            'target_lang': self.lang_map[example['语言']],
            'target_text': example['文本']
        }
        result['prompt'] = self._create_prompt(
            result['source_lang'],
            result['target_lang'],
            result['source_text'],
            result['target_text']
        )
        
        if self.double:
            reverse = {
                'source_lang': result['target_lang'],
                'source_text': result['target_text'],
                'target_lang': result['source_lang'],
                'target_text': result['source_text']
            }
            reverse['prompt'] = self._create_prompt(
                reverse['source_lang'],
                reverse['target_lang'],
                reverse['source_text'],
                reverse['target_text']
            )
            return {'samples': [result, reverse]}
        return {'samples': [result]}

    def _process_pt_sample(self, example: Dict) -> Dict:
        """
        处理单个预训练数据样本
        Args:
            example: 原始样本
        Returns:
            处理后的样本
        """
        trans = example['translation']
        source_lang = 'en'
        target_lang = None
        for lang in ['zh', 'ms', 'th']:
            if lang in trans:
                target_lang = lang
                break

        if source_lang and target_lang:
            result = {
                'source_lang': self.lang_map[source_lang],
                'source_text': trans[source_lang],
                'target_lang': self.lang_map[target_lang],
                'target_text': trans[target_lang]
            }
            result['prompt'] = self._create_prompt(
                result['source_lang'],
                result['target_lang'],
                result['source_text'],
                result['target_text']
            )
            
            if self.double:
                reverse = {
                    'source_lang': result['target_lang'],
                    'source_text': result['target_text'],
                    'target_lang': result['source_lang'],
                    'target_text': result['source_text']
                }
                reverse['prompt'] = self._create_prompt(
                    reverse['source_lang'],
                    reverse['target_lang'],
                    reverse['source_text'],
                    reverse['target_text']
                )
                return {'samples': [result, reverse]}
            return {'samples': [result]}
        return {'samples': []}

    def _process_x_sample(self, example: Dict) -> Dict:
        """
        处理单个x数据样本
        Args:
            example: 原始样本
        Returns:
            处理后的样本
        """
        if not self.target_lang:
            # 如果没有指定目标语言，生成所有可能的翻译对
            samples = []
            for lang_col in ['英语', '泰语', '马来语']:
                if lang_col in example and example[lang_col] and example[lang_col].strip():
                    result = {
                        'source_lang': 'Chinese',
                        'source_text': example['中文'],
                        'target_lang': self.lang_map[lang_col],
                        'target_text': example[lang_col]
                    }
                    result['prompt'] = self._create_prompt(
                        result['source_lang'],
                        result['target_lang'],
                        result['source_text'],
                        result['target_text']
                    )
                    samples.append(result)
                    
                    if self.double:
                        reverse = {
                            'source_lang': result['target_lang'],
                            'source_text': result['target_text'],
                            'target_lang': result['source_lang'],
                            'target_text': result['source_text']
                        }
                        reverse['prompt'] = self._create_prompt(
                            reverse['source_lang'],
                            reverse['target_lang'],
                            reverse['source_text'],
                            reverse['target_text']
                        )
                        samples.append(reverse)
            return {'samples': samples}
        else:
            # 如果指定了目标语言，只生成该语言的翻译对
            if self.target_lang in example and example[self.target_lang] and example[self.target_lang].strip():
                result = {
                    'source_lang': 'Chinese',
                    'source_text': example['中文'],
                    'target_lang': self.lang_map[self.target_lang],
                    'target_text': example[self.target_lang]
                }
                result['prompt'] = self._create_prompt(
                    result['source_lang'],
                    result['target_lang'],
                    result['source_text'],
                    result['target_text']
                )
                
                if self.double:
                    reverse = {
                        'source_lang': result['target_lang'],
                        'source_text': result['target_text'],
                        'target_lang': result['source_lang'],
                        'target_text': result['source_text']
                    }
                    reverse['prompt'] = self._create_prompt(
                        reverse['source_lang'],
                        reverse['target_lang'],
                        reverse['source_text'],
                        reverse['target_text']
                    )
                    return {'samples': [result, reverse]}
                return {'samples': [result]}
            return {'samples': []}

    def load_csv_data(self, data_path: str) -> Dataset:
        """
        加载微调数据集
        Args:
            data_path: 数据文件路径
        Returns:
            加载的数据集
        """
        cache_path = self._get_cache_path(data_path)
        
        # 检查是否存在缓存
        if os.path.exists(cache_path):
            print(f"发现缓存文件，从缓存加载数据: {cache_path}")
            return Dataset.load_from_disk(cache_path)
            
        # 如果没有缓存，处理数据
        print(f"未发现缓存，处理数据并创建缓存: {cache_path}")
        
        # 读取CSV并转换为Dataset格式
        df = pd.read_csv(data_path, encoding='utf-8')
        dataset = Dataset.from_pandas(df)
        
        # 使用map处理数据并直接展开
        processed = dataset.map(
            self._process_ft_sample,
            remove_columns=dataset.column_names,
            desc="处理微调数据",
            load_from_cache_file=False
        ).select_columns(['samples'])

        # 展平samples列表
        all_samples = []
        for item in tqdm(processed, desc="展平数据"):
            if item['samples']:
                all_samples.extend(item['samples'])
        
        if not all_samples:
            raise ValueError(f"处理后没有有效的样本。请检查输入文件：{data_path}")
            
        result = Dataset.from_list(all_samples, features=self.features)
        
        # 保存缓存
        result.save_to_disk(cache_path)
        return result

    def load_parquet_data(self, data_path: str) -> Dataset:
        """
        加载预训练数据集
        Args:
            data_path: 数据文件路径
        Returns:
            加载的数据集
        """
        cache_path = self._get_cache_path(data_path)
        
        # 检查是否存在缓存
        if os.path.exists(cache_path):
            print(f"发现缓存文件，从缓存加载数据: {cache_path}")
            return Dataset.load_from_disk(cache_path)
            
        # 如果没有缓存，处理数据
        print(f"未发现缓存，处理数据并创建缓存: {cache_path}")
        
        dataset = load_dataset("parquet", data_files=data_path)['train']

        # 使用map处理数据并直接展开
        processed = dataset.map(
            self._process_pt_sample,
            remove_columns=dataset.column_names,
            desc="处理预训练数据",
            load_from_cache_file=False
        ).select_columns(['samples'])

        # 展平samples列表
        all_samples = []
        for item in tqdm(processed, desc="展平数据"):
            if item['samples']:
                all_samples.extend(item['samples'])
        
        if not all_samples:
            raise ValueError(f"处理后没有有效的样本。请检查输入文件：{data_path}")
            
        result = Dataset.from_list(all_samples, features=self.features)
        
        # 保存缓存
        result.save_to_disk(cache_path)
        return result

    def load_x_data(self, data_path: str) -> Dataset:
        """
        加载x数据集
        Args:
            data_path: 数据文件路径
        Returns:
            加载的数据集
        """
        cache_path = self._get_cache_path(data_path)
        
        # 检查是否存在缓存
        if os.path.exists(cache_path):
            print(f"发现缓存文件，从缓存加载数据: {cache_path}")
            return Dataset.load_from_disk(cache_path)
            
        # 如果没有缓存，处理数据
        print(f"未发现缓存，处理数据并创建缓存: {cache_path}")
        
        dataset = load_dataset("parquet", data_files=data_path)['train']

        # 使用map处理数据并直接展开
        processed = dataset.map(
            self._process_x_sample,
            remove_columns=dataset.column_names,
            desc="处理x数据",
            load_from_cache_file=False
        ).select_columns(['samples'])

        # 展平samples列表
        all_samples = []
        for item in tqdm(processed, desc="展平数据"):
            if item['samples']:
                all_samples.extend(item['samples'])
        
        if not all_samples:
            raise ValueError(f"处理后没有有效的样本。请检查输入文件：{data_path}")
            
        result = Dataset.from_list(all_samples, features=self.features)
        
        # 保存缓存
        result.save_to_disk(cache_path)
        return result

    def load_data(self, data_path: str) -> Dataset:
        """
        根据数据类型加载相应的数据集
        Args:
            data_path: 数据文件路径
        Returns:
            加载的数据集，格式为：
            {
                'source_lang': str,  # 源语言
                'source_text': str,  # 源文本
                'target_lang': str,  # 目标语言
                'target_text': str,  # 目标文本
                'prompt': str        # 提示文本
            }
            如果double=True，会生成双向翻译数据
        """
        if self.data_type == "csv":
            return self.load_csv_data(data_path)
        elif self.data_type == "x":
            return self.load_x_data(data_path)
        else:
            return self.load_parquet_data(data_path) 