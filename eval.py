#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
翻译质量评估脚本
使用 BLEU2 和 XCOMET-XL 指标评估翻译质量
参考 train_nllb.py 中的评估方法
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
import sys
import pandas as pd
import numpy as np
import argparse
import warnings
from tqdm import tqdm
import logging

# 导入评估相关库
from nltk.translate.bleu_score import sentence_bleu
from comet import load_from_checkpoint

# 尝试导入泰语分词工具
try:
    from pythainlp import word_tokenize as thai_tokenize
    THAI_TOKENIZER_AVAILABLE = True
except ImportError:
    THAI_TOKENIZER_AVAILABLE = False
    print("警告：未安装 pythainlp，泰语将使用字符级分词")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 过滤警告
warnings.filterwarnings('ignore')


def tokenize_for_bleu(text, language):
    """根据语言类型进行适当的分词"""
    if language in ["Thai", "泰语"]:
        if THAI_TOKENIZER_AVAILABLE:
            # 使用泰语分词工具
            try:
                tokens = thai_tokenize(text, engine='newmm')
                return tokens
            except Exception as e:
                logger.warning(f"泰语分词失败，使用字符级分词: {e}")
                # 如果分词失败，使用字符级分词（去除空格）
                return list(text.replace(" ", ""))
        else:
            # 如果没有安装pythainlp，使用字符级分词
            return list(text.replace(" ", ""))
    elif language in ["English", "英语"]:
        # 英语使用空格分词
        return text.split()
    elif language in ["Malay", "马来语"]:
        # 马来语也使用空格分词
        return text.split()
    else:
        # 默认使用空格分词
        return text.split()


class TranslationEvaluator:
    """翻译评估器"""
    
    def __init__(self, comet_model_path=None):
        """
        初始化评估器
        
        Args:
            comet_model_path: COMET模型路径，如果为None则使用默认路径
        """
        self.comet_model = None
        self.comet_model_path = comet_model_path
        
    def _load_comet_model(self):
        """加载COMET模型"""
        if self.comet_model is None:
            try:
                # if self.comet_model_path:
                self.comet_model = load_from_checkpoint(self.comet_model_path)
                # else:
                #     # 尝试使用本地模型路径
                #     local_model_path = f"{os.path.dirname(os.path.abspath(__file__))}/models/XCOMET-XL/checkpoints/model.ckpt"
                #     if os.path.exists(local_model_path):
                #         self.comet_model = load_from_checkpoint(local_model_path)
                #         logger.info(f"加载本地COMET模型: {local_model_path}")
                #     else:
                #         # 如果本地模型不存在，尝试下载默认模型
                #         from comet import download_model
                #         model_path = download_model("Unbabel/XCOMET-XL")
                #         self.comet_model = load_from_checkpoint(model_path)
                #         logger.info(f"下载并加载COMET模型: {model_path}")
            except Exception as e:
                logger.error(f"COMET模型加载失败: {e}")
                self.comet_model = None
        return self.comet_model
    
    def calculate_bleu2(self, references, predictions, languages):
        """
        计算BLEU2分数
        
        Args:
            references: 参考翻译列表
            predictions: 预测翻译列表
            languages: 语言列表
            
        Returns:
            tuple: (bleu2_scores, avg_bleu2)
        """
        bleu_scores = []
        
        for ref, pred, lang in tqdm(zip(references, predictions, languages), 
                                   desc="计算BLEU2", total=len(references)):
            try:
                ref_tokens = tokenize_for_bleu(ref, lang)
                pred_tokens = tokenize_for_bleu(pred, lang)
                
                if len(pred_tokens) > 0 and len(ref_tokens) > 0:
                    # 使用BLEU2 (bigram)
                    bleu = sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0))
                    bleu_scores.append(bleu)
                else:
                    bleu_scores.append(0.0)
            except Exception as e:
                logger.warning(f"BLEU计算失败: {e}")
                bleu_scores.append(0.0)
        
        avg_bleu2 = np.mean(bleu_scores) if bleu_scores else 0.0
        return bleu_scores, avg_bleu2
    
    def calculate_comet(self, sources, references, predictions, batch_size=8):
        """
        计算COMET分数
        
        Args:
            sources: 源文本列表
            references: 参考翻译列表
            predictions: 预测翻译列表
            batch_size: 批处理大小
            
        Returns:
            tuple: (comet_scores, avg_comet)
        """
        comet_model = self._load_comet_model()
        
        if comet_model is None:
            logger.warning("COMET模型未加载，跳过COMET评估")
            return [0.0] * len(references), 0.0
        
        try:
            # 准备COMET评估数据
            comet_data = []
            for src, ref, pred in zip(sources, references, predictions):
                comet_data.append({
                    "src": src,
                    "mt": pred,
                    "ref": ref
                })
            
            logger.info(f"开始计算COMET分数，共{len(comet_data)}个样本")
            comet_output = comet_model.predict(comet_data, batch_size=batch_size, gpus=1)
            comet_scores = comet_output.scores
            avg_comet = np.mean(comet_scores)
            
            return comet_scores, avg_comet
            
        except Exception as e:
            logger.error(f"COMET分数计算失败: {e}")
            return [0.0] * len(references), 0.0
    
    def evaluate_by_language(self, df, source_col='中文', reference_col='中文', 
                           prediction_col='answer', language_col='语言', 
                           max_samples_per_lang=None):
        """
        按语言分别评估翻译质量
        
        Args:
            df: 包含翻译数据的DataFrame
            source_col: 源文本列名
            reference_col: 参考翻译列名（对于中译外任务，参考翻译就是源文本）
            prediction_col: 预测翻译列名
            language_col: 语言列名
            max_samples_per_lang: 每种语言最大评估样本数
            
        Returns:
            dict: 评估结果
        """
        results = {
            'by_language': {},
            'overall': {}
        }
        
        # 过滤掉无效数据（answer为空的行）
        valid_mask = (
            df[prediction_col].notna() & 
            (df[prediction_col] != "") &
            (df[prediction_col].astype(str).str.strip() != "") &  # 过滤只有空格的情况
            df[reference_col].notna() &
            (df[reference_col] != "")
        )
        df_valid = df[valid_mask].copy()
        
        logger.info(f"有效样本数: {len(df_valid)} / {len(df)}")
        
        # 按语言分组
        languages = df_valid[language_col].unique()
        logger.info(f"检测到的语言: {languages}")
        
        all_sources = []
        all_references = []
        all_predictions = []
        all_languages = []
        
        for lang in languages:
            lang_data = df_valid[df_valid[language_col] == lang]
            
            # 再次确保没有空的answer
            lang_data = lang_data[
                lang_data[prediction_col].notna() & 
                (lang_data[prediction_col] != "") &
                (lang_data[prediction_col].astype(str).str.strip() != "")
            ]
            
            # 如果指定了最大样本数，进行采样
            if max_samples_per_lang and len(lang_data) > max_samples_per_lang:
                original_count = len(df_valid[df_valid[language_col] == lang])
                lang_data = lang_data.sample(n=max_samples_per_lang, random_state=42)
                logger.info(f"{lang}: 从{original_count}个样本中采样{max_samples_per_lang}个（过滤后有效样本数：{len(lang_data)}）")
            else:
                logger.info(f"{lang}: 使用全部{len(lang_data)}个有效样本")
            
            if len(lang_data) == 0:
                logger.warning(f"{lang}: 没有有效样本，跳过")
                continue
            
            sources = lang_data[source_col].tolist()
            references = lang_data[reference_col].tolist()
            predictions = lang_data[prediction_col].tolist()
            languages_list = [lang] * len(lang_data)
            
            logger.info(f"开始评估 {lang} (样本数: {len(lang_data)})")
            
            # 计算BLEU2分数
            bleu_scores, avg_bleu2 = self.calculate_bleu2(references, predictions, languages_list)
            
            # 计算COMET分数
            comet_scores, avg_comet = self.calculate_comet(sources, references, predictions)
            
            # 计算最终分数 (BLEU2 * 0.4 + COMET * 0.6)
            final_score = avg_bleu2 * 0.4 + avg_comet * 0.6
            
            # 存储结果
            results['by_language'][lang] = {
                'samples': len(lang_data),
                'bleu2': avg_bleu2,
                'comet': avg_comet,
                'final': final_score,
                'bleu2_scores': bleu_scores,
                'comet_scores': comet_scores
            }
            
            logger.info(f"{lang} - BLEU2: {avg_bleu2:.4f}, COMET: {avg_comet:.4f}, Final: {final_score:.4f}")
            
            # 收集到总体评估数据中
            all_sources.extend(sources)
            all_references.extend(references)
            all_predictions.extend(predictions)
            all_languages.extend(languages_list)
        
        # 计算总体平均分数
        if all_references:
            # 加权平均（按样本数加权）
            total_samples = sum(results['by_language'][lang]['samples'] 
                              for lang in results['by_language'])
            
            weighted_bleu2 = sum(results['by_language'][lang]['bleu2'] * 
                                results['by_language'][lang]['samples']
                                for lang in results['by_language']) / total_samples
            
            weighted_comet = sum(results['by_language'][lang]['comet'] * 
                                results['by_language'][lang]['samples']
                                for lang in results['by_language']) / total_samples
            
            overall_final = weighted_bleu2 * 0.4 + weighted_comet * 0.6
            
            results['overall'] = {
                'total_samples': total_samples,
                'bleu2': weighted_bleu2,
                'comet': weighted_comet,
                'final': overall_final
            }
            
            logger.info(f"总体 - BLEU2: {weighted_bleu2:.4f}, COMET: {weighted_comet:.4f}, Final: {overall_final:.4f}")
        
        return results


def print_results(results):
    """打印评估结果"""
    print("\n" + "="*60)
    print("翻译质量评估结果")
    print("="*60)
    
    # 打印各语言结果
    print("\n各语言评估结果:")
    print("-" * 60)
    print(f"{'语言':<10} {'样本数':<8} {'BLEU2':<8} {'COMET':<8} {'最终分数':<8}")
    print("-" * 60)
    
    for lang, metrics in results['by_language'].items():
        print(f"{lang:<10} {metrics['samples']:<8} "
              f"{metrics['bleu2']:<8.4f} {metrics['comet']:<8.4f} {metrics['final']:<8.4f}")
    
    # 打印总体结果
    if 'overall' in results:
        print("-" * 60)
        print(f"{'总体':<10} {results['overall']['total_samples']:<8} "
              f"{results['overall']['bleu2']:<8.4f} {results['overall']['comet']:<8.4f} "
              f"{results['overall']['final']:<8.4f}")
    
    print("="*60)
    print("注：最终分数 = BLEU2 × 0.4 + COMET × 0.6")


def save_results(results, output_file):
    """保存详细评估结果到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("翻译质量评估详细结果\n")
        f.write("="*60 + "\n\n")
        
        # 写入各语言结果
        f.write("各语言评估结果:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'语言':<10} {'样本数':<8} {'BLEU2':<8} {'COMET':<8} {'最终分数':<8}\n")
        f.write("-" * 60 + "\n")
        
        for lang, metrics in results['by_language'].items():
            f.write(f"{lang:<10} {metrics['samples']:<8} "
                   f"{metrics['bleu2']:<8.4f} {metrics['comet']:<8.4f} {metrics['final']:<8.4f}\n")
        
        # 写入总体结果
        if 'overall' in results:
            f.write("-" * 60 + "\n")
            f.write(f"{'总体':<10} {results['overall']['total_samples']:<8} "
                   f"{results['overall']['bleu2']:<8.4f} {results['overall']['comet']:<8.4f} "
                   f"{results['overall']['final']:<8.4f}\n")
        
        f.write("="*60 + "\n")
        f.write("注：最终分数 = BLEU2 × 0.4 + COMET × 0.6\n")
    
    logger.info(f"详细结果已保存到: {output_file}")


def main():
    model = 'GemmaX2-28-2B-v0.1-ft-full-5'
    suffix = 'translate_7500'
    parser = argparse.ArgumentParser(description='翻译质量评估工具')
    parser.add_argument('--input_file', default=f'output/{model}/dev_{suffix}.csv', help='输入CSV文件路径')
    parser.add_argument('--source-col', default='中文', help='源文本列名（默认：中文）')
    parser.add_argument('--language-col', default='语言', help='语言列名（默认：语言）')
    parser.add_argument('--reference-col', default='文本', help='参考翻译列名（默认：中文）')
    parser.add_argument('--prediction-col', default='answer', help='预测翻译列名（默认：answer）')
    parser.add_argument('--comet-model-path', default=f'{os.path.dirname(os.path.abspath(__file__))}/models/XCOMET-XL/checkpoints/model.ckpt', help='COMET模型路径')
    parser.add_argument('--output-file', default=f'output/{model}/dev_{suffix}_eval.csv', help='结果输出文件路径')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input_file):
        logger.error(f"输入文件不存在: {args.input_file}")
        sys.exit(1)
    
    # 读取数据
    logger.info(f"读取数据文件: {args.input_file}")
    try:
        df = pd.read_csv(args.input_file)
        logger.info(f"成功读取数据，共{len(df)}行")
    except Exception as e:
        logger.error(f"读取文件失败: {e}")
        sys.exit(1)
    
    # 检查必要的列
    required_cols = [args.reference_col, args.prediction_col, args.language_col, args.source_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"缺少必要的列: {missing_cols}")
        logger.info(f"文件中的列: {list(df.columns)}")
        sys.exit(1)
    
    # 创建评估器
    evaluator = TranslationEvaluator(comet_model_path=args.comet_model_path)
    
    # 进行评估
    logger.info("开始翻译质量评估...")
    results = evaluator.evaluate_by_language(
        df, 
        source_col=args.source_col,
        reference_col=args.reference_col,
        prediction_col=args.prediction_col,
        language_col=args.language_col
    )
    
    # 打印结果
    print_results(results)
    
    # 保存结果到文件
    if args.output_file:
        save_results(results, args.output_file)


if __name__ == "__main__":
    main() 