import os
import re
import pandas as pd
import glob
from pathlib import Path

def clean_text(text):
    """清理文本，删除方括号"""
    if not text:
        return text
    # 删除方括号及其内容，但保留方括号外的内容
    cleaned = re.sub(r'\[([^\]]*)\]', r'\1', text)
    return cleaned.strip()

def parse_srt_file(file_path):
    """解析SRT文件，返回时间戳和内容的列表"""
    subtitles = []
    
    if not os.path.exists(file_path):
        return subtitles
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                content = f.read()
        except:
            print(f"无法读取文件: {file_path}")
            return subtitles
    
    # 检查文件是否为空
    if not content.strip():
        return subtitles
    
    # 分割字幕块
    blocks = re.split(r'\n\s*\n', content.strip())
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            # 第一行是序号
            try:
                index = int(lines[0])
            except:
                continue
            
            # 第二行是时间戳
            timestamp_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', lines[1])
            if timestamp_match:
                start_time = timestamp_match.group(1)
                end_time = timestamp_match.group(2)
                timestamp = f"{start_time} --> {end_time}"
                
                # 剩余行是字幕内容
                subtitle_text = '\n'.join(lines[2:]).strip()
                # 清理文本，删除方括号
                subtitle_text = clean_text(subtitle_text)
                
                subtitles.append({
                    'index': index,
                    'timestamp': timestamp,
                    'text': subtitle_text
                })
    
    return subtitles

def align_subtitles_from_end(chinese_subs, english_subs, thai_subs, malay_subs):
    """从后往前对齐字幕，以最少的台词数为准"""
    # 获取各语言的台词数量
    counts = [len(chinese_subs), len(english_subs), len(thai_subs), len(malay_subs)]
    # 排除0条的语言，从非0的语言中找最小值
    non_zero_counts = [count for count in counts if count > 0]
    
    if not non_zero_counts:
        print("    所有语言都没有台词，跳过")
        return []
    
    min_count = min(non_zero_counts)
    
    print(f"    台词数量: 中文{len(chinese_subs)}, 英语{len(english_subs)}, 泰语{len(thai_subs)}, 马来语{len(malay_subs)}, 非0最小值{min_count}")
    
    # 从后往前取最少的台词数
    aligned_chinese = chinese_subs[-min_count:] if chinese_subs else [{'timestamp': '', 'text': ''}] * min_count
    aligned_english = english_subs[-min_count:] if english_subs else [{'timestamp': '', 'text': ''}] * min_count
    aligned_thai = thai_subs[-min_count:] if thai_subs else [{'timestamp': '', 'text': ''}] * min_count
    aligned_malay = malay_subs[-min_count:] if malay_subs else [{'timestamp': '', 'text': ''}] * min_count
    
    aligned_data = []
    for i in range(min_count):
        aligned_data.append({
            'chinese_timestamp': aligned_chinese[i]['timestamp'] if aligned_chinese else '',
            'english_timestamp': aligned_english[i]['timestamp'] if aligned_english else '',
            'thai_timestamp': aligned_thai[i]['timestamp'] if aligned_thai else '',
            'malay_timestamp': aligned_malay[i]['timestamp'] if aligned_malay else '',
            'chinese': aligned_chinese[i]['text'],
            'english': aligned_english[i]['text'],
            'thai': aligned_thai[i]['text'],
            'malay': aligned_malay[i]['text']
        })
    
    return aligned_data

def process_tv_series():
    """处理所有电视剧数据"""
    base_dir = Path('.')
    all_data = []
    
    # 查找所有电视剧目录
    tv_dirs = []
    for item in os.listdir(base_dir):
        if os.path.isdir(item) and '电视剧' in item:
            tv_dirs.append(item)
    
    tv_dirs.sort()  # 按名称排序
    
    for tv_dir in tv_dirs:
        print(f"处理 {tv_dir}...")
        tv_path = base_dir / tv_dir
        
        # 提取电视剧编号
        tv_match = re.search(r'电视剧(\d+)', tv_dir)
        if tv_match:
            tv_number = f"电视剧{tv_match.group(1)}"
        else:
            tv_number = tv_dir
        
        # 查找所有EP目录
        ep_dirs = []
        for item in os.listdir(tv_path):
            if os.path.isdir(tv_path / item) and item.startswith('EP'):
                ep_dirs.append(item)
        
        ep_dirs.sort()  # 按EP编号排序
        
        for ep_dir in ep_dirs:
            print(f"  处理 {ep_dir}...")
            ep_path = tv_path / ep_dir
            
            # 读取四种语言的SRT文件
            chinese_file = ep_path / '中文.srt'
            english_file = ep_path / '英语.srt'
            thai_file = ep_path / '泰语.srt'
            malay_file = ep_path / '马来语.srt'
            
            # 解析各语言字幕
            chinese_subs = parse_srt_file(chinese_file)
            english_subs = parse_srt_file(english_file)
            thai_subs = parse_srt_file(thai_file)
            malay_subs = parse_srt_file(malay_file)
            
            # 从后往前对齐字幕
            aligned_data = align_subtitles_from_end(chinese_subs, english_subs, thai_subs, malay_subs)
            
            # 添加到总数据中
            for item in aligned_data:
                row_data = {
                    '电视剧': tv_number,
                    '集数': ep_dir,
                    '中文时间戳': item['chinese_timestamp'],
                    '英语时间戳': item['english_timestamp'],
                    '泰语时间戳': item['thai_timestamp'],
                    '马来语时间戳': item['malay_timestamp'],
                    '中文': item['chinese'],
                    '英语': item['english'],
                    '泰语': item['thai'],
                    '马来语': item['malay']
                }
                all_data.append(row_data)
    
    return all_data

def main():
    """主函数"""
    print("开始处理电视剧数据...")
    
    # 处理所有数据
    data = process_tv_series()
    
    if not data:
        print("没有找到数据!")
        return
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    print(f"总共处理了 {len(df)} 条记录")
    print(f"包含的电视剧: {df['电视剧'].unique()}")
    
    # 保存为parquet文件
    output_file = 'x.parquet'
    df.to_parquet(output_file, index=False)
    
    print(f"数据已保存到: {output_file}")
    print(f"文件大小: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    
    # 显示一些样本数据
    print("\n前5行数据预览:")
    print(df.head().to_string())
    
    # 显示统计信息
    print(f"\n数据统计:")
    print(f"总行数: {len(df)}")
    print(f"电视剧数量: {df['电视剧'].nunique()}")
    print(f"集数分布:")
    for tv in sorted(df['电视剧'].unique()):
        eps = df[df['电视剧'] == tv]['集数'].nunique()
        total_lines = len(df[df['电视剧'] == tv])
        print(f"  {tv}: {eps} 集, {total_lines} 条台词")

if __name__ == "__main__":
    main()
