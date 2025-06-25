#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理SRT文件：
1. 将带有换行的台词中的换行替换为空格
2. 删除所有的[和]符号
3. 在原位置保存文件
"""

import os
import re
from pathlib import Path

def process_srt_content(content):
    """
    处理SRT文件内容
    """
    lines = content.split('\n')
    processed_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # 如果是序号行（纯数字）
        if line.isdigit():
            processed_lines.append(line)
            i += 1
            continue
            
        # 如果是时间戳行
        if '-->' in line:
            processed_lines.append(line)
            i += 1
            continue
            
        # 如果是空行
        if not line:
            processed_lines.append('')
            i += 1
            continue
            
        # 处理字幕文本行
        # 收集连续的非空行作为一个字幕块
        subtitle_lines = []
        while i < len(lines) and lines[i].strip() and not lines[i].strip().isdigit() and '-->' not in lines[i]:
            subtitle_lines.append(lines[i].strip())
            i += 1
            
        # 将多行字幕合并为一行，用空格连接
        if subtitle_lines:
            combined_subtitle = ' '.join(subtitle_lines)
            # 删除所有的[和]
            combined_subtitle = combined_subtitle.replace('[', '').replace(']', '')
            processed_lines.append(combined_subtitle)
    
    return '\n'.join(processed_lines)

def process_srt_file(filepath):
    """
    处理单个SRT文件
    """
    try:
        # 读取文件内容
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 处理内容
        processed_content = process_srt_content(content)
        
        # 写回文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(processed_content)
        
        print(f"已处理: {filepath}")
        return True
        
    except Exception as e:
        print(f"处理文件 {filepath} 时出错: {e}")
        return False

def find_and_process_srt_files(root_dir):
    """
    查找并处理所有SRT文件
    """
    root_path = Path(root_dir)
    srt_files = list(root_path.rglob('*.srt'))
    
    print(f"找到 {len(srt_files)} 个SRT文件")
    
    processed_count = 0
    failed_count = 0
    
    for srt_file in srt_files:
        if process_srt_file(srt_file):
            processed_count += 1
        else:
            failed_count += 1
    
    print(f"\n处理完成!")
    print(f"成功处理: {processed_count} 个文件")
    print(f"处理失败: {failed_count} 个文件")

if __name__ == "__main__":
    # 设置根目录（当前脚本所在目录）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("开始处理SRT文件...")
    print(f"搜索目录: {current_dir}")
    
    find_and_process_srt_files(current_dir)
