#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SRT字幕文件时间戳调整脚本
用于调整指定目录下所有SRT文件中每句台词的时间戳
"""

import os
import re
import argparse
from pathlib import Path


def parse_time(time_str):
    """解析SRT时间格式：HH:MM:SS,mmm"""
    match = re.match(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})', time_str)
    if not match:
        raise ValueError(f"Invalid time format: {time_str}")
    
    hours, minutes, seconds, milliseconds = map(int, match.groups())
    total_ms = hours * 3600000 + minutes * 60000 + seconds * 1000 + milliseconds
    return total_ms


def format_time(total_ms):
    """将毫秒转换回SRT时间格式"""
    # 确保时间不为负数
    if total_ms < 0:
        total_ms = 0
    
    hours = total_ms // 3600000
    minutes = (total_ms % 3600000) // 60000
    seconds = (total_ms % 60000) // 1000
    milliseconds = total_ms % 1000
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def adjust_srt_file(file_path, offset_seconds):
    """调整单个SRT文件的时间戳"""
    offset_ms = int(offset_seconds * 1000)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 匹配时间戳行：00:00:00,000 --> 00:00:00,000
    time_pattern = r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})'
    
    def replace_time(match):
        start_time = match.group(1)
        end_time = match.group(2)
        
        start_ms = parse_time(start_time) + offset_ms
        end_ms = parse_time(end_time) + offset_ms
        
        new_start = format_time(start_ms)
        new_end = format_time(end_ms)
        
        return f"{new_start} --> {new_end}"
    
    new_content = re.sub(time_pattern, replace_time, content)
    
    # 创建备份
    backup_path = file_path.with_suffix('.srt.backup')
    file_path.rename(backup_path)
    
    # 写入调整后的内容
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    return backup_path


def adjust_directory_srt_files(directory_path, offset_seconds):
    """调整指定目录下所有SRT文件的时间戳"""
    directory = Path(directory_path)
    
    if not directory.exists():
        raise ValueError(f"目录不存在: {directory_path}")
    
    srt_files = list(directory.glob("*.srt"))
    
    if not srt_files:
        print(f"在目录 {directory_path} 中未找到SRT文件")
        return
    
    print(f"找到 {len(srt_files)} 个SRT文件:")
    for srt_file in srt_files:
        print(f"  - {srt_file.name}")
    
    print(f"\n开始调整时间戳，偏移量: {offset_seconds} 秒")
    
    for srt_file in srt_files:
        try:
            backup_path = adjust_srt_file(srt_file, offset_seconds)
            print(f"✓ 已处理: {srt_file.name} (备份: {backup_path.name})")
        except Exception as e:
            print(f"✗ 处理失败: {srt_file.name} - {str(e)}")
    
    print(f"\n调整完成！所有原文件已备份为 .srt.backup")


def main():
    parser = argparse.ArgumentParser(description='调整SRT字幕文件时间戳')
    parser.add_argument('directory', help='包含SRT文件的目录路径')
    parser.add_argument('offset', type=float, help='时间偏移量（秒），正数向后延迟，负数向前提前')
    parser.add_argument('--restore', action='store_true', help='恢复备份文件')
    
    args = parser.parse_args()
    
    if args.restore:
        restore_backups(args.directory)
    else:
        adjust_directory_srt_files(args.directory, args.offset)


def restore_backups(directory_path):
    """恢复备份文件"""
    directory = Path(directory_path)
    backup_files = list(directory.glob("*.srt.backup"))
    
    if not backup_files:
        print("未找到备份文件")
        return
    
    print(f"找到 {len(backup_files)} 个备份文件，开始恢复...")
    
    for backup_file in backup_files:
        original_file = backup_file.with_suffix('')
        if original_file.exists():
            original_file.unlink()
        backup_file.rename(original_file)
        print(f"✓ 已恢复: {original_file.name}")
    
    print("恢复完成！")


if __name__ == "__main__":
    # 如果直接运行脚本，可以在这里设置默认参数进行测试
    import sys
    
    if len(sys.argv) == 1:
        # 交互模式
        print("SRT字幕文件时间戳调整工具")
        print("=" * 40)
        
        directory = input("请输入包含SRT文件的目录路径: ").strip()
        if not directory:
            directory = "/mnt/gold/lz/trans/data/x/电视剧2/EP12"
            print(f"使用默认目录: {directory}")
        
        try:
            offset = float(input("请输入时间偏移量（秒，正数向后延迟，负数向前提前）: "))
        except ValueError:
            print("输入格式错误，使用默认值 0 秒")
            offset = 0
        
        try:
            adjust_directory_srt_files(directory, offset)
        except Exception as e:
            print(f"错误: {str(e)}")
    else:
        main()
