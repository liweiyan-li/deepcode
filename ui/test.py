"""
独立运行测试生成阶段
用于已经生成好代码的项目，只运行测试生成部分
"""

import asyncio
import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workflows.agents.generation_test_agent import TestGenerationAgent


async def run_test_generation_for_existing_code(
    paper_dir: str,
    code_directory: str = None,
    plan_file_path: str = None
):
    """
    为已存在的代码生成测试
    
    Args:
        paper_dir: 论文目录路径，例如 "deepcode_lab/papers/10"
        code_directory: 代码目录路径，如果为 None 则自动检测
        plan_file_path: 实现计划文件路径，如果为 None 则自动检测
    """
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    print("="*80)
    print("🧪 独立测试生成工具")
    print("="*80)
    
    # 验证论文目录
    if not os.path.exists(paper_dir):
        print(f"❌ 错误: 论文目录不存在: {paper_dir}")
        return
    
    print(f"📂 论文目录: {paper_dir}")
    
    # 自动检测代码目录
    if code_directory is None:
        possible_dirs = [
            os.path.join(paper_dir, "generate_code"),
            os.path.join(paper_dir, "code"),
            os.path.join(paper_dir, "implementation"),
        ]
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                code_directory = dir_path
                print(f"✅ 自动检测到代码目录: {code_directory}")
                break
        
        if code_directory is None:
            print(f"❌ 错误: 无法找到代码目录，请在以下位置之一创建代码:")
            for dir_path in possible_dirs:
                print(f"   - {dir_path}")
            return
    
    if not os.path.exists(code_directory):
        print(f"❌ 错误: 代码目录不存在: {code_directory}")
        return
    
    # 自动检测实现计划文件
    if plan_file_path is None:
        possible_files = [
            os.path.join(paper_dir, "initial_plan.txt"),
            os.path.join(paper_dir, "plan.txt"),
            os.path.join(paper_dir, "implementation_plan.txt"),
        ]
        for file_path in possible_files:
            if os.path.exists(file_path):
                plan_file_path = file_path
                print(f"✅ 自动检测到计划文件: {plan_file_path}")
                break
        
        if plan_file_path is None:
            print(f"⚠️  警告: 无法找到实现计划文件，将继续但测试质量可能降低")
            print(f"   建议在以下位置之一创建计划文件:")
            for file_path in possible_files:
                print(f"   - {file_path}")
            # 创建一个临时的空计划文件
            plan_file_path = os.path.join(paper_dir, "initial_plan.txt")
            with open(plan_file_path, "w", encoding="utf-8") as f:
                f.write("# Placeholder implementation plan\n")
    
    print(f"📋 实现计划: {plan_file_path}")
    print(f"🎯 代码目录: {code_directory}")
    print()
    
    try:
        # 创建测试生成智能体
        print("🤖 初始化测试生成智能体...")
        async with TestGenerationAgent(logger=logger) as test_agent:
            print("✅ 测试智能体初始化成功")
            print()
            
            # 生成测试
            print("🔄 开始生成测试套件...")
            print("-"*80)
            
            test_summary = await test_agent.generate_tests(
                code_directory=code_directory,
                plan_file_path=plan_file_path,
                paper_dir=paper_dir,
            )
            
            print("-"*80)
            print()
            
            # 保存测试报告
            test_report_path = os.path.join(paper_dir, "test_generation_report.txt")
            with open(test_report_path, "w", encoding="utf-8") as f:
                f.write(test_summary.get("raw_result", str(test_summary)))
            
            print("="*80)
            print("✅ 测试生成完成!")
            print("="*80)
            print(f"📁 测试目录: {test_summary.get('test_directory', 'N/A')}")
            print(f"📄 测试报告: {test_report_path}")
            print(f"📊 状态: {test_summary.get('status', 'unknown')}")
            print(f"💬 消息: {test_summary.get('message', 'N/A')}")
            print("="*80)
            
            return test_summary
            
    except Exception as e:
        print()
        print("="*80)
        print("❌ 测试生成失败!")
        print("="*80)
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        print("="*80)
        return None


def main():
    """主函数"""
    # 默认配置 - 可以修改这里的路径
    DEFAULT_PAPER_DIR = "deepcode_lab/papers/10"
    
    print()
    print("🧪 DeepCode 独立测试生成工具")
    print()
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        paper_dir = sys.argv[1]
    else:
        paper_dir = DEFAULT_PAPER_DIR
        print(f"使用默认论文目录: {paper_dir}")
        print(f"提示: 可以通过命令行参数指定其他目录")
        print(f"   python run_test_generation_only.py <paper_dir>")
        print()
    
    # 转换为绝对路径
    if not os.path.isabs(paper_dir):
        paper_dir = os.path.join(os.getcwd(), paper_dir)
    
    # 运行测试生成
    asyncio.run(run_test_generation_for_existing_code(paper_dir))


if __name__ == "__main__":
    main()