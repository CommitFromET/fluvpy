
"""
循环执行A_fluvpy_main.py的简单脚本
"""

import subprocess
import sys
import time


def run_fluvpy_multiple_times(num_iterations):
    """
    循环执行A_fluvpy_main.py指定的次数

    参数:
        num_iterations: 循环执行的次数
    """

    print(f"开始循环执行main.py，共{num_iterations}次")

    # 记录开始时间
    start_time = time.time()

    # 循环执行指定次数
    for i in range(num_iterations):
        print(f"\n==== 第 {i+1}/{num_iterations} 次执行 ====")

        # 使用subprocess调用外部程序
        try:
            # 执行A_fluvpy_main_TI64-64-25.py
            subprocess.run(["python", "main.py"], check=True)
            print(f"第 {i+1} 次执行完成")
        except subprocess.CalledProcessError as e:
            print(f"执行出错: {e}")
        except FileNotFoundError:
            print("错误: 找不到main.py文件，请确保文件存在且路径正确")
            break

    # 计算总耗时
    total_time = time.time() - start_time
    print(f"\n循环执行完毕，总共执行了{num_iterations}次")
    print(f"总耗时: {total_time:.2f}秒")

if __name__ == "__main__":
    # 从命令行参数获取循环次数，默认为1次
    iterations = 10

    if len(sys.argv) > 1:
        try:
            iterations = int(sys.argv[1])
        except ValueError:
            print("错误: 请输入有效的循环次数(整数)")
            sys.exit(1)

    # 执行循环
    run_fluvpy_multiple_times(iterations)