import os
import sys
import math
import struct
import heapq
import tempfile
import shutil
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor
import re
def read_ints_from_txt(filename):
    """从txt文件读取所有整数（支持大文件流式读取）"""
    ints = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                ints.append(int(line))
    return ints

INT_SIZE = 4           # 32位整数 = 4字节
SIGNED_INT_FMT = '<I'  # 小端有符号32位整数


def parse_args():
    """解析命令行参数：文件名，内存允许的整数个数M（可选）"""
    filename = sys.argv[1]

    # 如果提供了 M 参数，使用它；否则设为极大值（不限制）
    if len(sys.argv) >= 3:
        M = int(sys.argv[2])
    else:
        M = 10 ** 12  # 极大值，相当于不限制内存
        print(f"未指定 M，将使用默认值：{M}（相当于不限制内存）")

    # 自动检测文件类型并转换
    if filename.endswith('.txt'):
        print(f"检测到txt文件，正在转换为二进制格式...")
        bin_filename = filename.replace('.txt', '.bin')

        # 读取txt并写入bin
        ints = read_ints_from_txt(filename)
        with open(bin_filename, 'wb') as f:
            for n in ints:
                f.write(struct.pack('<I', n))  # 使用无符号
        print(f"转换完成！共{len(ints)}个整数，保存为{bin_filename}")
        filename = bin_filename

    return filename, M

def get_total_int_count(filename):
    """通过文件大小计算整数的总个数"""
    size = os.path.getsize(filename)
    if size % INT_SIZE != 0:
        raise ValueError("File size is not a multiple of 4 bytes")
    return size // INT_SIZE

def process_chunk(args):
    """
    处理一个数据块：读取、排序、写入临时文件。
    args: (chunk_index, start_byte, num_ints, orig_filename, temp_dir)
    """
    idx, start_byte, num_ints, orig_filename, temp_dir = args
    # 读取块内所有整数
    with open(orig_filename, 'rb') as f:
        f.seek(start_byte)
        data = f.read(num_ints * INT_SIZE)
    # 解包为整数列表
    ints = list(struct.unpack(f'<{num_ints}I', data))
    ints.sort()                     # 排序
    # 写入临时文件
    tmp_path = os.path.join(temp_dir, f"chunk_{idx:08d}.bin")
    with open(tmp_path, 'wb') as f:
        # 打包并写入
        packed = struct.pack(f'<{num_ints}I', *ints)
        f.write(packed)
    return tmp_path

def parallel_sort_chunks(orig_filename, total_ints, M, temp_dir):
    """
    并行对所有块进行排序，返回生成的临时文件路径列表。
    """
    num_chunks = math.ceil(total_ints / M)
    # 准备任务参数
    tasks = []
    for i in range(num_chunks):
        start_byte = i * M * INT_SIZE
        num_ints = min(M, total_ints - i * M)
        tasks.append((i, start_byte, num_ints, orig_filename, temp_dir))
    # 使用进程池并行处理
    with Pool(processes=cpu_count()) as pool:
        chunk_files = pool.map(process_chunk, tasks)
    return chunk_files  # 已经是排序后的列表（按索引顺序）

# ------------------------------------------------------------
# 归并阶段所需组件
# ------------------------------------------------------------
class BufferedIntReader:
    """带缓冲的整数读取器，按小端序从二进制文件中读取32位整数"""
    def __init__(self, filepath, buffer_ints):
        """
        filepath   : 文件路径
        buffer_ints: 内部缓冲区中最多存放的整数个数（控制内存）
        """
        self.f = open(filepath, 'rb')
        self.buffer_size = buffer_ints * INT_SIZE
        self.buffer = b''
        self.pos = 0
        self.eof = False

    def read_int(self):
        """返回下一个整数，若无数据则返回None"""
        if self.pos + INT_SIZE > len(self.buffer):
            if self.eof:
                return None
            # 读取下一块数据
            data = self.f.read(self.buffer_size)
            if not data:
                self.eof = True
                return None
            self.buffer = data
            self.pos = 0
            # 递归调用自身（新数据至少包含一个整数？不一定，可能文件结束但data长度不足4）
            return self.read_int()
        else:
            val = struct.unpack(SIGNED_INT_FMT, self.buffer[self.pos:self.pos+INT_SIZE])[0]
            self.pos += INT_SIZE
            return val

    def close(self):
        self.f.close()

def merge_one_group(input_files, output_file, M):
    """
    将一组有序的输入文件归并为一个有序的输出文件。
    内存使用控制在 M 个整数以内。
    """
    K = len(input_files)                     # 归并路数
    if K == 0:
        return
    if K == 1:
        # 直接复制文件（避免无谓归并）
        shutil.copy2(input_files[0], output_file)
        return

    # 计算内存分配：总内存限制 M 个整数
    # 内存占用 = 堆大小(K) + 所有输入缓冲区总大小 + 输出缓冲区大小
    # 设每个输入缓冲区大小为 b 个整数，输出缓冲区大小为 o 个整数
    # 则需满足: K + K*b + o <= M
    # 我们分配: b = max(1, (M - K) // (2*K))  同时保证 o = max(1, (M - K - K*b) // 2)
    # 若 M 不足以支持 K 个文件各占 1 个缓冲区（即 M < 2K），则强制 b = 1, o = 0（此时堆已占用K个整数，额外缓冲区会超，所以实际只能放弃输出缓冲）
    if M < 2 * K:
        # 内存极度紧张：不使用额外缓冲区（每次只读一个整数），也不使用输出缓冲（立即写入）
        b = 1      # 实际上 reader 内部不再保留额外缓冲区（buffer_ints=1 会导致读一个整数但 buffer 还是一次读 4 字节，内存占用很低）
        o = 0
    else:
        b = max(1, (M - K) // (2 * K))
        o = max(1, (M - K - K * b) // 2)

    # 创建所有输入文件的 reader
    readers = []
    for fpath in input_files:
        reader = BufferedIntReader(fpath, b)
        readers.append(reader)

    # 初始化堆：元素为 (value, reader_index)
    heap = []
    for i, r in enumerate(readers):
        val = r.read_int()
        if val is not None:
            heapq.heappush(heap, (val, i))

    # 打开输出文件，准备输出缓冲区
    out_f = open(output_file, 'wb')
    out_buffer = []
    out_buffer_limit = o

    while heap:
        val, i = heapq.heappop(heap)
        out_buffer.append(val)
        # 输出缓冲区满则写入
        if out_buffer_limit > 0 and len(out_buffer) >= out_buffer_limit:
            packed = struct.pack(f'<{len(out_buffer)}I', *out_buffer)
            out_f.write(packed)
            out_buffer.clear()
        # 从同一 reader 读取下一个值
        next_val = readers[i].read_int()
        if next_val is not None:
            heapq.heappush(heap, (next_val, i))

    # 写入剩余的输出缓冲区
    if out_buffer:
        packed = struct.pack(f'<{len(out_buffer)}I', *out_buffer)
        out_f.write(packed)

    # 清理
    out_f.close()
    for r in readers:
        r.close()

def parallel_merge(chunk_files, temp_dir, M):
    """
    多级并行归并，直到只剩下一个文件。
    返回最终文件的路径。
    """
    current_files = chunk_files
    round_num = 0
    # 归并路数限制：不能超过 M，同时考虑文件描述符限制（取较小值）
    max_merge_degree = min(M, 500)   # 500 可根据系统调整
    while len(current_files) > 1:
        print(f"Merge round {round_num}: {len(current_files)} files -> ...")
        # 将文件分组，每组大小不超过 max_merge_degree
        groups = []
        for i in range(0, len(current_files), max_merge_degree):
            groups.append(current_files[i:i+max_merge_degree])
        # 并行归并每一组
        next_files = []
        # 使用 ProcessPoolExecutor 并行执行多个归并任务
        with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            futures = []
            for gidx, group in enumerate(groups):
                output_path = os.path.join(temp_dir, f"merge_r{round_num}_g{gidx:04d}.bin")
                futures.append(executor.submit(merge_one_group, group, output_path, M))
                next_files.append(output_path)
            # 等待所有归并完成
            for fut in futures:
                fut.result()  # 如果有异常会抛出
        # 删除上一轮的临时文件（可选，保留磁盘空间）
        for f in current_files:
            try:
                os.remove(f)
            except OSError:
                pass
        current_files = next_files
        round_num += 1
    # 最终只有一个文件
    return current_files[0]

def main():
    orig_filename, M = parse_args()
    total_ints = get_total_int_count(orig_filename)
    print(f"Total integers: {total_ints}, M = {M}")

    # 创建临时目录（与原始文件同磁盘）
    work_dir = os.path.dirname(orig_filename) or '.'
    temp_dir = tempfile.mkdtemp(prefix="extsort_", dir=work_dir)
    try:
        # 第一阶段：并行分块排序
        print("Phase 1: Splitting and sorting chunks...")
        chunk_files = parallel_sort_chunks(orig_filename, total_ints, M, temp_dir)
        print(f"Created {len(chunk_files)} sorted chunks.")

        # 第二阶段：并行多路归并
        print("Phase 2: Merging chunks...")
        final_sorted_file = parallel_merge(chunk_files, temp_dir, M)
        # 最终文件移动到原始文件所在目录，命名为 "原文件名_sorted.bin"
        base, ext = os.path.splitext(orig_filename)
        result_file = f"{base}_sorted{ext}"
        shutil.move(final_sorted_file, result_file)
        print(f"Sorting completed. Result saved to: {result_file}")
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()