"""递归遍历文件夹下所有 h5 文件，打印结构并汇总。

用法:
    python inspect_h5.py <folder> [--max N] [--verbose]
"""

import argparse
import sys
import threading
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import h5py
sys.stdout.reconfigure(encoding='utf-8')
_print_lock = threading.Lock()


def _fmt_attr_value(v):
    """压缩属性值的显示：裁剪长 ndarray / bytes。"""
    import numpy as np
    if isinstance(v, bytes):
        try:
            v = v.decode("utf-8", errors="replace")
        except Exception:
            pass
    if isinstance(v, np.ndarray):
        if v.size > 8:
            return f"ndarray shape={v.shape} dtype={v.dtype} (truncated)"
        v = v.tolist()
    s = repr(v)
    return s if len(s) <= 200 else s[:197] + "..."


def _dump_attrs(obj, indent):
    lines = []
    for k in obj.attrs:
        try:
            val = obj.attrs[k]
        except Exception as e:
            lines.append(f"{indent}@{k} = <读取失败: {e}>")
            continue
        lines.append(f"{indent}@{k} = {_fmt_attr_value(val)}")
    return lines


def walk_h5(file_path: Path, verbose: bool = True):
    """返回 (结构签名, 数据集信息列表)。"""
    datasets = []  # list of (name, shape, dtype)
    dataset_details = []  # for verbose printing
    group_attrs = []  # (name, lines)
    attrs_count = 0

    def visit(name, obj):
        nonlocal attrs_count
        attrs_count += len(obj.attrs)
        if isinstance(obj, h5py.Dataset):
            datasets.append((name, tuple(obj.shape), str(obj.dtype)))
            dataset_details.append((name, obj))
        elif isinstance(obj, h5py.Group) and len(obj.attrs) > 0:
            group_attrs.append((name, _dump_attrs(obj, "      ")))

    with h5py.File(file_path, "r") as f:
        attrs_count += len(f.attrs)

        # --- 文件头 / 全局信息 ---
        header_lines = []
        if verbose:
            try:
                fsize = file_path.stat().st_size
                header_lines.append(f"  file size    : {fsize:,} bytes ({fsize/1024/1024:.2f} MB)")
            except Exception:
                pass
            try:
                header_lines.append(f"  libver       : {f.libver}")
            except Exception:
                pass
            try:
                header_lines.append(f"  driver       : {f.driver}")
            except Exception:
                pass
            try:
                header_lines.append(f"  userblock    : {f.userblock_size} bytes")
            except Exception:
                pass
            try:
                header_lines.append(f"  root members : {len(f.keys())}  ({', '.join(list(f.keys())[:10])}{'...' if len(f.keys()) > 10 else ''})")
            except Exception:
                pass
            # 根属性
            if len(f.attrs) > 0:
                header_lines.append(f"  root attrs ({len(f.attrs)}):")
                header_lines.extend(_dump_attrs(f, "    "))

        f.visititems(visit)

        # 渲染 dataset 详情（在文件仍打开时取 chunks/compression）
        ds_lines = []
        if verbose:
            for name, obj in dataset_details:
                extra = []
                if obj.chunks:
                    extra.append(f"chunks={obj.chunks}")
                if obj.compression:
                    c = obj.compression
                    if obj.compression_opts is not None:
                        c = f"{c}({obj.compression_opts})"
                    extra.append(f"compression={c}")
                extra_s = ("  " + "  ".join(extra)) if extra else ""
                ds_lines.append(f"    /{name}  shape={tuple(obj.shape)}  dtype={obj.dtype}{extra_s}")
                # 该 dataset 的属性
                for al in _dump_attrs(obj, "        "):
                    ds_lines.append(al)

    if verbose:
        lines = [f"\n=== {file_path} ==="]
        lines.extend(header_lines)
        lines.append(f"  datasets: {len(datasets)}, total attrs: {attrs_count}")
        lines.extend(ds_lines)
        for gname, galines in group_attrs:
            lines.append(f"    group /{gname} attrs:")
            lines.extend(galines)
        with _print_lock:
            print("\n".join(lines))

    # 结构签名：用于分组相同结构的文件
    signature = tuple(sorted((n, s, d) for n, s, d in datasets))
    return signature, datasets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="要递归扫描的根目录")
    ap.add_argument("--max", type=int, default=0,
                    help="最多处理多少个文件（0 = 不限）")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="不打印每个文件的结构，只输出汇总")
    ap.add_argument("-j", "--workers", type=int, default=8,
                    help="并发线程数（默认 8）")
    args = ap.parse_args()

    root = Path(args.folder)
    if not root.is_dir():
        print(f"错误: {root} 不是目录", file=sys.stderr)
        sys.exit(1)

    files = sorted(root.rglob("*.h5")) + sorted(root.rglob("*.hdf5"))
    if args.max > 0:
        files = files[: args.max]

    print(f"找到 {len(files)} 个 h5 文件，根目录: {root}")

    groups = defaultdict(list)  # signature -> [file paths]
    failed = []
    done = 0
    total = len(files)

    def work(fp):
        return fp, walk_h5(fp, verbose=not args.quiet)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [ex.submit(work, fp) for fp in files]
        for fut in as_completed(futures):
            done += 1
            try:
                fp, (sig, _) = fut.result()
                groups[sig].append(fp)
            except Exception as e:
                failed.append((str(e),))
                with _print_lock:
                    print(f"  [读取失败] {e}", file=sys.stderr)
            if args.quiet and done % 20 == 0:
                with _print_lock:
                    print(f"  进度: {done}/{total}", file=sys.stderr)

    # 汇总
    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    print(f"总文件数: {len(files)}   成功: {len(files) - len(failed)}   失败: {len(failed)}")
    print(f"不同结构数: {len(groups)}")

    for i, (sig, paths) in enumerate(
        sorted(groups.items(), key=lambda kv: -len(kv[1])), start=1
    ):
        print(f"\n--- 结构 #{i}  ({len(paths)} 个文件) ---")
        if sig:
            for name, shape, dtype in sig:
                print(f"  /{name}  shape={shape}  dtype={dtype}")
        else:
            print("  (无 dataset)")
        sample = paths[0]
        print(f"  示例: {sample}")
        if len(paths) > 1:
            print(f"  其余 {len(paths) - 1} 个文件结构相同")

    if failed:
        print(f"\n失败文件数: {len(failed)}")


if __name__ == "__main__":
    main()
