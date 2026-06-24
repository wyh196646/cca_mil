import os
import csv
import argparse


def find_svs_files(folder_path):
    svs_files = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".svs"):
                svs_files.append(file)

    return sorted(svs_files)


def write_csv(result_dict, output_csv):
    folder_names = list(result_dict.keys())
    max_len = max((len(files) for files in result_dict.values()), default=0)

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(folder_names)

        for i in range(max_len):
            row = []
            for folder in folder_names:
                files = result_dict[folder]
                row.append(files[i] if i < len(files) else "")
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="递归统计指定文件夹下的 SVS 文件，并输出为 CSV"
    )

    parser.add_argument(
        "root_dir",
        help="根目录路径"
    )

    parser.add_argument(
        "folders",
        nargs=3,
        help="根目录下的三个文件夹名字"
    )

    parser.add_argument(
        "-o",
        "--output",
        default="svs_files.csv",
        help="输出 CSV 文件名，默认 svs_files.csv"
    )

    args = parser.parse_args()

    result = {}

    for folder_name in args.folders:
        folder_path = os.path.join(args.root_dir, folder_name)

        if not os.path.isdir(folder_path):
            print(f"警告：文件夹不存在，跳过：{folder_path}")
            result[folder_name] = []
            continue

        result[folder_name] = find_svs_files(folder_path)

    write_csv(result, args.output)

    print(f"统计完成，结果已保存到：{args.output}")


if __name__ == "__main__":
    main()
    
    
#python test.py /data2/yuhaowang/WSIFew/TCGA TCGA-KICH  TCGA-KIRC  TCGA-KIRP -o result.csv

