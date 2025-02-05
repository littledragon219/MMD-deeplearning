preprocessing：
1.make sure config.py and preprocess.py are in same index
2.if have subfolder,needa use raw_dir.iterdir()；补充说明：若文件直接放在目录中而不是子目录中，代码会出错。因此需要同时考虑两种

for item in load_path.iterdir():
    if item.is_file() and item.suffix == ".mat":  # 直接读取 .mat 文件
        mat_files = [item]
    elif item.is_dir():
        mat_files = list(item.glob("*.mat"))
    else:
        continue
    
    for mat_file in mat_files:
        print(f"Processing file: {mat_file}")  # 检查是否获取到 mat 文件

3.前的`process_mat_to_excel`函数将每个键值对保存为单独的Excel文件，这显然不符合需求。需要重新设计该函数，使其能够分割信号，并将样本逐行保存到Excel中。
4.不确定取的数据是否正确，虽然window size，sample size等公式都对了，但是load data 的时候似乎对于选择0hp，1hp...等存在一定不理解
5.务必要一直记得优化代码的可读性
6.迁移任务（如 T_{03} 表示从 0hp 迁移到 3hp）对于source_load and target_load写的组合
31/1:
已成功预处理所有数据，唯对于是否需要标准化，分割数据（source，target domain）和数据点存在疑问，待解决中
2/2：
已成功分割数据，包括学会了迁移任务对于source_load and target_load 的基础知识。对于两者的样本量不一致存在疑问，待解决中