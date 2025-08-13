# python scripts/analyze_data_distribution.py /home/xieminhui/yinj/workplace/recsys-examples/examples/hstu/tmp_data/KuaiRand-Pure/data/log_standard_4_08_to_4_21_pure.csv /home/xieminhui/yinj/workplace/recsys-examples/examples/hstu/tmp_data/KuaiRand-Pure/data/log_standard_4_22_to_5_08_pure.csv --delta_t 3600000 --output_dir /home/xieminhui/yinj/workplace/recsys-examples/data/data_distribution/kuairand_pure --plot
# python scripts/analyze_data_distribution.py /home/xieminhui/yinj/workplace/recsys-examples/examples/hstu/tmp_data/KuaiRand-1K/data/log_standard_4_08_to_4_21_1k.csv /home/xieminhui/yinj/workplace/recsys-examples/examples/hstu/tmp_data/KuaiRand-1K/data/log_standard_4_22_to_5_08_1k.csv --delta_t 3600000 --output_dir /home/xieminhui/yinj/workplace/recsys-examples/data/data_distribution/kuairand_1k --plot
# python scripts/analyze_data_distribution.py /home/xieminhui/yinj/workplace/recsys-examples/examples/hstu/tmp_data/KuaiRand-27K/data/log_standard_4_08_to_4_21_27k_part1.csv /home/xieminhui/yinj/workplace/recsys-examples/examples/hstu/tmp_data/KuaiRand-27K/data/log_standard_4_08_to_4_21_27k_part2.csv /home/xieminhui/yinj/workplace/recsys-examples/examples/hstu/tmp_data/KuaiRand-27K/data/log_standard_4_22_to_5_08_27k_part1.csv /home/xieminhui/yinj/workplace/recsys-examples/examples/hstu/tmp_data/KuaiRand-27K/data/log_standard_4_22_to_5_08_27k_part2.csv --delta_t 3600000 --output_dir /home/xieminhui/yinj/workplace/recsys-examples/data/data_distribution/kuairand_27k --plot
# python scripts/analyze_data_distribution.py /home/xieminhui/yinj/workplace/recsys-examples/examples/hstu/tmp_data/KuaiRand-1K/data/log_standard_4_08_to_4_21_1k.csv /home/xieminhui/yinj/workplace/recsys-examples/examples/hstu/tmp_data/KuaiRand-1K/data/log_standard_4_22_to_5_08_1k.csv --delta_t 60000 --output_dir /home/xieminhui/yinj/workplace/recsys-examples/data/data_distribution/kuairand_1k --plot
# PYTHONUNBUFFERED=1 python -u scripts/analyze_data_distribution.py   /home/xieminhui/yinj/workplace/recsys-examples/examples/hstu/tmp_data/KuaiRand-1K/data/log_standard_4_08_to_4_21_1k.csv   /home/xieminhui/yinj/workplace/recsys-examples/examples/hstu/tmp_data/KuaiRand-1K/data/log_standard_4_22_to_5_08_1k.csv   --delta_t 60000   --output_dir /home/xieminhui/yinj/workplace/recsys-examples/data/data_distribution/kuairand_1k   --plot   > /home/xieminhui/yinj/workplace/recsys-examples/data/running_log/kuairand_1k_1min.log 2>&1
# PYTHONUNBUFFERED=1 python -u scripts/analyze_data_distribution.py /home/xieminhui/yinj/workplace/recsys-examples/examples/hstu/tmp_data/KuaiRand-27K/data/log_standard_4_08_to_4_21_27k_part1.csv /home/xieminhui/yinj/workplace/recsys-examples/examples/hstu/tmp_data/KuaiRand-27K/data/log_standard_4_08_to_4_21_27k_part2.csv /home/xieminhui/yinj/workplace/recsys-examples/examples/hstu/tmp_data/KuaiRand-27K/data/log_standard_4_22_to_5_08_27k_part1.csv /home/xieminhui/yinj/workplace/recsys-examples/examples/hstu/tmp_data/KuaiRand-27K/data/log_standard_4_22_to_5_08_27k_part2.csv --delta_t 3600000 --output_dir /home/xieminhui/yinj/workplace/recsys-examples/data/data_distribution/kuairand_27k --plot   > /home/xieminhui/yinj/workplace/recsys-examples/data/running_log/kuairand_27k_1h.log 2>&1
# PYTHONUNBUFFERED=1 python -u scripts/analyze_data_distribution.py   /home/xieminhui/yinj/workplace/recsys-examples/examples/hstu/tmp_data/KuaiRand-1K/data/log_standard_4_08_to_4_21_1k.csv   /home/xieminhui/yinj/workplace/recsys-examples/examples/hstu/tmp_data/KuaiRand-1K/data/log_standard_4_22_to_5_08_1k.csv   --delta_t 3600000   --output_dir /home/xieminhui/yinj/workplace/recsys-examples/data/data_distribution/kuairand_1k --plot --force_analyze   > /home/xieminhui/yinj/workplace/recsys-examples/data/running_log/kuairand_1k_1h.log 2>&1
# PYTHONUNBUFFERED=1 python -u scripts/analyze_data_distribution.py   /home/xieminhui/yinj/workplace/recsys-examples/examples/hstu/tmp_data/KuaiRand-1K/data/log_standard_4_08_to_4_21_1k.csv   /home/xieminhui/yinj/workplace/recsys-examples/examples/hstu/tmp_data/KuaiRand-1K/data/log_standard_4_22_to_5_08_1k.csv   --delta_t 60000   --output_dir /home/xieminhui/yinj/workplace/recsys-examples/data/data_distribution/kuairand_1k --plot --force_analyze   > /home/xieminhui/yinj/workplace/recsys-examples/data/running_log/kuairand_1k_1min.log 2>&1

import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os
from collections import defaultdict
import matplotlib as mpl

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='User Interaction Data Analysis and Visualization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('input_files', nargs='+', 
                      help='Path to input CSV file(s) (support multiple files)')
    parser.add_argument('--delta_t', type=int, default=60000,
                      help='Time window size in milliseconds')
    parser.add_argument('--output_dir', default='./results',
                      help='Output directory path')
    parser.add_argument('--output_prefix', default='interaction_stats',
                      help='Prefix for output filenames')
    parser.add_argument('--time_col', default='time_ms',
                      help='Timestamp column name')
    parser.add_argument('--user_col', default='user_id',
                      help='User ID column name')
    parser.add_argument('--force_analyze', action='store_true',
                      help='Force re-run data analysis (skip file check)')
    parser.add_argument('--plot', action='store_true',
                      help='Generate visualization plots')
    
    return parser.parse_args()

def load_and_combine_files(file_paths):
    """Load and combine multiple CSV files with validation"""
    dfs = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        print(f"Loading {os.path.basename(file_path)}...", end=' ', flush=True)
        df = pd.read_csv(file_path)
        dfs.append(df)
        print(f"loaded {len(df):,} records")
    
    combined_df = pd.concat(dfs, ignore_index=True).sort_values(time_col)
    print(f"\nCombined dataset: {len(combined_df):,} total records")
    return combined_df, total_records

def calculate_metrics(df, user_col, cumulative_data=None):
    """Calculate key metrics including per-window averages"""
    metrics = {
        'total_interactions': len(df),
        'unique_users': df[user_col].nunique(),
        'avg_interactions_per_user': len(df) / df[user_col].nunique() if df[user_col].nunique() > 0 else 0,
        'interaction_counts': df[user_col].value_counts().to_dict(),
    }
    metrics['max_interactions_per_user'] = max(metrics['interaction_counts'].values()) if metrics['interaction_counts'] else 0

    # 新增：计算平均用户总交互次数
    if cumulative_data is None:
        user_total_interactions = df[user_col].value_counts()
        metrics['avg_total_interactions_per_user'] = user_total_interactions.mean()
        # 新增：计算所有用户中总交互次数的最大值
        metrics['max_total_interactions_per_user'] = user_total_interactions.max() if not user_total_interactions.empty else 0
    else:
        metrics['avg_total_interactions_per_user'] = 0
        metrics['max_total_interactions_per_user'] = 0 # Initialize even if no cumulative data
    return metrics

def analyze_data(df, delta_t_ms, time_col, user_col):
    """Analyze data with progress display"""
    print("\n🔍 Starting data analysis...")
    
    # Data preprocessing
    print("⏳ Preprocessing data (time conversion, sorting)...", end=' ', flush=True)
    df['timestamp'] = pd.to_datetime(df[time_col], unit='ms')
    df = df.sort_values('timestamp')
    min_time = df['timestamp'].min()
    df['time_since_start'] = (df['timestamp'] - min_time).dt.total_seconds() * 1000
    total_records = len(df)
    print(f"Done! Total {total_records:,} records")
    
    # Initialize time windows
    max_time = df['time_since_start'].max()
    time_windows = np.arange(0, max_time + delta_t_ms, delta_t_ms)
    total_windows = len(time_windows)
    print(f"⏰ Will analyze {total_windows} time windows ({delta_t_ms/1000:.1f}s each)")
    
    window_stats = []
    cumulative_stats = []
    prev_users = 0
    
    # Progress display configuration
    progress_interval = max(1, total_windows // 10)  # Show progress at least 10 times
    
    print("\n📊 Analysis progress:")
    for i, window_end in enumerate(time_windows):
        # Window statistics
        window_mask = ((df['time_since_start'] > (window_end - delta_t_ms)) & 
                      (df['time_since_start'] <= window_end))
        window_data = df[window_mask]
        
        # Cumulative statistics
        cumulative_data = df[df['time_since_start'] <= window_end]
        
        # Calculate metrics
        window_metrics = calculate_metrics(window_data, user_col, cumulative_data)
        cumulative_metrics = calculate_metrics(cumulative_data, user_col)
        
        # Record results
        window_stats.append({'window_end': window_end, **window_metrics})
        cumulative_stats.append({'time_point': window_end, **cumulative_metrics})
        
        # Progress display
        if (i % progress_interval == 0) or (i == total_windows - 1):
            new_users = cumulative_metrics['unique_users'] - prev_users
            progress = (i + 1) / total_windows * 100
            time_elapsed = window_end / 1000
            print(
                f"▏{progress:.0f}% ▏Time: {time_elapsed:.1f}s ▏"
                f"Window interactions: {window_metrics['total_interactions']:4d} ▏"
                f"Window users: {window_metrics['unique_users']:3d} ▏"
                f"Window avg interactions: {window_metrics['avg_interactions_per_user']:4.1f} ▏" 
                f"New users: {new_users:3d} ▏"
                f"Total users: {cumulative_metrics['unique_users']:5d}"
            )
            prev_users = cumulative_metrics['unique_users']
    
    # Final statistics
    print("\n✅ Analysis completed!")
    final_stats = cumulative_stats[-1]
    print(f"• Total time span: {max_time/1000:.1f} seconds")
    print(f"• Total users: {final_stats['unique_users']:,}")
    print(f"• Total interactions: {final_stats['total_interactions']:,}")
    print(f"• Avg interactions/user: {final_stats['avg_interactions_per_user']:.1f}")
    
    return pd.DataFrame(window_stats), pd.DataFrame(cumulative_stats)

def visualize_results(window_df, cumulative_df, output_path):
    """Enhanced visualization with larger fonts and 2x3 layout"""
    # ======================
    # 1. 字体全局设置
    # ======================
    plt.style.use('seaborn-v0_8')
    mpl.rcParams.update({
        'font.size': 14,          # 适当减小字号以适应更多图表
        'axes.titlesize': 20,     # 标题字号
        'axes.labelsize': 18,     # 坐标轴标签
        'xtick.labelsize': 16,    # X轴刻度
        'ytick.labelsize': 16,    # Y轴刻度
        'legend.fontsize': 16,    # 图例字号
    })

    # ======================
    # 2. 创建画布（调整为2行3列）
    # ======================
    fig = plt.figure(figsize=(24, 16), dpi=120)  # 增加宽度以适应3列
    fig.suptitle('User Interaction Analysis', fontsize=24, y=0.98)

    # 计算条形宽度（所有条形图保持一致）
    bar_width = window_df['window_end'].diff().mean()/1000*0.6  # 稍微减小宽度

    # ======================
    # 3. 子图绘制（2行3列布局）
    # ======================
    # 3.1 累计用户数（第1行第1列）
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(cumulative_df['time_point']/1000, 
            cumulative_df['unique_users'], 
            'b-', linewidth=2.5)
    ax1.set_title('Cumulative Unique Users', pad=15)
    ax1.set_xlabel('Time (seconds)', fontsize=16)
    ax1.set_ylabel('User Count', fontsize=16)
    ax1.grid(True, alpha=0.3)
    
    # 3.2 窗口用户数（第1行第2列）
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.bar(window_df['window_end']/1000, 
           window_df['unique_users'],
           width=bar_width,
           color='#FF7F0E', edgecolor='none')
    ax2.set_title('Unique Users per Window', pad=15)
    ax2.set_xlabel('Time (seconds)', fontsize=16)
    ax2.set_ylabel('User Count', fontsize=16)
    ax2.grid(True, axis='y', alpha=0.3)
    
    # 3.3 窗口交互量（第1行第3列）
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.bar(window_df['window_end']/1000, 
           window_df['total_interactions'],
           width=bar_width,
           color='#2CA02C', edgecolor='none')
    ax3.set_title('Interactions per Window', pad=15)
    ax3.set_xlabel('Time (seconds)', fontsize=16)
    ax3.set_ylabel('Interaction Count', fontsize=16)
    ax3.grid(True, axis='y', alpha=0.3)
    
    # 3.4 平均交互次数（第2行第1列）
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.bar(window_df['window_end']/1000, 
            window_df['avg_interactions_per_user'],
            width=bar_width, 
            color='#D62728', edgecolor='none', label='Average Total Interactions')
    # Add line for max_interactions_per_user (within window)
    max_interaction_per_user_values = window_df['max_interactions_per_user']
    time_points_ax4 = window_df['window_end']/1000
    
    ax4.plot(time_points_ax4,
             max_interaction_per_user_values,
             'k--', linewidth=1.5, label='Max Interactions per User (Window)')
    
    # 找到max_interactions_per_user的最高点并标注
    max_idx_ax4 = max_interaction_per_user_values.idxmax()
    max_time_ax4 = time_points_ax4.loc[max_idx_ax4]
    max_value_ax4 = max_interaction_per_user_values.loc[max_idx_ax4]
    ax4.annotate(f'{int(max_value_ax4)}', 
                 xy=(max_time_ax4, max_value_ax4), 
                 xytext=(max_time_ax4 + 0.05 * (time_points_ax4.max() - time_points_ax4.min()), max_value_ax4 + 0.05 * max_value_ax4), # 调整文本位置
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                 fontsize=12, color='black')

    ax4.set_title('Avg & Max Interactions per User per Window', pad=15)
    ax4.set_xlabel('Time (seconds)', fontsize=16)
    ax4.set_ylabel('Per User Window Interactions Count', fontsize=16)
    ax4.grid(True, alpha=0.3)
    ax4.legend() 

    # 3.5 新增：平均用户总交互次数（第2行第2列）
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.bar(cumulative_df['time_point']/1000,
            cumulative_df['avg_total_interactions_per_user'],
            width=bar_width, 
            color='#D62728', edgecolor='none', label='Average Total Interactions') # 添加label
    
    # 添加折线，体现所有用户中总交互次数的最大值
    max_total_interactions_values = cumulative_df['max_total_interactions_per_user']
    time_points_ax5 = cumulative_df['time_point']/1000

    ax5.plot(time_points_ax5,
             max_total_interactions_values,
             'k--', linewidth=1.5, label='Max Total Interactions') # 黑色虚线
    
    # 找到max_total_interactions_per_user的最高点并标注
    max_idx_ax5 = max_total_interactions_values.idxmax()
    max_time_ax5 = time_points_ax5.loc[max_idx_ax5]
    max_value_ax5 = max_total_interactions_values.loc[max_idx_ax5]
    ax5.annotate(f'{int(max_value_ax5)}', 
                 xy=(max_time_ax5, max_value_ax5), 
                 xytext=(max_time_ax5 + 0.05 * (time_points_ax5.max() - time_points_ax5.min()), max_value_ax5 + 0.05 * max_value_ax5), # 调整文本位置
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                 fontsize=12, color='black')

    ax5.set_title('Avg & Max Total Interactions per User', pad=15)
    ax5.set_xlabel('Time (seconds)', fontsize=16)
    ax5.set_ylabel('Per User Total Interactions Count', fontsize=16)
    ax5.grid(True, alpha=0.3)
    ax5.legend() 

    # 3.6 保留为空（第2行第3列）
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.text(0.5, 0.5, 'Additional Metrics\n(To Be Determined)',
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax6.transAxes,
            fontsize=16)
    ax6.set_title('Reserved for Future Use', pad=15)
    ax6.axis('off')  # 隐藏坐标轴

    # ======================
    # 4. 统一设置和输出优化
    # ======================
    # 调整子图间距
    plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=2.0)
    
    # 保存图像
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"📊 Visualization saved with 2x3 layout: {output_path}")

    return output_path

def save_json_distributions(df, output_path):
    """Save distribution data as structured JSON"""
    dist_data = []
    for _, row in df.iterrows():
        dist_data.append({
            'time_point': row['time_point'],
            'user_distribution': {
                'total_users': row['unique_users'],
                'interaction_dist': {
                    str(k): int(v) for k, v in 
                    sorted(row['interaction_counts'].items())
                }
            }
        })
    
    with open(output_path, 'w') as f:
        json.dump(dist_data, f, indent=2)

def check_existing_files(output_dir, output_prefix, delta_t, total_records):
    """Check if analysis files already exist with new naming format"""
    delta_s = delta_t // 1000
    rec_k = total_records // 1000  # Convert to thousands
    file_suffix = f"{delta_s}s_{rec_k}k"
    
    files = {
        'window': os.path.join(output_dir, f"{output_prefix}_window_{file_suffix}.csv"),
        'cumulative': os.path.join(output_dir, f"{output_prefix}_cumulative_{file_suffix}.csv"),
        'json': os.path.join(output_dir, f"{output_prefix}_distributions_{file_suffix}.json"),
        'plot': os.path.join(output_dir, f"{output_prefix}_plot_{file_suffix}.png")
    }
    
    # Check if all required files exist
    required_files = [files['window'], files['cumulative'], files['json']]
    all_exist = all(os.path.exists(f) for f in required_files)
    return files, all_exist

def load_existing_data(output_files):
    """Load existing analysis results"""
    print("Loading existing analysis files...")
    window_df = pd.read_csv(output_files['window'])
    cumulative_df = pd.read_csv(output_files['cumulative'])
    return window_df, cumulative_df

def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 增强的启动信息（保持不变）
    print(f"\n{'='*50}")
    print(f"Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input file(s):")
    for i, file_path in enumerate(args.input_files, 1):
        print(f"  [{i}] {os.path.basename(file_path)}")
    print(f"Time window: {args.delta_t}ms ({args.delta_t/1000:.1f}s)")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")
    print(f"Output prefix: {args.output_prefix}")
    print(f"{'='*50}\n")

    try:
        # 第一步：加载数据并计算总记录数
        print("\n📂 Loading input files:")
        dfs = []
        total_records = 0
        for file_path in args.input_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Input file not found: {file_path}")
            
            df = pd.read_csv(file_path)
            record_count = len(df)
            dfs.append(df)
            total_records += record_count
            print(f"  - {os.path.basename(file_path)}: {record_count:,} records")
        
        combined_df = pd.concat(dfs, ignore_index=True).sort_values(args.time_col)
        print(f"\n📊 Combined dataset: {total_records:,} total records")

        # 第二步：生成带记录数的文件名
        delta_s = args.delta_t // 1000
        rec_k = total_records // 1000  # 转换为千单位
        file_suffix = f"{delta_s}s_{rec_k}k"
        
        output_files = {
            'window': os.path.join(args.output_dir, f"{args.output_prefix}_window_{file_suffix}.csv"),
            'cumulative': os.path.join(args.output_dir, f"{args.output_prefix}_cumulative_{file_suffix}.csv"),
            'json': os.path.join(args.output_dir, f"{args.output_prefix}_distributions_{file_suffix}.json"),
            'plot': os.path.join(args.output_dir, f"{args.output_prefix}_plot_{file_suffix}.png")
        }

        # 第三步：检查是否跳过分析（保持原有清晰提示）
        if not args.force_analyze:
            existing_files, all_exist = check_existing_files(args.output_dir, args.output_prefix, args.delta_t, total_records)
            if all_exist:
                print("\n🔍 Found existing analysis results:")
                for file_type, path in existing_files.items():
                    print(f"  - {file_type.ljust(10)}: {os.path.basename(path)}")
                window_df, cumulative_df = load_existing_data(existing_files)
                print("\n✅ Using existing results (use --force_analyze to re-run analysis)")
            else:
                print("\nℹ️ No complete analysis results found, proceeding with fresh analysis...")
                args.force_analyze = True

        # 第四步：运行分析（保持原有结构）
        if args.force_analyze:
            window_df, cumulative_df = analyze_data(combined_df, args.delta_t, args.time_col, args.user_col)
            
            # 保存结果（增强文件名提示）
            print("\n💾 Saving analysis results:")
            window_df.to_csv(output_files['window'], index=False)
            print(f"  - Window stats: {os.path.basename(output_files['window'])}")
            
            cumulative_df.to_csv(output_files['cumulative'], index=False)
            print(f"  - Cumulative stats: {os.path.basename(output_files['cumulative'])}")
            
            save_json_distributions(cumulative_df, output_files['json'])
            print(f"  - Distribution data: {os.path.basename(output_files['json'])}")
            
            print("✅ Analysis completed and saved")

        # 第五步：可视化（保持原有清晰提示）
        if args.plot:
            print("\n🎨 Generating visualization...")
            plot_path = visualize_results(window_df, cumulative_df, output_files['plot'])
            print(f"  - Saved to: {os.path.basename(plot_path)}")

        # 最终摘要（增强记录数显示）
        print(f"\n{'='*50}")
        print("Operation Summary".center(50))
        print(f"{'='*50}")
        print(f"📅 Time range analyzed: {cumulative_df['time_point'].iloc[-1]/1000:.1f}s")
        print(f"📊 Total records processed: {total_records:,}")
        print(f"👥 Unique users: {cumulative_df['unique_users'].iloc[-1]:,}")
        print(f"🔄 Total interactions: {cumulative_df['total_interactions'].iloc[-1]:,}")
        
        if args.plot:
            print(f"\n🎨 Visualization file: {os.path.abspath(output_files['plot'])}")
        print(f"\n💾 Results location: {os.path.abspath(args.output_dir)}")
        print(f"{'='*50}")

    except Exception as e:
        print(f"\n{'❌'*20}")
        print("Processing Failed!".center(40))
        print(f"{'❌'*20}")
        print(f"Error: {str(e)}")
        print(f"\nDebug info:")
        print(f"- Input files: {[os.path.basename(f) for f in args.input_files]}")
        print(f"- Output dir: {args.output_dir}")
        if 'total_records' in locals():
            print(f"- Loaded records: {total_records:,}")
        raise


if __name__ == "__main__":
    main()