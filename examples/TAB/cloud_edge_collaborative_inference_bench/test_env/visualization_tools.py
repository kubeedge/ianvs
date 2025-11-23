<<<<<<< HEAD
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import from_csv

__all__ = ["generate_report"]

class VisualizationTools:
    def __init__(self, output_dir="./results"):
        self.output_dir = output_dir
        self.rank_dir = os.path.join(output_dir, "rank") 
        os.makedirs(self.rank_dir, exist_ok=True)
    
    def plot_privacy_metrics(self, metrics_data):
        
        df = pd.DataFrame(metrics_data)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=df.melt(id_vars=['method']), 
                    x='method', y='value', hue='variable')
        plt.title('Privacy Metrics Comparison Across Methods')
        plt.xlabel('Privacy Protection Method')
        plt.ylabel('Score')
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'privacy_metrics.png'))
        plt.close()
    
    def plot_performance_metrics(self, performance_data):
        
        df = pd.DataFrame(performance_data)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        
        sns.barplot(data=df, x='method', y='lo', ax=axes[0])
        axes[0].set_title('Latency Overhead (%)')
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
        
        
        sns.barplot(data=df, x='method', y='apr', ax=axes[1])
        axes[1].set_title('Accuracy Preservation Rate')
        axes[1].set_ylim(0, 1.0)
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
        
       
        sns.barplot(data=df, x='method', y='ti', ax=axes[2])
        axes[2].set_title('Throughput Impact (%)')
        axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_metrics.png'))
        plt.close()
    
    def plot_privacy_vs_utility(self, metrics_data):
   
        df = pd.DataFrame(metrics_data)
        
        
        df['privacy_score'] = (df['pdr'] + (1 - df['sels']) + df['iar'] + df['cpp']) / 4
        df['utility_score'] = df['cpp']
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=df, x='privacy_score', y='utility_score', 
                       hue='method', s=100)
        plt.title('Privacy vs Utility Trade-off')
        plt.xlabel('Privacy Score (Higher is Better)')
        plt.ylabel('Utility Score (Higher is Better)')
        plt.xlim(0, 1.0)
        plt.ylim(0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'privacy_vs_utility.png'))
        plt.close()

    def _sort_metrics_df(self, df, sort_by):
        sort_columns = []
        ascending = []
        for item in sort_by:
            metric = next(iter(item))
            sort_columns.append(metric)
            ascending.append(item[metric] == "ascend")
        return df.sort_values(by=sort_columns, ascending=ascending)

    def save_metrics_to_csv(self, metrics_df, filename):
        csv_path = os.path.join(self.rank_dir, filename)
        metrics_df.index = range(1, len(metrics_df) + 1)
        metrics_df.index.name = "rank"
        metrics_df.to_csv(csv_path, encoding="utf-8")
        return csv_path

    def generate_table_report(self, csv_path, report_filename):
        table_path = os.path.join(self.rank_dir, report_filename)
        with open(csv_path, "r", encoding="utf-8") as f_in, \
             open(table_path, "w", encoding="utf-8") as f_out:
            table = from_csv(f_in)
            f_out.write(str(table))
        return table_path


def generate_report(all_metrics):
    viz = VisualizationTools()
    
 
    privacy_metrics = []
    for method, metrics in all_metrics.items():
        privacy_metrics.append({
            'method': method,
            'pdr': metrics['pdr'],
            'sels': metrics['sels'],
            'iar': metrics['iar'],
            'cpp': metrics['cpp']
        })
    privacy_df = pd.DataFrame(privacy_metrics)
    
    performance_data = []
    for method, metrics in all_metrics.items():
        performance_data.append({
            'method': method,
            'lo': metrics['lo'],
            'apr': metrics['apr'],
            'ti': metrics['ti']
        })
    performance_df = pd.DataFrame(performance_data)

    combined_df = pd.merge(privacy_df, performance_df, on='method')
    

    combined_df['privacy_score'] = (combined_df['pdr'] + (1 - combined_df['sels']) + 
                                   combined_df['iar'] + combined_df['cpp']) / 4
    sorted_df = viz._sort_metrics_df(combined_df, [
        {'privacy_score': 'descend'},
        {'apr': 'descend'}
    ])
    

    all_rank_csv = viz.save_metrics_to_csv(sorted_df, "all_metrics_rank.csv")
    

    viz.generate_table_report(all_rank_csv, "test_report.txt")
    

    viz.plot_privacy_metrics(privacy_metrics)
    viz.plot_performance_metrics(performance_data)
    viz.plot_privacy_vs_utility(privacy_metrics)
    
    return viz.output_dir
=======
version https://git-lfs.github.com/spec/v1
oid sha256:64646a959fe80a1c2c0acbc64b1f6961ca737c5da78d178897a7e064e9842b61
size 5066
>>>>>>> 9676c3e (ya toh aar ya toh par)
