import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_class_performance(metrics_df, save_path="class_performance.png"):
    """Plot per-class performance metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Precision plot
    axes[0, 0].bar(metrics_df['Class'], metrics_df['Precision'], color='skyblue')
    axes[0, 0].set_title('Precision by Class')
    axes[0, 0].set_xlabel('Domino Class')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Recall plot
    axes[0, 1].bar(metrics_df['Class'], metrics_df['Recall'], color='lightgreen')
    axes[0, 1].set_title('Recall by Class')
    axes[0, 1].set_xlabel('Domino Class')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # F1-Score plot
    axes[1, 0].bar(metrics_df['Class'], metrics_df['F1-Score'], color='salmon')
    axes[1, 0].set_title('F1-Score by Class')
    axes[1, 0].set_xlabel('Domino Class')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Support plot
    axes[1, 1].bar(metrics_df['Class'], metrics_df['Support'], color='gold')
    axes[1, 1].set_title('Number of Test Images by Class')
    axes[1, 1].set_xlabel('Domino Class')
    axes[1, 1].set_ylabel('Number of Images')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Class performance plots saved to: {save_path}")
    plt.show()

def plot_accuracy_summary(test_accuracy, save_path="accuracy_summary.png"):
    """Plot accuracy summary."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy bar plot
    categories = ['Test Accuracy']
    accuracies = [test_accuracy * 100]
    colors = ['green' if acc >= 90 else 'orange' if acc >= 80 else 'red' for acc in accuracies]
    
    bars = ax1.bar(categories, accuracies, color=colors, alpha=0.7)
    ax1.set_title('Model Test Accuracy')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(0, 100)
    
    # Add percentage text on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # Pie chart for correct vs incorrect predictions
    correct_pct = test_accuracy * 100
    incorrect_pct = 100 - correct_pct
    
    ax2.pie([correct_pct, incorrect_pct], 
            labels=[f'Correct\n({correct_pct:.1f}%)', f'Incorrect\n({incorrect_pct:.1f}%)'],
            colors=['lightgreen', 'lightcoral'],
            autopct='%1.1f%%',
            startangle=90)
    ax2.set_title('Prediction Distribution')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Accuracy summary saved to: {save_path}")
    plt.show()

def save_results_to_file(metrics_df, test_accuracy, test_loss, total_images, save_path="test_results.txt"):
    """Save all test results to a text file."""
    with open(save_path, 'w') as f:
        f.write("DOMINO TILE CNN MODEL TEST RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Overall Performance:\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Total Images Tested: {total_images}\n\n")
        
        f.write("Per-Class Performance:\n")
        f.write("-" * 30 + "\n")
        for _, row in metrics_df.iterrows():
            f.write(f"{row['Class']:>6}: Precision={row['Precision']:.3f}, "
                   f"Recall={row['Recall']:.3f}, F1={row['F1-Score']:.3f}, "
                   f"Support={row['Support']}\n")
        
        f.write(f"\nSummary Statistics:\n")
        f.write(f"Mean Precision: {metrics_df['Precision'].mean():.3f}\n")
        f.write(f"Mean Recall: {metrics_df['Recall'].mean():.3f}\n")
        f.write(f"Mean F1-Score: {metrics_df['F1-Score'].mean():.3f}\n")
        
        best_f1 = metrics_df.loc[metrics_df['F1-Score'].idxmax()]
        worst_f1 = metrics_df.loc[metrics_df['F1-Score'].idxmin()]
        
        f.write(f"\nBest performing class: {best_f1['Class']} (F1={best_f1['F1-Score']:.3f})\n")
        f.write(f"Worst performing class: {worst_f1['Class']} (F1={worst_f1['F1-Score']:.3f})\n")
    
    print(f"Test results saved to: {save_path}")

# Usage example - add this to your main() function after getting the results:
def save_all_results(metrics_df, test_accuracy, test_loss, total_images):
    """Save all plots and results."""
    print("\nSaving all results...")
    
    # Save performance plots
    plot_class_performance(metrics_df, "domino_class_performance.png")
    
    # Save accuracy summary
    plot_accuracy_summary(test_accuracy, "domino_accuracy_summary.png")
    
    # Save text results
    save_results_to_file(metrics_df, test_accuracy, test_loss, total_images, "domino_test_results.txt")
    
    print("All results saved successfully!")