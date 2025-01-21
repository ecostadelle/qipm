import os
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score,
                             recall_score, precision_score)
from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
from qipm import RandomForestClassifier as BSRandomForestClassifier
from qipm import AVALIABLE_DATASETS, get_qipm
from time import time
from scipy.stats import wilcoxon

class DatasetHandler:
    """
    Handles dataset loading and preparation for training and testing.
    """
    def __init__(self, base_path):
        self.base_path = base_path

    def load_data(self, dataset_name, data_type):
        file_path = os.path.join(self.base_path, f"{dataset_name}_{data_type}.feather")
        df = pd.read_feather(file_path)
        X = df.values[:, :-1]
        y = df.values[:, -1]
        return X, y

class ModelTrainer:
    """
    Trains models and evaluates their performance.
    """
    def __init__(self, n_jobs, max_samples):
        self.n_jobs = n_jobs
        self.max_samples = max_samples

    def train_base_model(self, X, y):
        model = SKRandomForestClassifier(
            n_estimators=1000,
            max_features='sqrt',
            n_jobs=self.n_jobs,
            random_state=2,
            max_samples=self.max_samples
        )
        model.fit(X, y)
        return model

    def train_adapted_model(self, X, y, feature_bias):
        model = BSRandomForestClassifier(
            n_estimators=1000,
            max_features='sqrt',
            n_jobs=self.n_jobs,
            random_state=2,
            max_samples=self.max_samples
        )
        model.fit(X, y, feature_bias=feature_bias)
        return model

class Evaluation:
    """
    Evaluates models and saves results.
    """
    def __init__(self, filename):
        self.results = pd.DataFrame(AVALIABLE_DATASETS, 
                                    columns=['dataset', 'experiment']
                                    ).set_index('experiment')
        self.filename = filename

    def save_results(self):
        self.results.to_markdown(f'{self.filename}.md')
        self.results.to_csv(f'{self.filename}.csv')

    def update_results(self, experiment, metric, value):
        self.results.loc[experiment, metric] = value
        self.save_results()

    def compute_metrics(self, y_true, y_pred, zero_division):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred, zero_division=zero_division, average='macro'),
            'precision': precision_score(y_true, y_pred, zero_division=zero_division, average='macro'),
            'recall': recall_score(y_true, y_pred, average='macro')
        }

    def generate_latex_report(self, evaluate, title):
        """
        Generate LaTeX tables for the evaluation results.

        Args:
            evaluate (dict): Metrics to evaluate.
            title (str): Title of the report.

        Returns:
            str: Complete LaTeX content for the report.
        """
        latex_content = []
        
        for ev, metrics in evaluate.items():
            latex_content.append(f"\\begin{{table}}[h!]\n\\centering\n\\caption{{{title} - evaluated by {ev}}}")
            latex_content.append("\\begin{tabular}{lc|cc}\n\\toprule")
            latex_content.append(r"{Dataset} & {ID} & {baseline OOD} & {OOD + QIPM} \\")
            latex_content.append("\\midrule")

            cols = ['dataset'] + metrics
            df_ev = self.results.reset_index()[cols]

            for _, row in df_ev.iterrows():
                dataset = row['dataset']
                id_score = row[metrics[0]]
                baseline_ood = row[metrics[1]]
                ood_qipm = row[metrics[2]]

                latex_content.append(f"{dataset} & {id_score:.2f} & {baseline_ood:.2f} & {ood_qipm:.2f} \\\\")

            latex_content.append("\\bottomrule")

            # Perform statistical test
            df_notna = df_ev.dropna()
            if not df_notna.empty:
                p_value = wilcoxon(df_notna[metrics[1]], df_notna[metrics[2]]).pvalue
                latex_content.append(f"\\multicolumn{{4}}{{l}}{{p-value: {p_value:.3f}}} \\\\")

            latex_content.append("\\end{tabular}")
            latex_content.append("\\end{table}\n")
        
        return "\n".join(latex_content)


def main():
    path = '../feather_one_hot_'
    filename = 'EC7h-RFBS-QIPM-normalizing_trees-f1-macro'

    zero_division = 0

    dataset_handler = DatasetHandler(path)
    evaluator = Evaluation(filename)

    for dataset_name, experiment in AVALIABLE_DATASETS:
        print(f"Processing dataset: {dataset_name}")
        start_time = time()

        X_train, y_train = dataset_handler.load_data(experiment, "train")
        max_samples = 0.1 if experiment == 'acspubcov' else None
        trainer = ModelTrainer(n_jobs=-1, max_samples=max_samples)

        # Train base model
        base_model = trainer.train_base_model(X_train, y_train)
        print("Base model trained")

        # Load test data
        X_id, y_id = dataset_handler.load_data(experiment, "id_test")
        X_ood, y_ood = dataset_handler.load_data(experiment, "ood_test")

        # Evaluate base model
        y_hat_id = base_model.predict(X_id)
        y_hat_ood = base_model.predict(X_ood)

        metrics_id = evaluator.compute_metrics(y_id, y_hat_id, zero_division)
        metrics_ood = evaluator.compute_metrics(y_ood, y_hat_ood, zero_division)

        for metric, value in metrics_id.items():
            evaluator.update_results(experiment, f'{metric}_id', value)

        for metric, value in metrics_ood.items():
            evaluator.update_results(experiment, f'{metric}_ood', value)

        # Compute QIPM
        acc_qipm = get_qipm(
            forest=base_model, X_A=X_train, y_A=y_train, X_B=X_ood,
            wheighted_by='accuracy', normalize=True, n_jobs=-1
        )

        f1_qipm = get_qipm(
            forest=base_model, X_A=X_train, y_A=y_train, X_B=X_ood,
            wheighted_by='fmeasure', normalize=True, n_jobs=-1
        )

        # Train adapted models
        strategies = [acc_qipm, f1_qipm]

        for i, strategy in enumerate(strategies, start=1):
            adapted_model = trainer.train_adapted_model(X_train, y_train, feature_bias=strategy)
            y_hat_ood_adapted = adapted_model.predict(X_ood)

            metrics_adapted = evaluator.compute_metrics(y_ood, y_hat_ood_adapted, zero_division)
            for metric, value in metrics_adapted.items():
                evaluator.update_results(experiment, f'{metric}_s{i}', value)

        print(f"Finished dataset {dataset_name} in {int(time() - start_time)} seconds")

    # Generate LaTeX reports
    evaluate = {
        'Accuracy': ['acc_id', 'acc_ood', 'acc_s1'], 
        'F1-measure': ['f1_id', 'f1_ood', 'f1_s2'],
    }
    
    latex_report = evaluator.generate_latex_report(evaluate, "Normalizing QIPM trees, with macro average score")

    with open("report.tex", "w") as f:
        f.write(latex_report)

if __name__ == "__main__":
    main()
