import os
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score,
                             recall_score, precision_score)
from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
from qipm import RandomForestClassifier as BSRandomForestClassifier
from qipm import AVALIABLE_DATASETS, get_qipm
from tableshift import get_dataset
from scipy.stats import wilcoxon

import logging
logging.basicConfig(
    level=logging.INFO,  # Sets the minimum log level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log output format
    datefmt="%Y-%m-%d %H:%M:%S",  # Date/time format
)
info = logging.info


class DatasetHandler:
    """
    Handles dataset loading and preparation for training and testing.
    """
    def __init__(self, base_path):
        self.base_path = base_path

    def load_data(self, dataset_name, partition):
        if isinstance(partition, str):
            filename = f"{dataset_name}_{partition}.feather"
            file_path = os.path.join(self.base_path, filename)
            df = pd.read_feather(file_path)
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
        elif isinstance(partition, list):
            df = pd.DataFrame()
            for p in partition:
                filename = f"{dataset_name}_{p}.feather"
                file_path = os.path.join(self.base_path, filename)
                df = pd.concat([df,pd.read_feather(file_path)])
        else:
            raise("The partition must be a string or a list of strings")
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        labels = np.array(df.columns)[:-1]
        return X.values, y.values, labels
    
    def load_tableshift(self, dataset_name, partition):
        dset = get_dataset(
            dataset_name, 
            cache_dir="../tableshift/tmp", 
            use_cached=True
        )
        
        X, y, _, _ = dset.get_pandas(partition)
        return X.values, y.values

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
            max_features="sqrt",
            n_jobs=self.n_jobs,
            random_state=2,
            max_samples=self.max_samples
        )
        model.fit(X, y)
        return model

    def train_adapted_model(self, X, y, feature_bias):
        model = BSRandomForestClassifier(
            n_estimators=1000,
            max_features="sqrt",
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
    def __init__(self, filename, zero_division=0, average="macro"):
        self.results = pd.DataFrame(AVALIABLE_DATASETS, 
                                    columns=["dataset", "experiment"]
                                    ).set_index("experiment")
        self.filename = filename
        self.zero_division = zero_division
        self.average = average

    def save_results(self):
        self.results.to_markdown(f"{self.filename}.md")
        self.results.to_csv(f"{self.filename}.csv")

    def load_results(self):
        self.results = pd.read_csv(f"{self.filename}.csv")

    def update_results(self, experiment, metric, value):
        self.results.loc[experiment, metric] = value
        self.save_results()

    def compute_metrics(self, y_true, y_pred):
        kwargs = {
            "zero_division": self.zero_division, 
            "average": self.average
        }
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, **kwargs),
            "precision": precision_score(y_true, y_pred, **kwargs),
            "recall": recall_score(y_true, y_pred, **kwargs)
        }

    def update_p_value(self, column):
        n = self.results.shape[0]
        self.results.iloc[n,0] = "p-value"
        arr_notna = self.results.iloc[:,[1,2]].dropna(axis=0).values
        p_value = wilcoxon(arr_notna[:,1], arr_notna[:,column]).pvalue
        self.results.iloc[n,column+1] = p_value


    def generate_latex_report(self, strategy_names=None):
        """
        Generate LaTeX tables for the evaluation results.

        Args:
            strategy_names (list of str): List of strategy names to include in the report.

        Returns:
            str: Complete LaTeX content for the report.
        """
        
        basic_columns = ["dataset", "accuracy_id", "f1_id", 
                         "precision_id", "recall_id", "accuracy_ood", 
                         "f1_ood", "precision_ood", "recall_ood"]

        metrics = [m[:-3] for m in basic_columns if m[-3:] == "_id"]
        n_strategies = int((self.results.shape[1] - len(basic_columns)) / len(metrics))
        
        if strategy_names is None:
            strategy_names = ["Strategy " + f"{i+1}" for i in range(n_strategies)]
            
        latex_content = []
        
        for i in range(n_strategies):
            for metric in metrics[:2]:
                raw_string = strategy_names[i] + ", evaluated by " + metric
                latex_content.append(r"\begin{table}[h!]")
                latex_content.append(r"\centering")
                latex_content.append(r"\caption{" + raw_string + r"}")
                latex_content.append(r"\begin{tabular}{lc|cc}")
                latex_content.append(r"\toprule")
                
                latex_content.append(r"\multirow{2}{*}{Dataset} & \multirow{2}{*}{ID} & \multicolumn{2}{c}{OOD} \\")
                latex_content.append(r"& & {baseline} & {biased} \\")
                latex_content.append(r"\midrule")
                cols = ["dataset"]
                cols.append(f"{metric}_id")
                cols.append(f"{metric}_ood")
                cols.append(f"{metric}_s{i+1}")
                df_ev = self.results[cols].copy()
                df_ev[cols[1:]] *= 100
                df_styled = df_ev.style.hide_index()
                df_styled = df_styled.highlight_max(
                    subset=cols[2:], axis=1, props="textbf:--rwrap;"
                    )
                df_styled = df_styled.format("{:.2f}", subset=cols[1:], na_rep="N/A")
                lines = df_styled.to_latex().split("\n")
                for l in lines[2:-2]:
                    latex_content.append(l)
                latex_content.append(r"\bottomrule")
                df_notna = df_ev.dropna()    
                p_value = wilcoxon(df_notna.iloc[:,-2], df_notna.iloc[:,-1]).pvalue
                mc_start = r"\multicolumn{4}{l}{pvalue: "
                mc_end = r"}"
                latex_content.append(mc_start + f"{p_value:0.3f}" + mc_end)
                latex_content.append(lines[-2])
                latex_content.append(r"\end{table}")
                latex_content.append("")
        
        print("\n".join(latex_content))
        return "\n".join(latex_content)


def main():
    path = "../feather_one_hot_"
    filename = "experiment_result"

    dataset_handler = DatasetHandler(path)
    evaluator = Evaluation(filename)


    for dataset_name, experiment in AVALIABLE_DATASETS[-1:]:
        info(f"Processing dataset: {dataset_name}")

        X_train, y_train, labels = dataset_handler.load_data(experiment, "train")
        max_samples = 0.05 if experiment == "acspubcov" else None
        trainer = ModelTrainer(n_jobs=-1, max_samples=max_samples)

        # Train base model
        base_model = trainer.train_base_model(X_train, y_train)
        info("Base model trained")

        # Load test data
        X_id, y_id, labels = dataset_handler.load_data(experiment, "id_test")
        X_ood, y_ood, labels = dataset_handler.load_data(experiment, "ood_test")

        # Evaluate base model
        y_hat_id = base_model.predict(X_id)
        y_hat_ood = base_model.predict(X_ood)

        metrics_id = evaluator.compute_metrics(y_id, y_hat_id)
        metrics_ood = evaluator.compute_metrics(y_ood, y_hat_ood)

        for metric, value in metrics_id.items():
            evaluator.update_results(experiment, f"{metric}_id", value)

        for metric, value in metrics_ood.items():
            evaluator.update_results(experiment, f"{metric}_ood", value)

        # Compute QIPM
        
        kwargs = {
            "forest": base_model, 
            "X_A": X_train, 
            "y_A": y_train, 
            "X_B": X_ood,
            "normalize": True, 
            "n_jobs": -1
        }
        
        acc_qipm = get_qipm(
            weighted_by="accuracy", 
            **kwargs)
        f1_qipm = get_qipm(
            weighted_by="fmeasure", 
            **kwargs)
        
        del(base_model)
        # Train adapted models
        strategies = [acc_qipm, f1_qipm]

        for i, strategy in enumerate(strategies, start=1):
            adapted_model = trainer.train_adapted_model(X_train, y_train, feature_bias=strategy)
            y_hat_ood_adapted = adapted_model.predict(X_ood)
            del(adapted_model)

            metrics_adapted = evaluator.compute_metrics(y_ood, y_hat_ood_adapted)
            for metric, value in metrics_adapted.items():
                evaluator.update_results(experiment, f"{metric}_s{i}", value)
            
            info(f"Strategy {i} finished")

        info(f"Finished dataset: {dataset_name}")

    # Generate LaTeX reports
    evaluate = {
        "Accuracy": ["accuracy_id", "accuracy_ood", "accuracy_s1"], 
        "F1-measure": ["f1_id", "f1_ood", "f1_s2"],
    }
    
    strategy_names = ["$QIPM_{acc}$", "$QIPM_{F1}$"]
    latex_report = evaluator.generate_latex_report(strategy_names)

    with open(f"{filename}.tex", "w") as f:
        f.write(latex_report)

if __name__ == "__main__":
    main()
