import argparse
from harmony_ai.config import load_config
from harmony_ai.dataio import make_splits_from_folder
from harmony_ai.features import extract_features_for_splits
from harmony_ai.model import train_and_save, load_model
from harmony_ai.eval import evaluate_and_report, plot_confusion_matrix_cli

def main():
    p = argparse.ArgumentParser(prog="harmonyai", description="HarmonyAI pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("splits");     s1.add_argument("--config", required=True)
    s2 = sub.add_parser("features");   s2.add_argument("--config", required=True)
    s3 = sub.add_parser("train");      s3.add_argument("--config", required=True)
    s4 = sub.add_parser("eval");       s4.add_argument("--config", required=True)
    s5 = sub.add_parser("plot");       s5.add_argument("--config", required=True)

    args = p.parse_args()
    cfg = load_config(args.config)

    if args.cmd == "splits":
        make_splits_from_folder(cfg)
    elif args.cmd == "features":
        extract_features_for_splits(cfg)
    elif args.cmd == "train":
        train_and_save(cfg)
    elif args.cmd == "eval":
        model = load_model(cfg["paths"]["model_path"])
        evaluate_and_report(cfg, model)
    elif args.cmd == "plot":
        plot_confusion_matrix_cli(cfg)

if __name__ == "__main__":
    main()
