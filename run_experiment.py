import torch
import argparse
from experiment import run_experiment
import random

manualSeed = 0
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type = str, default="compas", help="dataset to run(compas, framingham)")
parser.add_argument('--eval_metric', type = str, default="xauc", help="metric of ranking fairness, xauc or prf")
parser.add_argument('--classifier', type = str, default="lr", help="classificaion model. lr for logistic regression, rb for rankboost")
parser.add_argument('--num_train', type = int, default=999999, help="number of training examples")

if __name__ == "__main__":
    args = parser.parse_args()
    dataset, eval_metric, classifier, num_train = args.dataset, args.eval_metric, args.classifier, args.num_train
    # tee = subprocess.Popen(['tee', "results/{}_{}_{}_{}_sys.log".format(args.dataset,args.eval_metric,args.classifier,args.num_train)], stdin=subprocess.PIPE)
    # os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    # os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

    print("Run experiment for classifier {}, metric {} on {} dataset".format(classifier,eval_metric,dataset))
    run_experiment(args)
