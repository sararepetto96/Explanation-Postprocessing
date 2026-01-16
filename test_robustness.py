import os
import os
from plot import plot
import argparse
                        
if __name__ == "__main__":

    print("working in: ",os.getcwd())
    # Arguments
    parser = argparse.ArgumentParser(description='explanation analyzing')
    parser.add_argument('--corruption_type', type=str, default = 'gaussian_noise')
    parser.add_argument('--model_name', type=str, default = 'resnet50')
    parser.add_argument('--data_name', type=str, default = 'dermamnist')
    parser.add_argument('--agreement_measure', type=str, default = 'l1')
    parser.add_argument('--n_classes', type=int, default = 7)
    parser.add_argument('--normalization', type=str, default = 'quantil_local')
    parser.add_argument('--occluded_most_important', action='store_true', help='Enable my flag')
    parser.set_defaults(occluded_most_important=True)
    parser.add_argument('--no-occluded_most_important', dest='occluded_most_important', action='store_false', help='Disable my flag')
    
    args = parser.parse_args()
    
    attributions_per_algorithm = plot(data_name=args.data_name,model_name=args.model_name,
                                                         corruption_type = args.corruption_type,agreement_measure=args.agreement_measure,
                                                         normalization=args.normalization, occluded_most_important=args.occluded_most_important,
                                                         n_classes = args.n_classes)


   