import argparse
import predict_functions

# Setup the arg parser
parser = argparse.ArgumentParser(description = 'Flower prediction')
parser.add_argument('image_file', nargs='?', default="flowers/test/10/image_07104.jpg")
parser.add_argument('checkpoint', nargs='?', default="checkpoints/cli_checkpoint.pth", help="checkpoint file")

# Options
parser.add_argument('--top_k', dest="top_k", type=int, default="3", help="get the top k classes")
parser.add_argument('--category_names', dest="category_names", default="cat_to_name.json", help="file containing category-to-flower names")

# Enable GPU?
parser.add_argument('--gpu', dest="gpu", action='store_true', help="enable GPU mode, otherwise use CPU")

args = parser.parse_args()
probs, names = predict_functions.start(args)

print("Top {} classes' probabilities: {}".format(args.top_k, probs))
print("Top {} classes' flower names: {}".format(args.top_k, names))