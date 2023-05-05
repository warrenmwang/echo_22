import argparse
ap = argparse.ArgumentParser(description="test")
# ap.add_argument("-n", "--name", required=True, type=str, help="New model name")
# ap.add_argument("-m", "--model", required=True, type=str, help="Model Class name")

# args = ap.parse_args()

# print(args)

# print(f'you entered:\nname: {args.name}, model: {args.model}')

ap.add_argument('-l','--list', nargs='+', help='<Required> Set flag', required=True)
args = ap.parse_args()

print(type(args.list))
print(args.list)
print(args.list[0], type(args.list[0]))