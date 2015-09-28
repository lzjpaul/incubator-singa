import argparse
import json
import sys
sys.path.append("../tensor")

import tensorIO as tio         # noqa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="input file")
    parser.add_argument('outfile', help="output file")
    parser.add_argument("-sep", type=int, default=2,
                        help="delimiter seperator")
    parser.add_argument("-axis", type=int, nargs="+", default=[0, 2, 3])
    parser.add_argument("-val", type=int, default=3)
    parser.add_argument("-classFile", default=None)
    args = parser.parse_args()
    delim = ","
    if args.sep == 2:
        delim = "\t"
    X, axDict = tio.construct_tensor(args.infile, args.axis, args.val,
                                     sep=delim, axisDict=None)
    xClass = None
    if args.classFile is not None:
        xClass = json.load(open(args.classFile, "r"))
    tio.save_tensor(X, axDict, xClass, args.outfile)
    print "Tensor shape:", X.shape

if __name__ == "__main__":
    main()
