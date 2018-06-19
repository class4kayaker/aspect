import sys
import csv
import math

def compute_convergence(filename):
    with open(filename, 'r') as f:
        oldline = None
        reader = csv.reader(f)
        for line in reader:
            c_data = []
            if oldline is None:
                for d in line:
                    try:
                        d = float(d)
                    except TypeError:
                        c_data.append('{:13s}|{:4s}'.format('NaN', ''))
                    else:
                        c_data.append("{:13.6g}|{:4s}".format(d, ''))
            else:
                for od,nd in zip(oldline,line):
                    try:
                        od = float(od)
                        nd = float(nd)
                    except TypeError:
                        c_data.append('{:13s}|{:4s}'.format('NaN', ''))
                    else:
                        c_rate = math.log(od/nd, 2)
                        c_data.append("{:13.6g}|{:4.2f}".format(nd, c_rate))
            print("|".join(c_data))
            oldline = line

if __name__ == "__main__":
    compute_convergence(sys.argv[1])
