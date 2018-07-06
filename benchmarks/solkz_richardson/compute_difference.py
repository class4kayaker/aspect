import re
import sys
import csv
import glob
import numpy as np

class DataError(Exception):
    pass

class AspectStateQuadrature:
    diff_tolerance = 1e-7
    @staticmethod
    def load_file(filename):
        with open(filename, 'r') as f:
            head_vars = [v.strip() for v in f.readline().split('\t')]
            data_array = np.loadtxt(f, delimiter='\t')
            dim = len([v for v in head_vars if v.startswith('X_')])
        head_vars = ['V'] + head_vars[dim*2+1:]
        data = {}
        points = data_array[:, 0:dim] # points
        weights = data_array[:, dim] # weights
        data['V'] = data_array[:, dim+1:2*dim+1] # velocities
        for i,v in enumerate(head_vars[1:]):
            data[v] = data_array[:,2*dim+1+i]
        return (dim, head_vars, points, weights, data)

    def __init__(self, fnames):
        self.var_names = None
        self.points = None
        self.weights = None
        self.data = None
        for fn in fnames:
            if self.var_names is None:
                self.dim, self.var_names, self.points, self.weights, self.data = self.load_file(fn)
            else:
                dim, v_names, points, weights, data = self.load_file(fn)
                if dim != self.dim:
                    raise DataError("Dimension mismatch in file {}".format(fn))
                for v1, v2 in zip(self.var_names, v_names):
                    if v1 != v2:
                        raise DataError(
                            "Invalid file combination with specified variable lists {} and {}({})".format(
                                self.var_names,
                                v_names,
                                fn
                            )
                        )
                self.points = np.concatenate((self.points, points), axis=0)
                self.weights = np.concatenate((self.points, points), axis=0)
                for v in self.var_names:
                    self.data[v] = np.concatenate((self.data[v], data[v]), axis=0)

    def hash_order(self, ranges):
        npoints = self.weights.shape[0]
        nbins = np.floor(np.power(npoints, 1./self.dim))

        min_x = ranges[0, :]
        max_x = ranges[1, :]
        bins = (max_x-min_x)/nbins

        index_list = np.arange(0, npoints, 1, dtype=np.int64)
        hashes = np.zeros((npoints,), dtype=np.int64)
        for i in range(self.dim):
            hashes += ((nbins**i)*np.floor((self.points[:,i]-min_x[i])/bins[i])).astype(np.int64)

        self.reorder(hashes.argsort())


    def reorder(self, new_order):
        self.points = self.points[new_order, :]
        self.weights = self.weights[new_order]
        for v in self.var_names:
            if v == 'V':
                self.data[v] = self.data[v][new_order, :]
            else:
                self.data[v] = self.data[v][new_order]

    def point_check(self, other):
        if not isinstance(other, AspectStateQuadrature):
            raise DataError("AspectStateQuadrature.point_check requires AspectStateQuadrature argument")

        if self.weights.shape[0] != other.weights.shape[0]:
            raise DataError("Datasets not the same length".format(var))

        return np.sqrt(
            np.sum(
                np.linalg.norm(self.points-other.points, axis=1)**2
                *(self.weights+other.weights)/2.
            )
        )

    def diff_l1(self, other, var):
        if not isinstance(other, AspectStateQuadrature):
            raise DataError("AspectStateQuadrature.diff_l1 requires AspectStateQuadrature argument")

        if self.point_check(other) > self.diff_tolerance:
            raise DataError("Dataset point order mismatch")

        if var not in self.data.keys() or var not in other.data.keys():
            raise DataError("Variable {} not in at least one of source datasets".format(var))

        if var == 'V':
            return np.sum(
                np.linalg.norm(self.data[var]-other.data[var], axis=1)
                *(self.weights+other.weights)/2.
            )
        else:
            return np.sum(
                np.abs(self.data[var]-other.data[var])
                *(self.weights+other.weights)/2.
            )

    def diff_l2(self, other, var):
        if not isinstance(other, AspectStateQuadrature):
            raise DataError("AspectStateQuadrature.diff_l2 requires AspectStateQuadrature argument")

        if self.point_check(other) > self.diff_tolerance:
            raise DataError("Dataset point order mismatch")

        if var not in self.var_names or var not in other.var_names:
            raise DataError("Variable {} not in at least one of source datasets".format(var))

        if var == 'V':
            return np.sqrt(
                np.sum(
                    np.linalg.norm(self.data[var]-other.data[var], axis=1)**2
                    *(self.weights+other.weights)/2.
                )
            )
        else:
            return np.sqrt(
                np.sum(
                    np.abs(self.data[var]-other.data[var])**2
                    *(self.weights+other.weights)/2.
                )
            )


def load_def_file(fn):
    f_line_re = re.compile(
        "(?P<level>[0-9]+):(?P<type>coarse|fine):(?P<glob>[A-Za-z0-9_/.]+)"
    )
    flist = {}
    with open(fn, 'r') as f:
        for line in f:
            match = f_line_re.match(line)
            if match:
                level = match.group("level")
                ftype = match.group("type")
                f_glob = match.group("glob")
                if level not in flist.keys():
                    flist[level] ={}
                if ftype not in flist[level].keys():
                    flist[level][ftype] = []
                flist[level][ftype].extend(glob.glob(f_glob))

    return flist


def diff_data(flist):
    err_data = []
    col_list = ['level']
    var_list = None
    levels = list(flist.keys())
    levels.sort()
    for level in levels:
        coarse_data = AspectStateQuadrature(flist[level]['coarse'])
        fine_data = AspectStateQuadrature(flist[level]['fine'])
        if var_list is None:
            var_list = coarse_data.var_names
            col_list.extend([v+"_L1" for v in var_list])
            col_list.extend([v+"_L2" for v in var_list])

        coarse_data.hash_order(region)
        fine_data.hash_order(region)

        row_data = {'level': level}

        for v in var_list:
            row_data[v+"_L1"] = fine_data.diff_l1(coarse_data, v)

        for v in var_list:
            row_data[v+"_L2"] = fine_data.diff_l2(coarse_data, v)

        err_data.append(row_data)

    return col_list, err_data


def compute_convergence(err_names, data):
    convergence_data = []
    last_row = None
    for row in data:
        if last_row is None:
            for v in err_names:
                row[v+"_rate"] = "nan"
        else:
            for v in err_names:
                row[v+"_rate"] = "{:4.2f}".format(np.log2(last_row[v]/row[v]))
        last_row = row
    return [v+"_rate" for v in err_names]


def print_data(header, data):
    writer = csv.DictWriter(sys.stdout, fieldnames=header)
    writer.writeheader()
    writer.writerows(data)

if __name__=="__main__":
    list_file = sys.argv[1]
    flist = load_def_file(list_file)

    region = np.array([[0.,0.],[1.,1.]]);

    cols, data = diff_data(flist)
    cols.extend(compute_convergence(cols[1:], data))
    cols[1:].sort()
    print_data(cols, data)
