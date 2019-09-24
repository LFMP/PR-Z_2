INPUT = "./glia_cells.10.folds.txt"
OUTPUT = "./glial_cells.10.folds.txt"

with open(INPUT, 'r') as folds_file:
    out = open(OUTPUT, 'w')
    for line in folds_file:
        fold, dir_file, file_name = line.split(' ')
        dir_file = dir_file.split('/')[2:]

        out.write("{}&{}&{}&{} {}\n".format(dir_file[0], dir_file[1], dir_file[2], file_name, fold))