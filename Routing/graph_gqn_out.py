import csv
import matplotlib.pyplot as plt

#Basic code to graph results. Baseline formatting is result of "f.write..."
#   Evolutionary formatting is result of saving data output and then doing something like:
#   cat save_data_same_buf.txt | grep -A4 "LOSS LAST" | grep "AVG" | grep "Model  0" > model0_SB.txt

file_name = "./Routing/log_baseline.txt"
filetype = "Baseline"
filetype = "Evolutionary"

losses = []
eps = []
i = 0
if filetype in "Evolutionary":
    for m in range(1):
        file_name = "Model{}_SB.txt".format(m)
        f = open(file_name, newline="")
        my_reader = csv.reader(f, delimiter=" ")
        losses_for_m = []
        eps_for_m = []
        i = 0
        for row in my_reader:
            eps_for_m.append(i*100)
            i += 1
            losses_for_m.append(float(row[6]))
        losses.append(losses_for_m)
        eps.append(eps_for_m)
    
        plt.plot(eps_for_m, losses_for_m, label="model {}".format(m))
    plt.xlabel("Episodes")
    plt.ylabel("loss")
    plt.title("Evolutionary DGN episode/loss plot with shared buffer (Model 0)")
    plt.show()

else:
    f = open(file_name, newline="")
    tsv_reader = csv.reader(f, delimiter="\t")
    for row in tsv_reader:
        eps.append(i*100)
        i += 1
        losses.append(float(row[3]))
        if i*100 > 20000:
            break

    print("length: ")
    print(len(eps))
    plt.plot(eps, losses)
    plt.xlabel("Episodes")
    plt.ylabel("loss")
    plt.title("Baseline DGN episode/loss plot")
    plt.show()