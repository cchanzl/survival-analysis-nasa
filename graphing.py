import pandas as pd
import matplotlib.pyplot as plt

##########################
#   Loading Data
##########################

graph_data = pd.read_csv('graphing.csv', delimiter=",")


################################
#   Graphing RUL prediction
################################


def make_graph(selected_unit_num):
    fig, axs = plt.subplots(int(len(selected_unit_num)/2), 2)
    models = ['RUL', 'km_rmst', 'Cox', 'NN (pre-tuned)', 'RF (tuned)', 'rsf (pre-tuned)', 'rsf (tuned)']
    i = -1
    for ax in axs.flatten():
        i += 1
        df_graph = graph_data[graph_data['unit num'] == selected_unit_num[i]]
        for col in models:
            ax.plot(df_graph['cycle'], df_graph[col], label=col)
            ax.set_title('Test observation ' + str(selected_unit_num[i]))
    plt.xlabel('Cycles')
    plt.ylabel('RUL')
    plt.legend()
    plt.show()

# make_graph([5, 6, 20, 31, 38, 55, 78, 91])
