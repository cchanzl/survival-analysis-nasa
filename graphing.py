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
    fig = plt.figure()
    models = ['RUL', 'km_rmst', 'Cox', 'RF (tuned)', 'rsf (pre-tuned)']
    i = 1
    for num in selected_unit_num:
        df_graph = graph_data[graph_data['unit num'] == num]
        for col in models:
            plt.subplot(4, 2, i)
            plt.plot(df_graph['cycle'], df_graph[col], label=col)
            plt.gca().set_title(num)
        i += 1
    plt.xlabel('Cycles')
    plt.ylabel('RUL')
    plt.legend()
    plt.show()


make_graph([5, 6, 20, 31, 38, 55, 78, 91])
