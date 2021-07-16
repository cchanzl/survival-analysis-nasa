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
    models = ['RUL', 'NN (tuned)', 'RF (trended)', 'NN (tuned trended classification)', 'NN (tuned trended)']
    i = -1
    for ax in axs.flatten():
        i += 1
        df_graph = graph_data[graph_data['unit num'] == selected_unit_num[i]]
        ax.set_title('Test engine ' + str(selected_unit_num[i]))
        for col in models:
            if col == 'RUL':
                ax.plot(df_graph['cycle'], df_graph[col], label=col, linestyle='dashed', linewidth=0.75)
            else:
                ax.plot(df_graph['cycle'], df_graph[col], label=col, linewidth=0.75)
    fig.legend(models, loc='lower center',
               ncol=len(models), bbox_transform=fig.transFigure)
    plt.xlabel('Cycleas')
    plt.ylabel('RUL')
    plt.show()

# make_graph([5, 6, 20, 31, 38, 55, 78, 91])
