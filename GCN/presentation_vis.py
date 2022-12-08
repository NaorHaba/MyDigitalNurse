import matplotlib.pyplot as plt
import numpy as np

def plot_accs(train_acc,test_acc, model):
    starting_point = [1,2,3,4,5,6,7,8]
    train_color = '#009900'
    test_color = '#6600cc'
    max_train = np.max(train_acc)
    max_test = np.max(test_acc)
    plt.plot(starting_point, train_acc, color=train_color)
    plt.plot(starting_point, test_acc, color=train_color, linestyle='-.')
    plt.plot(np.argmax(train_acc)+1, max_train, color=train_color, marker='*')
    plt.plot(np.argmax(test_acc)+1, max_test, color=train_color, marker='*')
    plt.xlabel('Prediction Start Point')
    plt.ylabel('Prediction Accuracy')
    plt.legend(['Train','Test'])
    print(f'model:{model},max train: {max_train},max_test:{max_test}')
    plt.title(model)
    plt.show()

def plot_all(train_accs,test_accs, model):
    starting_point = [1,2,3,4,5,6,7,8]
    colors = ['#009900','#6600cc','#660600']
    legend_elements = [plt.Line2D([0], [0], color=colors[0], linestyle='-', label='1'),
                       plt.Line2D([0], [0], color=colors[1], linestyle='-', label='3'),
                       plt.Line2D([0], [0], color=colors[2], linestyle='-', label='2'),
                       plt.Line2D([0], [0], color='k', linestyle='-',label='Train'),
                       plt.Line2D([0], [0], color='k', linestyle='-.',label='Test')]
    # Create the figure
    for i in range(3):
        train_acc =train_accs[i]
        test_acc = test_accs[i]
        max_train = np.max(train_acc)
        max_test = np.max(test_acc)
        plt.plot(starting_point, train_acc, color=colors[i], linestyle='-')
        plt.plot(starting_point, test_acc, color=colors[i],linestyle='-.')
        plt.plot(np.argmax(train_acc)+1, max_train, color=colors[i], marker='*')
        plt.plot(np.argmax(test_acc)+1, max_test, color=colors[i], marker='*')
    plt.xlabel('Prediction Start Point')
    plt.ylabel('Prediction Accuracy')
    plt.legend(handles=legend_elements, loc='lower right', fontsize=9.5)
    # print(f'model:{model},max train: {max_train},max_test:{max_test}')
    plt.title('Next Step Prediction Using GCNs')
    plt.show()

#
# f = open(f"node_to_state_surgeon_new.pkl", "rb")
# import pickle
# node_to_state = pickle.load(f)
# states = list(node_to_state.values())
# possible = []
# for state1 in states:
#     for state2 in states:
#         if state1[0]==state2[0] and state1[1]!=state2[1] and (state1[1]==0 or state2[1]==0):
#             possible.append(f'{state1}->{state2}')
#         elif state1[1]==state2[1] and state1[0]!=state2[0] and (state1[0]==0 or state2[0]==0):
#             possible.append(f'{state1}->{state2}')
#         elif ((state1[0]==0 and state2[0]!=0) or (state1[0]!=0 and state2[0]==0)) and ((state1[1]==0 and state2[1]!=0) or (state1[1]!=0 and state2[1]==0)):
#             possible.append(f'{state1}->{state2}')



##NEW 3/12
#5nmt5mjz
train_acc_5nmt5mjz = [54.37,59.81,58.95,60.96,60.65,59.12,57.99,61.57]
test_acc_5nmt5mjz = [50.8,26.94,53.19,39.45,51.69,47.48,41.21,60.57]
plot_accs(train_acc_5nmt5mjz,test_acc_5nmt5mjz,'1')

# #elznjvsn
train_acc_elznjvsn = [58.37,54.82,53.15,51.74,56.14,58.97,57.08,60.4]
test_acc_elznjvsn = [43.37,24.89,44.99,38.12,50.61,40.2,35.83,49.17]
plot_accs(train_acc_elznjvsn,test_acc_elznjvsn,'3')

#d1ngf5p9
train_acc_d1ngf5p9 = [59.81,60.37,60.16,60.68,63.57,59.24,61.47,61.84]
test_acc_d1ngf5p9 = [50.8,25.81,53.19,38.12,51.15,51.41,42.05,60.01]
plot_accs(train_acc_d1ngf5p9,test_acc_d1ngf5p9,'2')

all_train = [train_acc_5nmt5mjz,train_acc_elznjvsn,train_acc_d1ngf5p9]
all_tests = [test_acc_5nmt5mjz, test_acc_elznjvsn, test_acc_d1ngf5p9]

plot_all(all_train,all_tests,'Rich')

# ##RNN
# #truesweep
# train_acc_truesweep = [77.67,80.97,90,60.96,60.65,59.12,57.99,61.57]
# test_acc_truesweep = [64.87,58.65,45.94,39.45,51.69,47.48,41.21,60.57]
# plot_accs(train_acc_truesweep,test_acc_truesweep,'truesweep')
#
# # #playful
# train_acc_playful = [84.71,54.82,53.15,51.74,56.14,58.97,57.08,60.4]
# test_acc_playful = [65.63,24.89,44.99,38.12,50.61,40.2,35.83,49.17]
# plot_accs(train_acc_playful,test_acc_playful,'playful')
#
# #exaulted
# train_acc_exaulted = [75.93,60.37,60.16,60.68,63.57,59.24,61.47,61.84]
# test_acc_exaulted = [65.88,25.81,53.19,38.12,51.15,51.41,42.05,60.01]
# plot_accs(train_acc_exaulted,test_acc_exaulted,'Rich')
#
# all_train = [train_acc_5nmt5mjz,train_acc_elznjvsn,train_acc_d1ngf5p9]
# all_tests = [test_acc_5nmt5mjz, test_acc_elznjvsn, test_acc_d1ngf5p9]
#
# plot_all(all_train,all_tests,'Rich')
