from matplotlib import pyplot as plt


def att_visualization(sentence_attentions):
    plt.imshow(sentence_attentions, cmap='plasma', interpolation='nearest')
    plt.xticks([i for i in range(0, len(sentence_attentions), 32)])
    plt.yticks([i for i in range(0, len(sentence_attentions), 32)])
    plt.show()