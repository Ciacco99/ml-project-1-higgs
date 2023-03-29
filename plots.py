import matplotlib.pyplot as plt


def plot_lambda_search(param, acc_tr, acc_te, file):
    plt.figure()
    plt.semilogx(param, acc_tr, label="Training accuracy")
    plt.semilogx(param, acc_te, label="Test accuracy")
    plt.xlabel("Lambda")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(file)
