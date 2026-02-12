import matplotlib.pyplot as plt

def plot_f1_vs_k(k_values, f1_scores):
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, f1_scores, marker='o')
    plt.title("Évolution du F1-score selon k (k-NN)")
    plt.xlabel("k (n_neighbors)")
    plt.ylabel("F1-score")
    plt.grid(True)
    plt.savefig("outputs/f1_vs_k.png")
    plt.show()
