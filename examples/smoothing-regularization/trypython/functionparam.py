def optimizator(alpha):
    for i in range(10):
        print alpha
        alpha -= alpha * 0.5


if __name__ == "__main__":
    optimizator(10)
