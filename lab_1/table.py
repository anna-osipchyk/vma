from lab_1.main import generation,gauss,inaccuracy, gen

with open("Results_of_gauss.txt", "w", encoding="utf-8") as file1:
    print("Порядок матрицы          Относительная погрешность", file=file1)
    for k in range(3, 104):
        matrix, f = generation(k)
        normal_answer = gen(k)
        answ = gauss(matrix, f)
        print(" "*7, k, " "*15, inaccuracy(normal_answer, answ), file=file1)
