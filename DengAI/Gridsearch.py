from EpidemicModels import EpidemicModels
from DeepModels import DeepModels
def returnOptimizedModel_epidemic(X, y, bs=0, ep=0):
    # SJ Optimization
    scores = []
    aa = []
    bb = []
    if bs == 0:
        for a in range(10, 50, 20):  # 10, 100, 20
            for b in range(100, 501, 200):  # 100, 501, 200
                scores.append(EpidemicModels(e).cross_eval(X,y,a,b,3))
                aa.append(a)
                bb.append(b)
    else:
        aa = [bs]
        bb = [ep]
        scores.append(100)
    print("MAX SCORE: ", max(scores), " \n")
    max_score = max(scores)
    index = scores.index(max_score)
    batch = aa[index]
    epochs = bb[index]
    model = EpidemicModels(e)
    model.train(X, y, batch, epochs)
    # model.fit(X, y)
    return model, max_score, batch, epochs

def returnOptimizedModel(X, y, idim=20):
    scores = []
    aa = []
    bb = []

    for a in range(10, 50, 20):  # 10, 100, 20
        for b in range(100, 501, 200):  # 100, 501, 200
            print("a: ", a, " b: ", b)
            scores.append(DeepModels(m,idim).cross_eval(X,y,a,b,3))
            aa.append(a)
            bb.append(b)

    min_score = min(scores)
    index = scores.index(min(scores))
    batch = aa[index]
    epochs = bb[index]
    optimized_model = DeepModels(m, idim)
    optimized_model.train(X, y, batch, epochs)
    return optimized_model, min_score, batch, epochs
