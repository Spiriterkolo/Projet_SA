import torch
import math

def big_p(label):

    n = len(label)
    p = []
    for i in range(n):
        p_i = []
        for j in range(n):
            if i != j and label[i] == label[j]:
                p_i.append(j)
        p.append(p_i)
    return p


def sc_loss(prediction, label):

    n = len(label)
    p = big_p(label)
    tot = 0
    for i in range(n):
        sum2 = 0
        for indice in p[i]:
            sum2 += math.exp(torch.dot(prediction[i], prediction[indice]))
            for j in range(n):
                sum = 0
                if i != j:
                    sum += math.exp(torch.dot(prediction[i], prediction[j]))
        if len(p[i])>0:
            card = len(p[i])
        else:
            card = 1
        tot -= (1/card)*math.log((sum2/(sum+1)))
    return tot


def y_prediction(prediction_i, prototypes):

    mini = torch.dot(prediction_i, prototypes[0])
    ypred = 0
    for k in range(len(prototypes)):
        arg = torch.dot(prediction_i, prototypes[k])
        if arg < mini:
            mini = arg
            ypred = k
    return torch.tensor(ypred)


def pos(label):
    return torch.tensor(int((label+1)/2))


def sym(position, nloc):
    n = int(torch.sqrt(torch.tensor(nloc)))
    q = torch.div((position-1), n, rounding_mode='floor')
    r = torch.remainder((position-1), n)
    r = (n - 1) - r
    return q * n + r + 1


def dgraph(pos1, pos2, nloc):
    n = int(torch.sqrt(torch.tensor(nloc)))
    r1 = torch.remainder((pos1 - 1), n)
    q1 = torch.div((pos1 - 1), n, rounding_mode='floor')
    r2 = torch.remainder((pos2 - 1), n)
    q2 = torch.div((pos2 - 1), n, rounding_mode='floor')
    distance = (r1 - r2) + (q1 - q2)
    return torch.abs(distance)


def geo_dist(prediction, gt, gamma, nloc):
    ppos = pos(prediction)
    gtpos = pos(gt)
    if ppos == 0:
        return gamma
    elif ppos == gtpos:
        return gamma/4
    elif ppos == sym(gtpos, nloc):
        return gamma/2
    else:
        return gamma*dgraph(ppos, gtpos)


def lgui(predictions, labels, prototypes, gamma, nloc = 16):
    somme = torch.zeros(len(predictions))
    qi = len(prototypes)/(len(prototypes)-1)
    for i in range(len(predictions)):
        somme[i] = geo_dist(y_prediction(predictions[i], prototypes), labels[i], gamma, nloc)
    return torch.sum(somme) * qi


def dcos(prediction_i, prototypes, label_i):
    cos = torch.nn.CosineSimilarity()
    return 1 - cos(prediction_i, prototypes[label_i])


def score(predictions, prototypes, labels, h):
    somme = 0
    for i in range(len(predictions)):
        somme += torch.linalg.norm(h[i])*(2-dcos(predictions[i], prototypes, labels[i]))
    return torch.tensor(somme)
