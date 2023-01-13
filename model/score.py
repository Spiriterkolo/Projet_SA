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
    return ypred


def pos(label):
    return int((label+1)/2)


def sym(position, nloc):
    n = int(torch.sqrt(torch.tensor(nloc)).item())
    r = (position-1) % n
    q = (position-1) // n
    r = (n - 1) - r
    return q * n + r + 1


def dgraph(pos1, pos2):
    delta = pos1 - pos2
    return -delta*delta/2


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
    somme = 0
    qi = len(prototypes)/(len(prototypes)-1)
    for i in range(len(predictions)):
        somme += geo_dist(y_prediction(predictions[i], prototypes), labels[i], gamma, nloc) * qi
    return torch.tensor(somme)


