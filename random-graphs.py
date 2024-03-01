import random
n = 20
m = 100


def randomConnectedGraph(n,m):
    if n < 1:
        print("Number of vertice must be positive")
        return None
    if m < 1:
        print("Number of edges must be positive")
        return None
    if m < n-1:
        print("Cannot create connected graph: small number of edges")
        return None
    max = n*(n-1) // 2
    if m > max:
        print("Cannot create graph with the given parameters: too many edges")
        return None

    v = [i+1 for i in range(n)]
    random.shuffle(v)
    neig = [[] for i in range(n)]
    deg = n*[0]
    d = 0

    with open("input.txt", 'w') as outfile:
        for j in range(1,n):
            i = random.randint(0,j-1)
            neig[i].append(j)
            deg[i] += 1
            deg[j] += 1
            if deg[i] > d: d = deg[i]
            if deg[j] > d: d = deg[j]
            outfile.write(str(v[i])+" "+str(v[j])+"\n")

        m = m - n + 1
        e = max - n + 1

        for i in range(n-1):
            for j in range(i+1,n):
                if not (j in neig[i]):
                    if m < e:
                        x = random.random()
                        if x < m/e:
                            m = m-1
                            e = e-1
                            outfile.write(str(v[i])+" "+str(v[j])+"\n")
                            deg[i] += 1
                            deg[j] += 1
                            if deg[i] > d: d = deg[i]
                            if deg[j] > d: d = deg[j]
                        else:
                            e = e-1
                    else:
                        outfile.write(str(v[i])+" "+str(v[j])+"\n")
                        deg[i] += 1
                        deg[j] += 1
                        if deg[i] > d: d = deg[i]
                        if deg[j] > d: d = deg[j]
    outfile.close()
    print(d)
    return(d)

# # Call the randomConnectedGraph function
result_d = randomConnectedGraph(n, m)

def randomConnectedMultigraph(n,m):
    if n < 1:
        print("Number of vertice must be positive")
        return None
    if m < 1:
        print("Number of edges must be positive")
        return None
    if m < n-1:
        print("Cannot create connected multigraph: small number of edges")
        return None


    v = [i+1 for i in range(n)]
    random.shuffle(v)
    deg = n*[0]
    d = 0

    p = 1 / (100*n)
    i = 0
    j = 1

    with open("input.txt", 'w') as outfile:
        while m > 0:
            x = random.random()
            if x < p:
                m = m-1
                outfile.write(str(v[i])+" "+str(v[j])+"\n")
                deg[i] += 1
                deg[j] += 1
                if deg[i] > d: d = deg[i]
                if deg[j] > d: d = deg[j]
            if j < n-1:
                j = j+1
            else:
                if i < n-2:
                    i = i+1
                else:
                    i = 0
                j = i+1
    outfile.close()
    print(d)
    return(d)

# Call the randomConnectedMultigraph function
#result_d_multigraph = randomConnectedMultigraph(n, m)




