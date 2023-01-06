import os
import matplotlib.pyplot as plt

directory = 'result'

rating = {}
K = 16
 
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    
    if os.path.isfile(f):
        arr = filename.split('+')
        agent1 = arr[0].split('.')[1]
        agent2 = arr[1].split('.')[1]
        if agent1 not in rating:
            rating[agent1] = 1500
        if agent2 not in rating:
            rating[agent2] = 1500
        with open(f, 'r') as file:
            data = file.read()
            res = int(data)

            SA = 0.5
            SB = 0.5
            RA = rating[agent1]
            RB = rating[agent2]
            EA = 1 / (1 + 10 ** ((RA - RB) / 400))
            EB = 1 / (1 + 10 ** ((RB - RA) / 400))
            if res == 1:
                SA = 1
                SB = 0
            elif res == -1:
                SA = 0
                SB = 1
            newRA = RA + K * (SA - EA)
            newRB = RB + K * (SB - EB)

            rating[agent1] = newRA
            rating[agent2] = newRB
print(rating)

x = []
y = []
for agent in rating:
    x.append(agent)
    y.append(rating[agent])
plt.bar(x, y)
plt.show()