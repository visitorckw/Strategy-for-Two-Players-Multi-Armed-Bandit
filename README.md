# Strategy for Two Players Multi-Armed Bandit


## Background and Motivation

As one of the team leaders, you know that nothing keeps your fellow elves more productive and motivated than a steady supply of candy canes! But all seven levels of the Candy Cane Forest are closed for revegetation, so the only ones available are stuck in the break room vending machines. And even though you receive free snacks on the job, the vending machines are always broken and don’t always give you what you want.

Due to social distancing, only two elves can be in the break room at once. You and another team leader will take turns trying to get candy canes out of the 100 possible vending machines in the room, but each machine is unpredictable in how likely it is to work. You do know, however, that the more often you try to use a machine, the less likely it will give you a candy cane. Plus, you only have time to try 2000 times on the vending machines until you need to get back to the workshop!

If you can collect more candy canes than the other team leaders, you’ll surely be able to help your team win Santa's contest! Try your hand at this multi-armed candy cane challenge!

### Problem Definition
Your objective is to find a strategy to beat your opponent as much as possible.

Both participants will work with the same set of $N$ vending machines (bandits). Each bandit provides a random reward based on a probability distribution specific to that machine. Every round each player selects ("pulls") a bandit, the likelihood of a reward decreases by 3%.

Each agent can see the move of the other agent, but will not see whether a reward was gained in their respective bandit pull.

## Methodology
### Agent Strategy
1. Pure Greedy base on Average Profit in history

    In this strategy, we play each machine one time in the beginning. After that, each time we play, we calculate the average profit of each machine in history. In other words, we choose to play the machine that makes the following formula have the highest value:
    
    $Avg_i = \frac{w_i}{n_i}$
    
    $w_i$: Rewards we have gotten on machine $i$ in history.
    $n_i$: Times we have played on machine ${i} in history.
    
    Note that the strategy is straightforward and doesn't consider the effect of the decaying rate.
    
2. Epsilon-Delta Strategy

    The problem is an "exploitation" and "exploration" tradeoff dilemma. Epsilon-Delta is one of the simple and famous strategies. We choose a probability $p$ for exploration. Each time when we play, we use $p$ of chance randomly choose a machine to play (exploration), and another $1 - p$ of the chance to play the machine that has highest average reward in history just like we mentioned in pure greedy strategy (exploitation).
    
    $chosen \ machine = \begin{cases} random, & {chance \ of \ p} \\ argmax \ Avg_i, & {chance \ of \ 1 - p} \end{cases}$

3. UCB Strategy

    UCB formula is one of the famous mathematical methods for balancing the "exploitation" and "exploration" strategy. Each time when we play, we choose a machine $i$ which makes the following formula has the highest value:
    
    $UCB = \frac{w_i}{n_i} + c * \sqrt{\frac{ln{N}}{n_i}}$
    
    $w_i$: Rewards we have gotten on machine $i$ in history.
    $n_i$: Times we have played on machine $i$ in history.
    $c$: Exploration parameter, equals to $\sqrt{2}$ in theory.
    $N$: Times we have played on all machines in history.
4. Advance Greedy Method base on Maximum Likelihood Estimation

    We would like to maximize our expected profit each time we play based on the observation of historical results. To know the expected profit of each machine, we have to consider the effect of the decaying rate. For a specific machine $i$, we have history information $I$, representing in a series $(player_j, w_j)$. If we have a given probability $p$ for machine $i$ beginning expected profit, we could calculate $P(I|p)$ easily:
    
    $P(I|p) = \prod_{j=0}{\begin{cases} (0.97)^j * p, & {  player_j = me \land w_j = 1} \\  (1-p*(0.97)^j), & {  player_j = me \land w_j = 0} \\ 1, & {  otherwise} \end{cases}}$
    
    However, what we want to know is the value of $p$. So we could try to find the value $p = p*$ which makes $P(p|I)$ have the highest value. We use the likelihood function $L(p|I) = P(I|p)$ to find the maximum likelihood estimation of $p$.
    Each time we play, we find a probability $p_i$ for machine $i$ which has maximum likelihood. In practice, we use interior-point method to find the optimal value $p*$. Then we choose the machine which has the highest expected value. The expected value for machine $i$:
    
    $EV_i = p_i * (0.97) ^ {n_i}$
    
    $p_i$: maximum likelihood estimation of $p$ on machine $i$.
    $n_i$: times we and out opponent have played on mahine $i$.
    
5. Advance UCB Method by Combining Advance Greedy and Pure UCB

    We could deem the advanced greedy method as a strategy to predict the expected reward of each machine more precisely. Hence we could take advantage of it to adjust the formula of UCB:
    
    $newUCB = EV_i + c \sqrt{\frac{\ln{N}}{n_i}}$
    
    $EV_i$: Ecpected value of machine $i$ by calculating maximum likelihood.
    $n_i$: Times we play on machine $i$ in history.
    $c$: Exploration parameter, equals to $\sqrt{2}$ in theory.
    $N$: Times we play on all machines in history.


    
7. Combining Advance UCB with a ML/DL model:
    
    We could also let the output of a machine learning model affect the decision to advance UCB strategy. Hoping the model's output will increase the performance of the UCB strategy. The new UCB formula is described as follows: 
    
    $newUCB = y_i + c \sqrt{\frac{\ln{N}}{n_i}}$
    
    $EV_i$: Ecpected value of machine $i$ predicted by the ML model.
    $n_i$: Times we have played on machine $i$ in history.
    $c$: Exploration parameter, equals to $\sqrt{2}$ in theory.
    $N$: Times we have played on all machines in history.
    $y_i$: Output of the ML/DL model on machine $i$.

8. Thompson Sample and Bayesian Distribution
    Although each machine will have a 3% probability drop after being selected, we think it still has a certain relationship with Bayesian inference, therefore we use the Thompson sample.



    
    * For each round $t = 1,2,...$do
        * calculate $N^{1}_{i}$ and $N^{0}_{i}$
            * $N^{1}_{i}$ = the amount of candy get from machine $i$ 
            * $N^{0}_{i}$ = the amount round that didn't get candy from machine  $i$ 
        * $\theta_{i}(t)$ $\approx$ $\beta(N^{1}_{i},N^{0}_{i})$
        * choose the machine $k$ biggest $\theta_{k}(t)$

9. Maching Learning Predicting Strategy

    We try to use a machine learning model to predict the original probability. We collect the data at the 25, 50, 75, and 90, percentile of the total step.
    *  The model we used
        * Linear Regressor
        * K-Neighbors Regressor
        * Epsilon-Support Vector Regressor
        * Random Forest Regressor
        * Lightgbm
    * The main 12 features used in the LightGBM are as follows.
        1. Current step
        2. Number of times my agent chose the bandit
        3. Number of times my agent chose the bandit, corrected by 1/decay factor at the time of choice
        4. Number of times my agent obtained reward from the bandit
        5. Number of times my agent obtained reward from the bandit, corrected by 1/decay factor at the time of choice
        6. Hit rate (4. / 2.)
        7. Adjusted hit rate (5. / 2.)
        8. Number of times opponent chose the bandit
        9. Number of times opponent chose the bandit, corrected by 1/decay factor at the time of choice
        10. Number of steps passed since my agent chose the bandit
        11. Number of steps passed since opponent chose the bandit

7. Gradient Bandit Algorithm

    For each possible action, we have a preference value $H$. The probability of each action will be chosen to obey the softmax distribution. The formula is as follows:

    $Pr(a) = \pi(a) = \frac{e^{H(a)}}{\sum{e^{H(b)}}}, b \in Action$
    
    $\pi$ denotes our playing strategy. The preference value $H$ will initialize to $0$ for all actions. Then we apply SGD algorithm to our strategy for updating preference value. After we make an action $A_t$, we could update by using the formula below:

    $H(A_t)' = H(A_t) + \alpha(R - \bar R)(1 - \pi(A_t))$
    $H(a)' = H(a) - \alpha(R - \bar R)\pi(a), \forall a \neq A_t$
    
    $H'$: New preference value.
    $H$: Old preference value.
    $\alpha$: Learing rate.
    $R$: Last reward
    $\bar R$: Average reward

9. Exponential smoothing predict strategy

    When it comes to the times series predicting method, exponential smoothing is a classic one. It fixes the predicting value each time we get an accurate result. The predicting value will be updated in the following formula:
    
    $EV' = (1 - \alpha)EV + \alpha R$
    
    $EV'$: New prediction
    $EV$: Old prediction
    $\alpha$: A hyperparameter controls the influence of new results.
    $R$: Accurate result.

11. ARIMA time seires predicting algorithm

    ARIMA is another time series predicting algorithm. AR stands for autoregressive model, and MA stands for moving average. In addition to combining the two methods, the model also considers the number of differences to make the data stationary. The predicting formula is shown as follows:
    
    ![](https://i.imgur.com/zaSQdNO.png)


13. Adaptive UCB with exponential smoothing exploratin paramter

    We try to find a way to change the value of the exploration parameter in UCB formula dynamically. Intuitively speaking, when we observe the agent exploiting, we will want to encourage the agent to try more exploration and vice versa. So we compare the decisions made with UCB and greed. If the decisions are the same, we will increase the value of $c$. Otherwise, we will decrease $c$. The updated formula for $c$:
    
    $c' = c * 1.1, if A = B$
    $c' = c * 0.9, if A \neq B$
    
    $c'$: New value of c
    $c$: Old value of c
    $A$: UCB dicision
    $B$: greedy dicision 
    
14. Combining ML method with adaptive UCB strategy
    According to a previous study, we try to combine ML method with adaptive UCB.
    
    $newUCB = y_i + c \sqrt{\frac{\ln{N}}{n_i}}$
    
    $EV_i$: Ecpected value of machine $i$ predicted by the ML model.
    $n_i$: Times we have played on machine $i$ in history.
    $c$: Exploration parameter, changes dynamically using exponential smooth strategy.
    $N$: Times we have played on all machines in history.
    $y_i$: Output of the ML/DL model on machine $i$.


15. Deep Q Learning
In deep Q learning, we use the features mentioned above in machine learning as the state and the action is to choose which machine.

    $Q(s,a)=r+\gamma\max_{a'}Q(s',a')$
    $L=(r+\gamma\max_{a'}Q(s',a')-Q(s,a))^2$

### Analysis Strategy

<!-- 9. Analysis method by competing with a random agent

    To compare and analysis whether each method is good or not, we run agent builded up by each strategy against a random agent to see the scoring status. -->
    
1. ELO rating system analysis

    The Elo rating system is a method for calculating the relative skill levels of players in zero-sum games such as chess. It is named after its creator Arpad Elo, a Hungarian-American physics professor. We compete with all the agents with each other. If agent A has a rating of $R_A$ and agent B rating of $R_B$, the exact formula (using the logistic curve with base 10) for the expected score of player A is
    
    $E_A = \frac{1}{1+10^{(R_B-R_A)/400}}$
    
    Similarly the expected score for agent B is
    
    $E_B = \frac{1}{1+10^{(R_A-R_B)/400}}$
    
    Suppose player A (again with rating $R_A$) was expected to score $E_A$ points but actually scored $S_A$ points $(win = 1, tie = 0.5, lose = 0)$. The formula for updating that player's rating is
    
    $R^{'}_{A} = R_A + K(S_A - E_A)$
    $R^{'}_{B} = R_B + K(S_B - E_B)$

    
    

## Data Collection and Analysis Result

### Analysis
#### Model Training
* Train data: 12000
* Test data: 2000

| | LinearRegressor | LightBGM | KNN | RandomForest| SVR | DecisionTree |
| --------| -------- | -------- | -------- | -------| ------ | ---- |
| Loss(MSE) | 0.294     | 0.205     | 0.287     |0.267 | 0.239 | 0.303 |

#### ELO Rating
![](https://i.imgur.com/DIvFHgH.png)
![](https://i.imgur.com/GTi4pMJ.png)


#### Adaptive exploration parameter
![](https://i.imgur.com/cmqKZJK.png)



### Result
1. Light GBM has the best ELO rating and the highest winning rate.
2. In model training, a model with a smaller loss does not necessarily perform the best. For example, SVR performs about the same as random.
3. We use Knn with UCB to explore the effect of dynamically adjusting the exploration coefficient. It can be seen that Knn with dynamic adjustment of the exploration coefficient is the most effective, while the fixed exploration coefficient is the worst.
4. Initially, the value of the dynamically adjusted exploration coefficient is larger, which means that it tends to explore more at first. Later, it tends to exploit more.
5. Exponential smoothing performs the best among the methods other than machine learning.


### Conclusion
1. We recommend using the Lightgbm model to solve the problem due to its high ELO rating and fast enough for real-time usage.
2. We recommend using an exponential smoothing strategy for a no-ML solution to solve the problem, due to its relatively high ELO rating, simplicity, and fast.
3. We speculate that the poor performance of SVR may be due to an inappropriate choice of the kernel function (in this case we only used the RBF function).
4. In the UCB formula, we have a hyperparameter - "c" which is theoretically equal to $\sqrt2$. However, we found that The smaller the "c", the better the performance of the model. So we thought "exploitation" is much more important than "exploration" in this game.
5. LightGBM with UCB actually performs worse. We believe this is because LightGBM can already make very accurate predictions on its own, so the exploration coefficient given by UCB becomes more of a disturbance.
### Future Work
1. Try applying deep learning models such as LSTM, transformer.
2. Analysis of each machine learning model and finding better hyper parameters for each model.
4. Research on optimizing each hyperparameter.

## Reference
* Aleksandrs Slivkins (2019), "Introduction to Multi-Armed Bandits", Foundations and Trends® in Machine Learning: Vol. 12: No. 1-2, pp 1-286. http://dx.doi.org/10.1561/2200000068
* Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., … Liu, T.-Y. (2017). Lightgbm: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems, 30, 3146–3154.
* Telecom ParisTech (2017), Reinforcement learning: Multi-armed bandits Thomas Bonald






