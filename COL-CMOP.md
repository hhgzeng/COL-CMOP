# Collaborative Orthogonal Learning for Constrained Multi-Objective Optimization

Yubo Wang, Chengyu Hu, Xinyi Wu, Tingyu Zhang, Wenyin Gong, Senior Member, IEEE, Xuesong Yan, Liang Gao, Senior Member, IEEE, 

Abstract—Balancing feasibility, convergence, and diversity is a fundamental challenge in constrained multi-objective optimization, particularly in landscapes with irregular or disconnected boundaries. Existing methods primarily exchange solution positions but neglect valuable convergence direction information contained within the evolutionary process, which may lead to inefficient and redundant searches for the constrained Pareto front (CPF). To address this, we propose a dual-swarm competitive optimizer with collaborative orthogonal learning (COL) strategy, which effectively decouples global exploration and diversity exploitation. Specifically, the main swarm performs trend learning to identify convergence directions from boundary and winner-loser interaction information, enabling global exploration through infeasible regions toward the CPF. Guided by the learned trends, the auxiliary swarm executes orthogonal learning to search complementary subspaces, which broadens the solution distribution while avoiding redundant searches, ensuring diversity exploitation capability. Additionally, a niche-guided subset selection strategy is introduced to maintain uniform distribution within the objective space through a three-level subset division mechanism based on local niche capacity. Extensive experiments on standard and extended benchmark instances demonstrate the robustness and superiority of our approach over nine state-ofthe-art methods. 

Index Terms—Constrained multi-objective optimization problems, global exploration, diversity exploitation, collaborative orthogonal learning, niche-guided subset selection. 

## I. INTRODUCTION

cations, such as job shop scheduling problems [1], circuit design problem [2] and optimal power flow problems [3]. Unlike unconstrained optimization, the primary challenge in solving CMOPs lies in balancing feasibility, convergence, and diversity. This challenge is particularly difficult when handling CMOPs with complex landscapes characterized by narrow, disconnected, or irregular boundaries. Such complex landscapes often impede traditional constrained multi-objective optimization evolutionary algorithms (CMOEAs), causing the population to be easily trapped in local feasible regions or fail to cover the entire constrained Pareto front (CPF). 

To tackle these challenges, numerous constraint-handling techniques (CHTs) have been developed and incorporated into evolutionary algorithms (EAs) [4]–[7]. Among them, the constraint dominance principle (CDP) stands out as a classic approach due to its simplicity and parameter-free nature [8]. However, when the problem contains large infeasible barriers or disjoint feasible islands, CDP often leads to premature convergence. To alleviate this, ε methods have gained popularity [9], [10]. By dynamically relaxing constraint boundaries through a threshold ε, these methods allow the population to preserve high-quality infeasible solutions, thereby enabling the traversal of infeasible regions to converge to the CPF. More recently, multi-population and multi-tasking frameworks have been proposed and widely utilized to decouple constraint satisfaction and objective optimization [11]–[13]. The key feature of these methods lies in equipping different populations with distinct CHTs and assigning them different tasks, thereby enhancing algorithmic performance through knowledge transfer between populations. Nevertheless, as reflected in the experimental results in Section IV-B, existing approaches, despite the integration of various advanced CHTs, often struggle to maintain a satisfactory and comprehensive performance when solving CMOPs with diverse landscapes. To systematically address challenges arising from different types of problems, we present the following analysis of the current studies. 

• In many existing multi-population and multi-tasking frameworks, knowledge transfer between populations is primarily achieved by sharing the spatial coordinates of high-quality solutions. While this approach effectively transmits the locations of promising regions, it fails to uncover and convey the potential beneficial convergence directions within the evolutionary process that could help populations converge and search more efficiently. When handling problems with complex landscapes, the main population may converge slowly, while the auxiliary populations may perform redundant searches in the regions explored by the main population. Hence, a directionguided collaborative framework is needed to decouple global exploration and diversity exploitation. 

• Even existing CMOEAs equipped with various advanced CHTs still primarily rely on traditional reproduction operators, such as differential evolution (DE) and genetic algorithms (GA), to generate offspring. Their search process is mainly driven by random neighborhood perturbations around the current solutions, which may limit algorithmic performance in CMOPs featuring expansive search space landscapes. Therefore, there is significant potential in extracting reusable search experience from population evolution to form promising search directions and guide more efficient evolution. 

• Several CMOEAs tend to favor solutions with better feasibility or convergence in their environment selection [14], [15]. Although such preference facilitates rapid convergence, it may also cause the population to converge unidirectionally toward CPF in a clustered state, hindering the population’s ability to perform extensive searches. When the feasible region is narrow and separated, such as LIR-CMOP3, this strategy makes the population prone to being trapped in local feasible regions, resulting in only a small fraction of CPFs being searched. Thus, it is necessary to maintain good distribution in the objective space and promote uniform search across multiple directions. 

Based on the above considerations, in this article, we propose a dual-swarm optimizer with collaborative orthogonal learning, named DSOCOL. Specifically, the main contributions of our research include the following. 

1) A dual-swarm cooperative framework driven by evolutionary direction learning is proposed to decouple the conflicting tasks of convergence and diversity. Specifically, the framework synergizes solution knowledge transfer with evolutionary direction learning. The primary swarm primarily focuses on rapidly converging to feasible regions to localize the CPF, while the auxiliary swarm emphasizes extensive exploitation of the objective space to enhance uniform search for the CPF. By sharing both high-quality offspring solutions and promising evolutionary directions, the two swarms effectively collaborate to overcome challenges in CMOPs with complex landscapes. 

2) A novel collaborative orthogonal learning (COL) strategy is developed to synergize the search behaviors of the two swarms. Instead of merely sharing the solutions, the main swarm performs trend learning to extract promising convergence directions by the solution status within different niches and transmits them to the auxiliary swarm, ensuring global exploration. The auxiliary swarm then executes orthogonal learning based on these trends to explore complementary subspaces, aiming to reduce redundant searches and enhance diversity exploitation along the CPF. 

3) A niche-guided subset selection strategy is designed for environmental selection to handle irregular Pareto fronts. By implementing a three-level subset division mechanism based on local niche capacity, this strategy dynamically regulates swarm density. The strategy ensures the exploration of sparse niches and effectively manages the trade-off between local convergence and global diversity, enhancing the capability to capture discrete CPF fragments. 

In addition, comprehensive experiments are conducted to validate the proposed algorithm, which covers four standard benchmark suites, one specific suite characterized by deceptive constraints, and ten real-world application scenarios spanning four distinct domains, covering both standard and large-scale decision variables. By comparing DSOCOL with nine state-ofthe-art CMOEAs, the experimental results confirm its superior performance and robustness across a wide range of CMOPs with diverse and complex landscapes. 

The remainder of this article is organized as follows: Section II reviews the related work. Sections III and IV provide the details of DSOCOL and the experimental results and analysis. Section V summarizes conclusions and future work. 

## II. RELATED WORK

## A. Preliminaries

A CMOP can be mathematically denoted as follows: 

$$
\begin{array}{l l} \text {Minimize} & \mathbf {F} (\mathbf {x}) = (f _ {1} (\mathbf {x}), f _ {2} (\mathbf {x}), \dots , f _ {M} (\mathbf {x})) ^ {T}, \\ \text {s.t.} & \mathbf {x} \in \mathbb {S} ^ {D}, \\ & g _ {i} (\mathbf {x}) = 0, i = 1, \dots , p \\ & h _ {j} (\mathbf {x}) \leq 0, j = 1, \dots , q \end{array}\tag{1}
$$

where $\mathbf { F } ( \mathbf { x } )$ are objective functions; M is the number of objectives; $\mathbf { x } = \left( x _ { 1 } , x _ { 2 } , \cdots , x _ { D } \right)$ represents one solution in the decision space $\mathbb { S } ^ { D } \mathrm { : }$ ; D is the decision variable dimension; $g _ { i } ( \mathbf { x } )$ and $\underline { { h _ { i } ( \mathbf { x } ) } }$ denote the degree of constraint violation (CV) of the solution on the ith and jth constraint functions. The overall CV of x is 

$$
C V (\mathbf {x}) = \sum_ {i = 1} ^ {p} | g _ {i} (\mathbf {x}) | + \sum_ {j = 1} ^ {q} m a x (0, h _ {j} (\mathbf {x}))\tag{2}
$$

$\begin{array} { r } { C V ( \mathbf { x } ) = 0 } \end{array}$ means x is a feasible solution, otherwise x is infeasible solution. 

For any two solutions, x is said to Pareto dominate y (notated as $\underline { { \textbf { x } \prec \textbf { y } } }$ if x is no worse than y regarding all objective components and outperforms y in at least one. z is an unconstrained Pareto optimal solution if z is not dominated by any other candidate x within the entire search space. The aggregate of these solutions is termed the unconstrained Pareto optimal set (UPS). Similarly, a feasible solution z (where $C V ( { \bf z } ) ~ = ~ 0 )$ is defined as a constrained Pareto optimal solution if no other feasible solution x can dominate it. These points together constitute the constrained Pareto optimal set (CPS). The projections of the UPS and CPS onto the objective space are designated as the unconstrained Pareto front (UPF) and the CPF, respectively. 

## B. Existing CMOEAs

Over the past few decades, many CMOEAs have been proposed to solve the challenge of balancing convergence, feasibility, and diversity. Based on their underlying CHTs, these existing approaches can be broadly categorized into the following groups. 

1) Methods based on the separation of objectives and constraints. When addressing CMOPs, these approaches treat the evaluation of objective functions and constraint satisfaction as distinct processes. The CDP stands out as a classic example of this category, applying the rules of standard Pareto dominance to constrained environments [8]. Despite the simplicity of CDP, methods driven purely by feasibility often suffer from premature convergence. To better leverage information from infeasible individuals, Fan et al. [10] integrated an improved ε method into the MOEA/D. This method dynamically tunes the relaxation parameter ε according to the current ratio of feasible solutions. Similarly, to prevent the population from becoming stagnant, Zhu et al. [9] introduced a mechanism that utilizes an enhanced ε-constrained strategy to identify the CPF. Specifically, with a constrained relaxation value ε, a solution x is considered to ε-constraint dominate solution $\textbf { y } ( \textbf { x } \prec _ { \varepsilon } \textbf { y } )$ when any of the criteria listed below are met. 

$$
\begin{array}{l} \bullet C V (\mathbf {x}) <   \varepsilon \& C V (\mathbf {y}) <   \varepsilon \& \mathbf {x} \prec \mathbf {y}. \\ \bullet C V (\mathbf {x}) <   \varepsilon <   C V (\mathbf {y}). \\ \bullet \varepsilon <   C V (\mathbf {x}) <   C V (\mathbf {y}). \end{array}
$$

Furthermore, Stochastic Ranking (SR) offers another perspective by introducing a probability parameter, $p f .$ . This parameter determines whether the comparison between two individuals should focus on their objective values or their degree of constraint violation. To improve this, Ying et al. [16] developed an adaptive SR mechanism that automatically adjusts $p f$ based on the evolutionary status and the variation in violation levels among individuals. Additionally, Jan and Khanum et al. [17] incorporated a modified version of SR into MOEA/D for solving CMOPs. Gu et al. [18] developed a surrogate-assisted evolutionary algorithm, where they introduced an enhanced SR strategy by integrating a fitness-based mechanism with an adaptive probability operator. Liu et al. [19] investigated indicator-based CMOEAs by incorporating the indicator-based MOEA with SR, the ε-constraint method, and CDP, respectively. Nevertheless, a major drawback of both SR and ε-based methods is their reliance on extra parameters, which can be difficult to tune and may negatively affect the algorithm’s overall performance. 

2) Methods based on penalty function. The fundamental principle of this category involves incorporating constraint violations directly into the objective functions in the form of penalty terms. While this strategy is attractive due to its straightforward implementation, the algorithm’s effectiveness relies heavily on the precise tuning of penalty coefficients. To optimize the search process, Ma et al. [20] developed a shiftbased penalty scheme. This approach directs the population toward feasible regions during the initial phases of evolution, while focusing on converging to the CPF in the latter stages. Jiao et al. [21] introduced a feasibility-guided strategy that revises the objective functions according to the level of constraint violation, thereby producing updated fitness values. In terms of parameter automation, Yu et al. [22] introduced a mechanism assisted by dynamic selection preference to adjust the priority between objective minimization and constraint satisfaction. More precisely, the emphasis placed on objectives is progressively decreased from 1 down to 0, following a specific cosine curve trajectory. In addition, to better coordinate objective optimization and constraint satisfaction, the penalty coefficient is dynamically tuned based on the proportion of feasible solutions in the current population. Vaz et al. [23] proposed a three-stage penalty scheme, where distinct penalty coefficients are applied during different phases of the evolutionary process. 

3) Methods of transforming CMOPs into other problem. Popular strategies for addressing CMOPs often involve restructuring the problem into either a multi-phase process or a collaborative framework involving multiple populations. In the context of multi-stage CMOEAs, the evolutionary process is segmented into distinct periods, each employing a unique CHT. For instance, Fan et al. [15] introduced a push-pull framework. This method operates by initially driving the population toward the UPF during the push phase, and subsequently retrieving individuals back towards the feasible regions and the CPF in the pull phase. Similarly, CMOEA-MS developed by Tian et al. [24] functions in two stages: one dedicated to locating the feasible region, and the other designed to distribute the solutions along the feasible boundary. Xiang et al. [25] presented a two-stage algorithm called CIC-MOEA/D. In the first stage, the algorithm focuses solely on the objective functions to identify the UPF. In the second stage, constraint information is progressively strengthened so that the search can better converge toward the CPF. 

On the other hand, multi-population CMOEAs evolve several independent groups of individuals simultaneously, assigning different CHTs to each group. A notable example is the work of Sun et al. [12], who devised a novel constraint relaxation technique aimed at retaining high-potential solutions within the population. Concurrently, they utilized an external archive based on CDP to store the set of feasi ble non-dominated solutions. Additionally, Wang et al. [14] presented a cooperative framework based on DE that utilizes m distinct subpopulations. In this structure, each subpopulation is tasked with optimizing a specific objective subject to constraints. Alongside this, an archive population is employed to preserve the identified constrained non-dominated solutions, thereby facilitating convergence to the CPF. A dualpopulation framework was introduced by Liu et al. [26] to reformulate CMOPs. Within this architecture, one specific group is dedicated entirely to handling constraints, whereas the second group concentrates solely on improving the ob jective optimization. Throughout the evolutionary process, information exchange and knowledge sharing occur between these two distinct populations. Conversely, Yang et al. [27] tackled CMOPs by dividing the main problem into several smaller sub-tasks through the segmentation of the objective space. To effectively address these generated sub-problems, a combination of different CHTs was simultaneously employed. 4) Methods of altering the reproduction operators. The core of this category of algorithms lies in the construction of specialized and efficient reproductive operators. To enhance the capability of competitive swarm optimizer (CSO) in solving CMOPs, Ming et al. [28] developed a competitive and cooperative swarm optimizer. This technique refines the competitive process via an extension parameter to expedite convergence towards the Pareto front. Concurrently, it employs a cooperative mutual-learning scheme to assist the swarm in bypassing local feasible traps. In terms of mutation strategies, Yu et al. [29] introduced a novel mechanism capable of processing both feasible and infeasible individuals, thereby ensuring a uniform distribution along the CPF. More recently, Ming et al. [30] integrated deep reinforcement learning into an operator selection framework. By treating the population’s status as environmental states, this approach adaptively identifies the most suitable evolutionary operator for the current stage. Miyakawa et al. [31] developed a strategy for direct matching, through which each individual is provided with a specific search direction determined by its distribution in the objective space. In contrast, He et al. [32] focused on the reproduction stage and pointed out that generating offspring with either feasibility potential or useful infeasibility is critical to the search process. 

5) Methods of evolutionary multi-tasking optimization. As a specialized branch of evolutionary transfer optimization (ETO) [33], evolutionary multi-tasking optimization (EMT) focuses on enhancing performance through the collaborative exchange of beneficial information between distinct tasks. This mechanism shares conceptual similarities with the multipopulation strategy in CMOEAs. Drawing inspiration from this information interaction, Qiao et al. [13] developed a method that treats populations (specifically parent or offspring sets) as transferable knowledge. They introduced an unconstrained auxiliary task designed to locate the UPF. The transfer strategy is condition-dependent: parent populations are transferred when the CPF is a subset of the UPF, whereas offspring populations are exchanged when there is a partial overlap between the CPF and UPF. In a different approach, Ming et al. [11] constructed a tri-task framework that integrates the strengths of both constraint-ignoring and constraintrelaxing tasks. Qiao et al. [34] proposed MTCMO, which handles CMOPs by introducing a dynamic auxiliary task whose constraint boundary is progressively tightened to better match the main task. With this design, the population can bypass infeasible barriers at the early stage and then move toward the CPF from infeasible regions as the search proceeds. Nevertheless, the simultaneous optimization of multiple tasks inevitably increases algorithmic complexity, which may hinder convergence when computational resources are restricted. 

Remarks: To retain the synergistic benefits of multipopulation interaction while bypassing the burden of task reformulation, the proposed DSOCOL adopts an approach distinct from EMT. Specifically, from a structural perspective, both DSOCOL and the EMT framework belong to the multi-population paradigm, leveraging information interaction between different populations. However, they diverge fundamentally in two aspects. First, in terms of problem formulation, EMT methods typically model the CMOP as multiple explicit tasks and solve them concurrently. In contrast, DSOCOL maintains the original problem structure without reformulating it into auxiliary tasks; instead, it decouples the search process into global exploration and diversity exploitation through a collaborative orthogonal learning mechanism. Second, regarding the knowledge transfer strategy, while EMT focuses on transferring high-quality spatial coordinates (positional knowledge) by selecting elite solutions from different tasks, DSOCOL emphasizes extracting and transferring favorable evolutionary trends to provide more efficient directional guidance for the search. 

## III. THE PROPOSED METHOD

## A. Procedure of DSOCOL

Algorithm 1 provides the detailed explanation of DSOCOL. The maximum generation T and swarm size N are input. The algorithm starts by evolving two different swarms $S _ { 1 }$ and $S _ { 2 }$ (line 1). Afterwards, the evolutionary generation counter t and the number of niches K are initialized (line 2). The COL strategy is configured to execute every 75 generations, and the initial value for constraint relaxation $\varepsilon _ { m a x }$ is set to the maximum constraint violation degree in the initial population (line 3). K uniformly weighted vectors V are initialized (line 4). We adopt the fitness evaluation functions of SPEA2 [35] in this research, which can be represented as Eq. (3). It is worth noting that when $\varepsilon = 0 ,$ , this strategy is equivalent to CDP. In line $5 , S _ { 1 }$ is evaluated under constrained relaxation $\varepsilon ( t )$ to traverse infeasible regions and enhance global exploration. In contrast, $\varepsilon$ for $S _ { 2 }$ is set to 0, thereby allowing the population to enter the feasible region and enhance diversity exploitation along the CPF (line 6). The main loop runs until the computational resources are exhausted (lines 7-20). 

$$
\left\{ \begin{array}{c} \gamma_ {(\mathbf {x}, \mathbf {y})} ^ {\varepsilon} = \left\{ \begin{array}{l l} 1, & \text { if   } \mathbf {y} \prec_ {\varepsilon} \mathbf {x} \\ 0, & \text { otherwise } \end{array} \right. \\ R _ {\mathbf {x}} ^ {\varepsilon} = \sum_ {\mathbf {i} \in S, \mathbf {i} \neq \mathbf {x}} \gamma_ {(\mathbf {x}, \mathbf {y})} ^ {\varepsilon} \\ f i t n e s s (\mathbf {x}) = \sum_ {\mathbf {i} \in S, \mathbf {i} \neq \mathbf {x}} | R _ {\mathbf {x}} ^ {\varepsilon} | + \frac {1}{d i s t (\mathbf {x} , \mathbf {x} ^ {\prime}) + 2} \end{array} \right. \tag {3}
$$

where $\underline { { d i s t ( { \bf x } , { \bf x } ^ { \prime } ) } }$ is the Euclidean distance from x to its closest neighbor in the objective space. 

Algorithm 1: The framework of DSOCOL

Input: T (maximum generation), N (swarm size)
Output: $S_{1}$ (particle swarm)

1 Generate initial swarms $S_{1}$ and $S_{2}$ ;

2 Set the current generation t = 1, the number of niches $K = \frac{N}{10}$ ;

3 Set the frequency of COL $T_{COL} = 75$ , the initial constraint relaxation value $\varepsilon(t) = \varepsilon_{max} = \max_{\mathbf{x} \in S_{1}} CV(\mathbf{x})$ ;

4 Initialize the weight vectors $V = [V^{1}, \ldots, V^{K}]$ ;

5 $F_{1} \leftarrow$ Evaluate $S_{1}$ according to Eq. (3) where $\varepsilon = \varepsilon(t)$ ;

6 $F_{2} \leftarrow$ Evaluate $S_{2}$ according to Eq. (3) where $\varepsilon = 0$ ;

7 while $t \leq T$ do

8 $O_{1} \leftarrow$ Offspring Generation ( $S_{1}, 1, F_{1}$ ) (Algorithm 2);

9 $O_{2} \leftarrow$ Offspring Generation ( $S_{2}, 2, F_{2}$ ) (Algorithm 2);

10 $(S_{1}, F_{1}) \leftarrow$ Environmental Selection ( $S_{1} \cup O_{1} \cup O_{2}, N, \varepsilon(t)$ )
    (Algorithm 4);

11 $(S_{2}, F_{2}) \leftarrow$ Niche-Guided Subset Selection ( $S_{2} \cup O_{1} \cup O_{2}, N, V$ )
    (Algorithm 5);

12    if mod(t, $T_{COL}$ ) == 0 then

13 $(O_{1}, O_{2}) \leftarrow$ Collaborative Orthogonal Learning ( $S_{1}, S_{2}, V, N, \varepsilon(t)$ ) (Algorithm 3);

14 $(S_{1}, F_{1}) \leftarrow$ Environmental Selection ( $S_{1} \cup O_{1} \cup O_{2}, N, \varepsilon(t)$ )
    (Algorithm 4);

15 $(S_{2}, F_{2}) \leftarrow$ Niche-Guided Subset Selection $(S_{2} \cup O_{1} \cup O_{2}, N, V)$ (Algorithm 5);

16    end

17 $t = t + 1$ ;

18 $\varepsilon(t) \leftarrow$ Update the constraint relaxation value according to Eq. (7);

19 end

20 Return $S_{1}$ ; 

Loser Updates: In lines 8-9, new offspring are generated swarm $S _ { 1 }$ and $S _ { 2 }$ based on different conditions using Algorithm 2. Specifically, in lines 1 to 11 of Algorithm 2, the parental swarm is randomly divided into two groups. Through pairwise comparison of fitness values, a winner group $\underline { { \boldsymbol { S } } } _ { w }$ and a loser group $\underline { { s _ { l } } }$ are formed. Next, regardless of whether the loser group belongs to $S _ { 1 }$ or $S _ { 2 }$ , new offspring are generated using Eq. (4), which originates from [36] and represents the most fundamental CSO loser group update method. 

$$
\left\{ \begin{array}{c} \Delta (t) = \mathbf {x} _ {w} (t) - \mathbf {x} _ {l} (t) \\ \mathbf {v} _ {l} (t + 1) = r _ {0} \mathbf {v} _ {l} (t) + r _ {1} \Delta (t) \\ \mathbf {x} _ {l} (t + 1) = \mathbf {x} _ {l} (t) + \mathbf {v} _ {l} (t + 1) \end{array} \right.\tag{4}
$$

where $\mathbf { x } _ { w } , \mathbf { x } _ { l }$ , and ${ \bf \underline { v } } _ { l }$ refer to the positions of winner particle and loser particle, and the velocity of the loser particle, with t indicating the current generation. $r _ { 0 }$ and $r _ { 1 }$ are uniformly distributed random values between 0 and 1. 

Winner Updates: Since $S _ { 1 }$ and $S _ { 2 }$ are assigned different task goals, we develop two approaches to further improve their winner groups, as observed in lines 13-17 of Algorithm 2. In detail, $S _ { 1 }$ and $S _ { 2 }$ update their winner group using Eq. (5) and Eq. (6), respectively. This approach differs from conventional CSO methods in that standard CSO only applies polynomial mutation to the winning group, whereas our method further enhances the winning group. While consuming the same number of evaluations, this improvement significantly boosts CSO performance, which will be demonstrated in the subsequent ablation study section. 

$$
\left\{ \begin{array}{c} C (t) = \frac {1}{2} (\mathbf {x} _ {w} ^ {f} (t) + \mathbf {x} _ {w} ^ {l} (t)) \\ \Delta (t) = \mathbf {x} _ {w} ^ {f} (t) - \mathbf {x} _ {w} ^ {l} (t) \\ \mathbf {x} _ {w} ^ {f} (t + 1) = C (t) + \frac {\beta}{2} \Delta (t) \\ \mathbf {x} _ {w} ^ {l} (t + 1) = C (t) - \frac {\beta}{2} \Delta (t) \end{array} \right.\tag{5}
$$

where $\underline { { \mathbf { x } _ { w } ^ { f } ( t ) } }$ and $\underline { { \mathbf { x } _ { w } ^ { l } ( t ) } }$ are the first and last halves of the winner group at generation t respectively; $\overline { { \beta } }$ is the spread factor, controlled by a distribution index to regulate the distance between the generated offspring and the center point. 

$$
\mathbf {x} _ {w} ^ {k} (t + 1) = \mathbf {x} _ {w} ^ {k} (t) + \frac {\mathbf {x} _ {w} ^ {b} (t) - \mathbf {x} _ {w} ^ {k} (t)}{2} + \frac {\mathbf {x} _ {w} ^ {i} (t) - \mathbf {x} _ {w} ^ {j} (t)}{2}\tag{6}
$$

where $\underline { { \mathbf { x } _ { w } ^ { k } ( t ) } }$ and $\mathbf { x } _ { w } ^ { b } ( t )$ represent the k-th particle and the best solution in the winner group respectively; $\overline { { { \bf x } _ { w } ^ { \ i } ( t ) } }$ and $\overline { { \mathbf { x } _ { w } ^ { j } ( t ) } }$ are two distinct random particles. 

Environmental Selection: After the generation of the new offspring $\mathcal { O } _ { 1 }$ and $\mathcal { O } _ { 2 } ,$ Algorithms 4 and 5 are executed separately to select the new generation of the two swarms (lines 10-11). In lines 12-16, the proposed COL strategy is executed every 75 generations. New solutions generated through trend learning and orthogonal learning (line 13) are then selected by the different methods (lines 14-15). Finally, the generation counter t and the constraint relaxation $\varepsilon ( t )$ are updated. In this paper, we use Eq. (7) to update $\varepsilon ( t )$ , with parameters in this formula identical to the original reference [9]. 

$$
\varepsilon (t + 1) = \left\{ \begin{array}{l l} (1 - \sigma) \varepsilon (t), & \text { if } f _ {r} \leq \alpha \\ \varepsilon_ {\max}, & \text { otherwise } \end{array} \right.\tag{7}
$$

where t denotes the index for the current generation; σ is confined within the range $[ \sigma _ { \mathrm { m i n } } , 1 ] .$ $\sigma _ { \operatorname* { m i n } } = \bar { 1 } - ( 1 / \varepsilon ( 0 ) ) ^ { \overline { { ( T _ { \operatorname* { m a x } } / 3 ) } } } ;$ 

```txt
Algorithm 2: Offspring Generation
Input: S (parent swarm), T (swarm type), F (fitness value of the S)
Output: O (offspring swarm)
1 while |S| > 1 do
2    {p, q} ← Randomly select two particles from S;
3    S ← S\{p, q};
4    if F(p) < F(q) then
5    S_w ← q;
6    S_l ← p;
7    else
8    S_w ← p;
9    S_l ← q;
10    end
11 end
12 S'_l ← Update S_l according to Eq. (4);
13 if T == 1 then
14    S'_w ← Update S_w according to Eq. (5);
15 else
16    S'_w ← Update S_w according to Eq. (6);
17 end
18 O ← {S'_w, S'_l};
19 Return O; 
```

$\underline { { \varepsilon } } _ { \mathrm { m a x } }$ is determined by the highest constraint violation value in the initial population; α is set to 0.95; $f _ { r }$ represents the feasible ratio. 

## B. Collaborative Orthogonal Learning

Unlike traditional collaboration frameworks that rely on sharing solution information, COL facilitates a convergence direction synergy between the two swarms, which is achieved through two distinct yet complementary mechanisms. Algorithm 3 provides the details. The COL process is executed within each niche (lines 2-24). Specifically, for $S _ { 1 }$ , the process begins by identifying candidate solutions associated with the i-th niche (lines 3-4). Subsequently, these candidates are categorized into a feasible non-dominated set and a remaining set (line 5). If both sets are non-empty, a representative solution is randomly selected from each set (line 7). Otherwise, the candidates are sorted based on constraint violation and divided into two sets at the median value, from which one representative solution is selected respectively (lines 8-11). Upon determining the representative solutions, the trend learning strategy is executed by $\underline { { \mathrm { E q . } } }$ (8) to extract the promising convergence direction, which is then utilized to generate the new experimental solutions (line 12). 

$$
\left\{ \begin{array}{c} \Delta_ {1} = \mathbf {x} _ {w} - \mathbf {x} _ {l}, \\ \Delta_ {2} = \mathbf {x} _ {w} - \mathbf {x} _ {r}, \\ \mathbf {v} = \frac {\tau \Delta_ {1} + (1 - \tau) \Delta_ {2}}{\| \tau \Delta_ {1} + (1 - \tau) \Delta_ {2} \|}, \\ \mathbf {u} _ {m a i n} = \mathbf {x} _ {r} + \eta_ {k} \cdot \mathbf {v}, \end{array} \right.\tag{8}
$$

where $\mathbf { x } _ { w }$ and $\mathbf { x } _ { l }$ represent the winner and loser solutions within a specific niche of $S _ { 1 }$ respectively; $\underline { { \mathbf { x } _ { r } } }$ is set to the upper bound and lower bound of the decision space respectively; $\begin{array} { r } { \tau = \frac { F E } { F E _ { m a x } } } \end{array}$ , where $F E$ and $F E _ { m a x }$ are the current and maximum number of function evaluations respectively; η<sub>k</sub> is the kth random search step size. 

Local exploitation vector $\underline { { ( \Delta _ { 1 } ) } } \mathrm { : }$ Defined as $\mathbf { x } _ { w } - \mathbf { x } _ { l }$ , which captures the relative difference between the superior and inferior individuals. It reflects local competitive trends, guiding the search to focus on the neighborhood of high-quality solutions to enhance the local mining capability of the CPF. 

Global exploration vector $\underline { { ( \Delta _ { 2 } ) } } \mathrm { : }$ Defined as $\mathbf { x } _ { w } - \mathbf { x } _ { r } ,$ , this component captures a global evolutionary tendency of the population. Specifically, since the population is initialized within the bounded decision space, and the losers iteratively evolve by learning from winners, the winners act as a carrier of the most successful search experiences at the current stage. Consequently, the positional shift of $\mathbf { x } _ { w }$ across iterations reflects a favorable evolutionary trend. By constructing a direction from the decision space boundary to the winner, $\Delta _ { 2 }$ converts the population’s local successful experiences into a global exploration direction. This global exploration vector orients the search towards a broad search of the entire decision space, preventing stagnation in local regions and enhancing the population’s capability to rapidly converge towards the CPF across the global landscape. 

Algorithm 3: Collaborative Orthogonal Learning
Input: $S_1$ (particle swarm), $S_2$ (particle swarm), $V$ (weight vectors), $N$ (swarm size), $\varepsilon$ (constraint relaxation)
Output: $\mathcal{O}_1$ (offspring swarm), $\mathcal{O}_2$ (offspring swarm)
1 $\mathcal{O}_1 = \mathcal{O}_2 = \emptyset$ ;
2 for $i = l$ to $K$ do
3 $\theta_1 \leftarrow$ Calculate the angle between $V^i$ and each solution in $S_1$ ;
4 $\Phi_1 \leftarrow$ Select $\lfloor \frac{N}{K} \rfloor$ solutions from $S_1$ having the minimum angular distance to $V^i$ according to $\theta_1$ ;
5 $(\Phi_1^{nf}, \Phi_1^{rem}) \leftarrow$ Divide $\Phi_1$ into two groups according to $\epsilon$ ;
// the non-dominated feasible solution set $\Phi_1^{nf}$ and the rest of the solutions $\Phi_1^{rem}$ 6 if $\Phi_1^{nf} \neq \emptyset$ and $\Phi_1^{rem} \neq \emptyset$ then
7 $(\mathbf{x}_w, \mathbf{x}_l) \leftarrow$ Randomly select a solution from $\Phi_1^{nf}$ and $\Phi_1^{rem}$ respectively;
8 else
9 $(\Phi_1^{fh}, \Phi_1^{sh}) \leftarrow$ Sort $\Phi_1$ in ascending order based on constraint violation degree;
// $\Phi_1^{fh}$ and $\Phi_1^{sh}$ represent the sorted solutions from the first half and the second half
10 $(\mathbf{x}_w, \mathbf{x}_l) \leftarrow$ Randomly select a solution from $\Phi_1^{fh}$ and $\Phi_1^{sh}$ respectively;
11 end
12 $(\mathbf{u}_{main}, \mathbf{v}) \leftarrow$ Performing the trend learning strategy based on $\mathbf{x}_w$ and $\mathbf{x}_l$ according to Eq. (8);
13 $\mathcal{O}_1 = \mathcal{O}_1 \cup \mathbf{u}_{main}$ ;
14 $\theta_2 \leftarrow$ Calculate the angle between $V^i$ and each solution in $S_2$ ;
15 $\Phi_2 \leftarrow$ Select $\lfloor \frac{N}{K} \rfloor$ solutions from $S_2$ having the minimum angular distance to $V^i$ according to $\theta_2$ ;
16 $\Phi_2^{nf} \leftarrow$ Select non-dominated feasible solutions from $\Phi_2$ based on CDP;
17 if $\Phi_2^{nf} \neq \emptyset$ then
18 $\mathbf{x}_b \leftarrow$ Randomly select a solution from $\Phi_2^{nf}$ ;
19 else
20 $\mathbf{x}_b \leftarrow$ Select a solution with the minimum constraint violation from $\Phi_2$ ;
21 end
22 $\mathbf{u}_{aux} \leftarrow$ Performing the orthogonal learning strategy based on $\mathbf{x}_b$ and v according to Eq. (9);
23 $\mathcal{O}_2 = \mathcal{O}_2 \cup \mathbf{u}_{aux}$ ;
24 end
25 Return $\mathcal{O}_1, \mathcal{O}_2$ ; 

To dynamically balance these two behaviors, a time-varying weight coefficient τ is introduced. Specifically, in the early stage (small τ), the weight $( 1 - \tau )$ dominates, prompting the algorithm to prioritize $\Delta _ { 2 }$ for global exploration. In the later stage (large τ ), the weight τ increases, shifting the focus towards $\Delta _ { 1 }$ for precise local exploitation. 

Conversely, since $S _ { 1 }$ has already identified a promising convergence direction, $S _ { 2 }$ is tasked with leveraging this directional information to explore the complementary subspace. This strategy effectively avoids redundant searches and maximizes the likelihood of uniformly covering the CPF. Given that the learning direction for $S _ { 2 }$ is predetermined by $S _ { 1 }$ , the primary goal is to identify a representative solution to serve as a positional anchor. Specifically, candidate solutions in $S _ { 2 }$ associated with the i-th niche are first identified (lines 14- 15). Subsequently, the non-dominated feasible solutions within this set are extracted (line 16). If this set is non-empty, a representative is randomly selected from it; otherwise, the solution with the minimum constraint violation is chosen as the representative (lines 17-21). Finally, utilizing the promising direction determined by $S _ { 1 }$ and the selected representative solution, the orthogonal learning strategy is executed according to Eq. (9) (line 22). 

$$
\mathbf {u} _ {a u x} = \mathbf {x} _ {b} + \boldsymbol {\eta} _ {j} \cdot \mathbf {v} ^ {\perp}\tag{9}
$$

where $\mathbf { x } _ { b }$ is the best winner solution within a specific niche of $s _ { 2 } ;$ $\underline { { \mathbf { v } ^ { \perp } } }$ is the orthogonal direction derived from the convergence direction v (Eq. (8)). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/3e831ea9-6727-4c0e-809d-d04660adc717/1a068c3ce7809620472d800f41b5c6d14226c70e0dc67285ea86d1cdcfe80f9a.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/3e831ea9-6727-4c0e-809d-d04660adc717/655df80ee13ec53a53e087c0b7306973252cde63aae53451cc510881f7788446.jpg)



(a) Trend Learning



(b) Orthogonal Learning



Fig. 1. An illustration example of the proposed collaborative orthogonal learning. (a) Trend Learning in $s _ { \mathrm { 1 } } \mathrm { : }$ : The convergence direction (blue solid line) is synthesized from local and global vectors to guide offspring (green stars) to converge toward the CPS. (b) Orthogonal Learning in $\quad s _ { 2 } { \mathrm { : } }$ The complementary direction (purple solid line) is generated by orthogonalizing the transferred trend (blue dashed line) to expand diversity along the CPS.


For a clearer understanding of the proposed collaborative orthogonal learning strategy, Fig. 1 illustrates the process of executing the COL strategy within a niche. As shown in Fig. 1(a), the trend learning process of the main swarm $( S _ { 1 } )$ is depicted. Specifically, $\mathbf { x } _ { w }$ and $\mathbf { x } _ { l }$ represent the selected representative solutions. They form the basis for constructing potential convergence vectors: the local exploitation vector $( \Delta _ { 1 } )$ and the global exploration vector $( \Delta _ { 2 }$ and $\Delta _ { 2 } ^ { \prime } )$ . These vectors are dynamically synthesized to form a unified convergence direction (the blue solid line). Subsequently, experimental solutions (represented by green stars) are generated along the directions. The primary task of $ { \boldsymbol { S } } _ { 1 }$ is to identify and learn promising directions that enable rapid convergence towards the CPS, ensuring global exploration. 

In contrast, Fig. 1(b) illustrates the orthogonal learning process of the auxiliary swarm $( S _ { 2 } )$ . Through the knowledge transfer, the convergence directions identified by $S _ { 1 }$ are inherited by $S _ { 2 }$ (shown as the blue dashed lines). The selected representative solution $\mathbf { x } _ { b }$ then serves as a pivot to orthogonalize these inherited directions, yielding the complementary search directions (indicated by the purple solid lines). Similarly, $S _ { 2 }$ learns these complementary search directions, which demonstrates that $S _ { 2 }$ focuses on exploring the complementary subspace to the main trend, thereby avoiding redundant searches and enhancing diversity along the CPS. It is worth noting that $\mathbf { x } _ { b }$ is selected as the solution closest to the CPF within its niche. This implies that $\mathbf { x } _ { b }$ is likely situated in close proximity to the CPS in the decision space. Selecting these solutions as positional anchors aims to maximize the probability that the determined orthogonal directions cover the CPS, thereby effectively expanding the distribution range of the solutions. 

## C. Niche-Guided Subset Selection

To better decouple the tasks of approximating the feasible region and distributing along the CPF, $ { \boldsymbol { S } } _ { 1 }$ and $S _ { 2 }$ are assigned different strategies for environmental selection. Specifically, $S _ { 1 }$ utilizes the "Environmental Selection()" procedure, as outlined in Algorithm 4, to select the next generation. The fitness evaluation is determined by ε (lines 2). Then, the next generation is selected based on the evaluated fitness values, and their fitness values are output (line 3-11). This simple environmental selection strategy is widely used in CMOEAs [37] to increase the selection pressure on $S _ { 1 }$ and thus accelerate convergence. 

```txt
Algorithm 4: Environmental Selection

Input: O (candidate offspring), N (required size), ε (constraint relaxation)
Output: O' (offspring after selection), F' (fitness values of O')
1 O' ← ∅;
2 F ← Evaluate O according to Eq. (3) based on ε;
3 O' ← {O_i|F_i < 1, i = 1, 2, · · · , |F|};
4 if |O'| < N then
5 Sort O based on F;
6 O' ← Select N particles with smaller value of F;
7 else
8 | O' ← perform the truncation strategy of SPEA2;
9 end
10 F' ← F(O');
11 Return O', F'; 
```

To enhance $\boldsymbol { S _ { 2 } } ^ { \prime } \boldsymbol { \mathbf { s } }$ capability for extensive exploitation of the objective space, we propose a niche-guided subset selection (NGSS) strategy, as detailed in Algorithm 5. The core mechanism consists of the following three steps: 

• First-Level Subset Division (lines 1-3): First, a set of uniform weight vectors $V = \{ V ^ { 1 } , . . . , V ^ { K } \}$ is employed to divide the objective space into K distinct search spaces. The candidate pool O is first stratified into two subsets based on the CDP: Φ<sup>nf</sup> and $\Phi ^ { r e m }$ 

• Second-Level Subset Division (lines 3-9): Then, each solution in $\Phi ^ { n f }$ is then associated with a specific niche to which it has the minimum angular distance, forming K niches $( \Phi _ { 1 } ^ { n f } , \dots , \Phi _ { K } ^ { n f } )$ . To prevent the population from aggregating in easily accessible feasible regions, we enforce a local capacity constraint $\begin{array} { l } { L \ = \ \lfloor N / K \rfloor } \end{array}$ for each niche. Specifically, if the niche $\Phi _ { i } ^ { n f }$ exceeding capacity, only the L solutions with the best crowding degree are retained. Simultaneously, redundant solutions are removed from $\Phi _ { i } ^ { n f }$ to the residual set $\Phi ^ { r e m }$ 

```txt
Algorithm 5: Niche-Guided Subset Selection

Input: O (candidate offspring), N (required size), V (weight vectors)
Output: O' (offspring after selection), F' (fitness values of O')
θ ← Calculate the angle between each vector [V¹, ..., V^K] in V and each solution in O;
(Φⁿᶠ, Φ^rem) ← Divide O into two groups according to CDP;
// the non-dominated feasible solution set Φⁿᶠ and the rest of the solutions Φ^rem
(Φ₁ⁿᶠ, ..., Φ_Kⁿᶠ) ← Divide Φⁿᶠ into K niches according to θ;
for i = l to K do
    if |Φᵢⁿᶠ| > ⌊N/K⌋ then
    Φᵢⁿᶠ ← Select ⌊N/K⌋ solutions with better crowding from Φᵢⁿᶠ;
    Φ^rem ← Add the unselected solutions from Φᵢⁿᶠ to Φ^rem;
    end
end
(Φ₁^rem, ..., Φ_K^rem) ← Divide Φ^rem into K niches according to θ;
for j = l to K do
    if |Φⱼⁿᶠ| < ⌊N/K⌋ then
    if |Φⱼ^rem| > ⌊N/K⌋ - |Φⱼⁿᶠ| then
    Φⱼⁿᶠ ← Select ( ⌊N/K⌋ - |Φⱼⁿᶠ| ) solutions from Φⱼ^rem to Φⱼⁿᶠ based on the CDP;
    else
    Φⱼⁿᶠ = Φⱼⁿᶠ ∪ Φⱼ^rem;
    Φⱼⁿᶠ ← Select ( ⌊N/K⌋ - |Φⱼⁿᶠ| - |Φⱼ^rem| ) solutions from Φ^rem that have a smaller angular distance to the j-th niche to Φⱼⁿᶠ;
    end
    end
end
20 end
21 O' = Φ₁ⁿᶠ ∪ Φ₂ⁿᶠ ... ∪ Φₖⁿᶠ ;
22 F ← Evaluate O' according to Eq. (3) where ε = 0 ;
23 Return O', F'; 
```

• Third-Level Subset Division (lines 10-20): After adjusting the capacity of each niche, the crowded niches are adjusted, but sparse niches may exist. This step primarily involves dividing the residual set into several subsets to supplement the sparse niches in $\Phi _ { 1 } ^ { n f } , \dots , \Phi _ { K } ^ { n f }$ . Specifically, the updated residual set $( \Phi ^ { r e m } )$ is divided into K niches $\left( \Phi _ { 1 } ^ { r e m } , \dots , \Phi _ { K } ^ { r e m } \right)$ . Iterate through each niche. If jth niche $( \Phi _ { j } ^ { n f } )$ remains sparse $( | \Phi _ { i } ^ { n f } | ^ { - } < \lfloor N / K \rfloor )$ , it is filled by the jth residual subset $\Phi _ { j } ^ { r e m }$ . If the number of solutions in $\Phi _ { j } ^ { r e m }$ exceeds the required number $( \lfloor { \frac { N } { K } } \rfloor -$ $| \Phi _ { j } ^ { n f } | )$ , then select $\begin{array} { r l } { ( \lfloor \frac { N } { K } \rfloor - | \Phi _ { j } ^ { n f } | ) } & { { } } \end{array}$ solutions based on the CDP to fill into $\Phi _ { j } ^ { n f }$ . Otherwise, all solutions of $\Phi _ { j } ^ { r e m }$ are added to $\Phi _ { j } ^ { n f }$ . Then, $( \left\lfloor \frac { N } { K } \right\rfloor - | \Phi _ { j } ^ { n f } | - | \Phi _ { j } ^ { r e m } | )$ solutions are selected from Φ<sup>rem</sup> based on angular distance to be added to $\Phi _ { j } ^ { n f }$ 

To visualize the proposed NGSS strategy, Fig. 2 presents an illustrative example assuming a population size $N = 6$ and $K = 3$ niches. There are $2 N = 1 2$ candidate solutions (x<sub>1</sub>- $x _ { 1 2 } )$ and K weight vectors $( V ^ { 1 } , V ^ { 2 } , V ^ { 3 } )$ . The capacity limit for each niche is $L = \lfloor N / K \rfloor = 2$ 

Fig. 2(a) illustrates the first-level subset division. All solutions are divided into two subsets based on the CDP: the non-dominated feasible set $\Phi ^ { n f } \left( x _ { 1 } , x _ { 2 } , x _ { 3 } \right)$ and the remainder set $\Phi ^ { r e m } \left( x _ { 4 } – x _ { 1 2 } \right)$ . This step prioritizes feasibility and convergence. 

Fig. 2(b) shows the second-level subset division. Solutions in $\bar { \Phi ^ { n f } }$ are assigned to niches based on angular distance, $\Phi _ { 1 } ^ { n f } = \Phi _ { 3 } ^ { n f } = \breve { \emptyset } , \Phi _ { 2 } ^ { n f } = \left\{ x _ { 1 } , x _ { 2 } , x _ { 3 } \right\}$ . Since the number of solutions in $\Phi _ { 2 } ^ { n f }$ exceeds $L \left( 3 > 2 \right)$ , the most crowded solution x<sub>2</sub> is removed and added to Φ<sup>rem</sup>. Consequently, $\overline { { \Phi _ { 2 } ^ { n J } } }$ retains $\overline { { \{ x _ { 1 } , x _ { 3 } \} } }$ . This level effectively controls population density in 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/3e831ea9-6727-4c0e-809d-d04660adc717/c82c05fecf965f683b3a9d12219fb25cbc6efc40681bf849e5763dda43380f5b.jpg)



(a) First-Level Subset Division


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/3e831ea9-6727-4c0e-809d-d04660adc717/32ce594308538070c8fe6536321023d441d6ae2833f9841f7d106fca4abd81bf.jpg)



(b) Second-Level Subset Division


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/3e831ea9-6727-4c0e-809d-d04660adc717/a613c659242e4f9643aa2660bbe4263001c99046ba26b1f3077b6b9a807f635c.jpg)



(c) Third-Level Subset Division



Fig. 2. An illustration example of the proposed niche-guided subset selection.


favored objective regions. 

Fig. 2(c) depicts the third-level subset division. The solutions in Φ<sup>rem</sup> are also categorized into niches: $\Phi _ { 1 } ^ { r e m } = $ $\{ x _ { 4 } , x _ { 5 } , x _ { 1 0 } \} , \Phi _ { 2 } ^ { r e m } = \{ x _ { 2 } , x _ { 6 } , . . . , x _ { 1 2 } \}$ , and $\Phi _ { 3 } ^ { r e m } = \{ x _ { 9 } \}$ Then the sparse niches are filled (where size $< L )$ . Specifically, for the empty $\Phi _ { 1 } ^ { n f }$ , solutions $x _ { 4 }$ and $x _ { 5 }$ are selected from $\Phi _ { 1 } ^ { r e m }$ based on CDP. For the empty $\Phi _ { 3 } ^ { n f }$ $x _ { 9 }$ from $\Phi _ { 3 } ^ { r e m }$ is added first. Since it still requires one more solution, $x _ { 8 }$ is selected from the remaining candidates due to its smaller angular distance to $V ^ { 3 }$ . Finally, the updated niches are $\Phi _ { 1 } ^ { n f } = \{ \bar { x _ { 4 } } , x _ { 5 } \} , \Phi _ { 2 } ^ { n f } = \{ x _ { 1 } , x _ { 3 } \}$ , and $\Phi _ { 3 } ^ { n f } \dot { { \bf \Phi } } = \{ x _ { 8 } , x _ { 9 } \}$ . The third level ensures that the swarm does not exhibit preferential search for specific niches. In summary, the proposed NGSS strategy aims to uniformly search the objective space while ensuring a degree of convergence pressure, thereby enhancing the ability to capture discrete CPFs. 

## D. Computational Complexity

For DSOCOL, the time complexity of parameter and swarm initialization is $O ( M N )$ and $O ( D N )$ , respectively. The complexity of the swarm update including the COL strategy is dominated by $O ( M N ^ { 2 } )$ . The time complexity of environmental selection for the main swarm is $O ( N ^ { 3 } )$ , primarily attributed to the truncation strategy of SPEA2. The time complexity of the NGSS strategy for the auxiliary swarm is $O ( M N ^ { 2 } + D N )$ ). As a result, the overall computational complexity of DSOCOL in the worst case is $O ( N ^ { 3 } )$ . Due to space limitations, a detailed computational complexity analysis is provided in Supp-Section III of the Supplementary Files. 

## IV. EXPERIMENTAL STUDIES

## A. Experimental Settings

1) Benchmark Instances: To measure the overall performance of the proposed DSOCOL in addressing problems with varying landscape, we conducted comprehensive experiments on 33 CMOP instances belonging to DAS-CMOPs [38], C-DTLZ [39], DC-DTLZ [40] and LIR-CMOPs [41]. Beyond the standard settings with decision variables $D < 1 0 0$ , we further extended these four benchmarks to large-scale configurations (D = 500 and D = 1000) to verify the algorithm’s efficacy in high-dimensional search spaces. Furthermore, a specific test suite named FCPs [42] was introduced to assess the robustness of DSOCOL against deceptive constraints. All experiments were performed on PlatEMO [43]. 

2) CMOEAs for Comparison: To comprehensively evaluate the performance of the proposed DSOCOL, we compared it against nine state-of-the-art CMOEAs: APSEA [44], C3M [45], CMOEMT [11], DRLOS-EMCMO [30], IM-C-MOEA/D [46], CMOCSO [28], DVCEA [47], LCMEA [48], and POCEA [32]. The selection of these comparison algorithms was driven by the need to assess algorithmic robustness across diverse constraint landscapes and varying problem scales. Specifically, APSEA, C3M, CMOEMT, DRLOS-EMCMO, and IM-C-MOEA/D serve as representative solvers designed for standard-scale CMOPs; CMOCSO and DVCEA are included for their versatility in handling both standard and large-scale CMOPs; while LCMEA and POCEA are selected for their design for large-scale CMOPs. 

3) Parameter Settings: In the experimental setup, the population size N is set to 100. All comparison algorithms are run independently on each problem 30 times. The termination criterion is determined by the maximum number of function evaluations $( \underline { { F } } \underline { { E } } _ { m a x } )$ , which is set to 100, 000 for standardscale benchmarks, and extended to 200, 000 and 300, 000 for large-scale benchmarks with $D = 5 0 0$ and $D = 1 0 0 0$ respectively. All the parameters of GA and DE operator are the same as the previous references [8], [49] or original studies of the compared CMOEAs. Furthermore, to ensure the consistency of experimental conditions and the comparability of results, the parameters for all comparison algorithms are strictly configured in accordance with their original publications. 

4) Performance Indicators: To clarify the performance of the methods, we use two popular performance metrics: Inverted Generation Distance (IGD) [50] and Hypervolume (HV) [51]. The Subsection II of the Supplementary Files provides specific information and formulas for these two indicators. In addition, the performance indicator results were statistically analyzed using KEEL software [52] at a significance level of 0.05. Based on the Wilcoxon analysis, the notations $^ { \prime \prime } { + ^ { \prime \prime } } ,$ $U { \mathrm { - } } ^ { \prime \prime } , \mathrm { a n d } ^ { \prime \prime } \approx ^ { \prime \prime }$ signify that the performance of the comparative algorithm is significantly superior to, inferior to, or statistically indistinguishable from DSOCOL. Furthermore, “NaN” implies that the algorithm failed to find any feasible solution. 

## B. Comparison Studies

1) Comparison on Standard CMOPs: To comprehensively assess the performance of DSOCOL on standardscale CMOPs, the proposed DSOCOL is compared with nine state-of-the-art CMOEAs and the statistical results on four benchmark suites with the standard dimensions are presented in Tables S-V and S-VI of the Supplementary Files, respectively. For a more intuitive view, we summarize all the results in Table I. In terms of IGD, DSOCOL demonstrated significant performance advantages over the comparison methods on 19 of the 33 problems. DVCEA achieved the best results in six instances. Concerning HV, DSOCOL outperformed APSEA, C3M, CMOEMT, DRLOS-EMCMO, IM-C-MOEA/D, CMOCSO, DVCEA, LCMEA and POCEA in 25, 31, 17, 22, 32, 16, 18, 32 and 33 of the 33 instances. The reasons for the poor performance of these comparison algorithms can be analyzed as follows. 

For APSEA, it employs an adaptive population sizing strategy within a multi-population framework to balance objective optimization and constraint satisfaction. By dynamically adjusting the number of function evaluations for different populations, it achieves high efficiency on problems with simple feasibility landscapes. However, its reproduction relies on traditional operators that lack explicit directional guidance. On problems with extremely narrow feasible regions or large infeasible regions like LIR-CMOPs, APSEA struggles to guide the population across infeasible barriers, often leading to premature convergence in local feasible regions. 

C3M adopts a multistage framework that prioritizes constraints based on the relationship between single-constraint Pareto fronts (PFs) and the common PF. This allows the algorithm to explore the objective space more freely in the early stages. Nevertheless, its performance is sensitive to the transition between stages. This mechanism risks partial CPF exploration under insufficient evaluations, as full constraints are only considered in later stages. In cases where the CPF is far from the UPF, the population may get trapped in distant infeasible regions or local optima, making it difficult to reconverge to the true CPF within limited evaluations. 

CMOEMT transforms a CMOP into a multitasking problem with three distinct tasks (CDP, constraint neglect, and ε- relaxation) and facilitates knowledge transfer among them. While solution transfer accelerates convergence for conventional problems, the transferred knowledge remains limited to the spatial positions of individuals. For problems with narrow or disjoint feasible regions, such as LIR-CMOPs, this passive transfer mechanism may induce negative transfer or fail to capture potential evolutionary trends, leading to diversity loss or insufficient CPF coverage. 

DRLOS-EMCMO utilizes deep reinforcement learning to adaptively select the most suitable evolutionary operator based on the population’s real-time state. This intelligent selection improves the algorithm’s flexibility across different search stages. However, the decision is primarily based on historical performance feedback and lacks a geometric understanding of the search space. Specifically different problems indeed exhibit preferences for different operators. However, the preference for rapid convergence and constraint satisfaction prevents uniform exploration of the objective space, resulting in missing some CPF fragments. 

IM-C-MOEA/D integrates inverse modeling with the framework based on decomposition, attempting to map the objective space back to the decision space to guide offspring generation. This method is highly effective for problems with a clear Pareto optimal set and Pareto front mapping. However, when confronting complex constraints that result in irregular feasible boundaries, the accuracy of the inverse model deteriorates significantly. The inaccurate model guidance may lead the population far away from the feasible region, leading to dual failures in convergence and diversity. 

CMOCSO leverages competitive learning to accelerate the search process. Its robust convergence capability makes it competitive for problems requiring fast approximations or multimodal characteristics, such as DTLZs. Nevertheless, its search behavior is often characterized by a single-path convergence toward the feasible regions. In landscapes with multiple local optima or complex infeasible regions, CMOCSO lacks a mechanism to explore complementary subspaces, limiting its ability to maintain a wide distribution along the CPF. 

DVCEA classifies decision variables into constraint-related and constraint-irrelevant categories and applies targeted variation strategies. However, for DAS-CMOPs with convergencehardness, diversity-hardness, and feasibility-hardness, the decoupling of constraints may come at the expense of global convergence. Due to the intense coupling among variables, this strategy may break the evolutionary relationships, leading to a loss of convergence performance while attempting to satisfy complex constraints. Moreover, it lacks global search mechanism to perform the uniform search required to locate the narrow and discrete regions of LIR-CMOPs. 

LCMEA employs a sampling approach to generate offspring in preferred directions. However, the sampling points rely on local population distribution. For problems like LIR-CMOPs with fragmented CPFs and narrow feasible regions, local sampling fails to identify the global direction needed to cross large infeasible barriers, resulting in incomplete CPF coverage. 


TABLE I



A SHORT SUMMARY OF COMPARISON STUDY RESULTS ON FOUR BENCHMARK SUITES WITH STANDARD DIMENSIONS


<table><tr><td colspan="2">DSOCOL vs (+/-/=)</td><td>APSEA</td><td>C3M</td><td>CMOEMT</td><td>DRLOS-EMCMO</td><td>IM-C-MOEA/D</td><td>CMOCSO</td><td>DVCEA</td><td>LCMEA</td><td>POCEA</td></tr><tr><td rowspan="2">DAS-CMOPs</td><td>IGD</td><td>3/6/0</td><td>0/7/2</td><td>3/6/0</td><td>0/4/5</td><td>0/8/1</td><td>1/5/3</td><td>3/5/1</td><td>0/9/0</td><td>0/9/0</td></tr><tr><td>HV</td><td>3/6/0</td><td>0/7/2</td><td>3/3/3</td><td>0/4/5</td><td>0/8/1</td><td>0/4/5</td><td>3/3/3</td><td>0/9/0</td><td>0/9/0</td></tr><tr><td rowspan="2">C-DTLZs</td><td>IGD</td><td>1/1/2</td><td>1/3/0</td><td>1/3/0</td><td>1/1/2</td><td>0/4/0</td><td>1/0/3</td><td>1/2/1</td><td>1/2/1</td><td>1/2/1</td></tr><tr><td>HV</td><td>2/1/1</td><td>0/4/0</td><td>0/1/3</td><td>1/2/1</td><td>0/4/0</td><td>1/0/3</td><td>2/1/1</td><td>0/3/1</td><td>0/4/0</td></tr><tr><td rowspan="2">DC-DTLZs</td><td>IGD</td><td>1/3/2</td><td>0/6/0</td><td>1/3/2</td><td>1/3/2</td><td>0/6/0</td><td>2/1/3</td><td>1/4/1</td><td>0/6/0</td><td>0/6/0</td></tr><tr><td>HV</td><td>1/4/1</td><td>0/6/0</td><td>2/2/2</td><td>0/3/3</td><td>0/6/0</td><td>1/1/4</td><td>0/3/3</td><td>0/6/0</td><td>0/6/0</td></tr><tr><td rowspan="2">LIR-CMOPs</td><td>IGD</td><td>0/14/0</td><td>0/14/0</td><td>0/11/3</td><td>0/12/2</td><td>0/14/0</td><td>1/11/2</td><td>2/11/1</td><td>0/14/0</td><td>0/13/1</td></tr><tr><td>HV</td><td>0/14/0</td><td>0/14/0</td><td>0/11/3</td><td>0/13/1</td><td>0/14/0</td><td>2/9/3</td><td>2/11/1</td><td>0/14/0</td><td>0/14/0</td></tr></table>

Authorized licensed use limited to: XIDIAN UNIVERSITY. Downloaded on July 01,2026 at 07:47:47 UTC from IEEE Xplore. Restrictions apply. © 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information. 

POCEA uses a pairing strategy to reproduce promising offspring by emphasizing useful infeasible solutions. However, its pairing process is primarily based on static Euclidean distance and fails to capture dynamic trend information. When facing the combined three-hardness of DAS-CMOP, simple pairing is insufficient to generate offspring that can simultaneously penetrate multiple constraints and maintain distribution. 

The aforementioned CMOEAs encounter significant challenges when confronting diverse and complex constraint landscapes. To visually demonstrate the performance advantages of DSOCOL, we present the final solution sets across all test instances in Figs. S-1 to S-3 of the Supplementary Files. As shown, DSOCOL achieved complete coverage of the CPF with uniform distribution on nearly all problems. The proposed DSOCOL achieved outstanding performance because DSOCOL simultaneously extracts promising evolutionary directions while deeply decoupling the tasks of convergence and diversity. Specifically, instead of merely performing traditional offspring generation, the main swarm $S _ { 1 }$ identifies valuable convergence directions that facilitate crossing vast infeasible regions through trend learning. Simultaneously, the auxiliary swarm $S _ { 2 }$ minimizes redundant searches by receiving this trend information and conducting exploration within complementary orthogonal subspaces, thereby achieving efficient diversity exploitation. Coupled with the NGSS strategy for three-level subset division mechanism, DSOCOL ensures a uniform search across the objective space, significantly enhancing its capability to capture discrete and fragmented CPFs. This synergistic mechanism allows DSOCOL to exhibit higher robustness and superior performance than these comparison algorithms when handling CMOPs with different landscapes. 

Furthermore, to further verify the superiority of DSOCOL, we conducted the Friedman test with Holm correction to perform the statistical analysis, which is presented in Table II. From the table, DSOCOL got the best rankings on IGD and HV. Furthermore, all $p \mathrm { - }$ values were below a significance level of 0.05, which indicating that DSOCOL significantly outperforms the other nine compared CMOEAs. These results further validate the effectiveness of DSOCOL in balancing feasibility, convergence, and diversity when dealing with CMOPs with different characteristics. 


TABLE II



AVERAGE RANKINGS AND p-VALUES OF IGD AND HV BY THE FRIEDMAN TEST OF DSOCOL AND OTHER COMPARED CMOEAS ON FOUR STANDARD BENCHMARK SUITES.


<table><tr><td>Algorithm</td><td>IGD Ranking</td><td>p-value</td><td>HV Ranking</td><td>p-value</td></tr><tr><td>APSEA</td><td>6.5455</td><td>0.000000</td><td>6.4091</td><td>0.000000</td></tr><tr><td>C3M</td><td>5.4545</td><td>0.000006</td><td>5.3182</td><td>0.000080</td></tr><tr><td>CMOEMT</td><td>4.4848</td><td>0.001319</td><td>4.0909</td><td>0.021616</td></tr><tr><td>DRLOS-EMCMO</td><td>3.7576</td><td>0.025347</td><td>4.0303</td><td>0.026709</td></tr><tr><td>IM-C-MOEA/D</td><td>9.5303</td><td>0.000000</td><td>9.5303</td><td>0.000000</td></tr><tr><td>CMOCSO</td><td>3.5909</td><td>0.044171</td><td>3.8939</td><td>0.042074</td></tr><tr><td>DVCEA</td><td>4.9545</td><td>0.000122</td><td>4.9394</td><td>0.000592</td></tr><tr><td>LCMEA</td><td>7.0909</td><td>0.000000</td><td>6.8030</td><td>0.000000</td></tr><tr><td>POCEA</td><td>7.5000</td><td>0.000000</td><td>7.6061</td><td>0.000000</td></tr><tr><td>DSOCOL</td><td>2.0909</td><td></td><td>2.3788</td><td></td></tr></table>

2) Comparison on Large-Scale CMOPs: For validating the proposed DSOCOL’s performance in handling CMOPs with large-scale decision variable landscapes, we further compared DSOCOL with nine CMOEAs on the four benchmarks with large-scale decision variables (D = 500 and 1000). The statistical results for IGD and HV across two dimensions are reported in Tables S-VII and S-VIII of the Supplementary Files. The summary results are reported in Table S-II of the Supplementary Files, indicating that DSOCOL obtained clear advantages over most compared CMOEAs on both IGD and HV, demonstrating its strong scalability in handling CMOPs with large-scale search space landscape. Furthermore, the Friedman test results in Table S-III of the Supplementary Files show that DSOCOL achieved the best average rankings on both IGD and HV, with all p-values below 0.05. Therefore, the experimental results on large-scale CMOPs statistically verify the superiority of DSOCOL. Due to space limitations, the detailed analysis of the experimental results is provided in Supp-Section IV of the Supplementary Files. 

3) Comparison on Special CMOPs: To further evaluate the robustness of DSOCOL, we conducted experiments on the FCP benchmark suite with special difficulties. Distinct from conventional benchmarks, the FCP series initializes the population in the transition zone between the UPF and the CPF. A defining characteristic of this suite is the presence of fraudulent constraints, where the constraint violation degrees exhibit non-monotonicity. These deceptive landscapes act as evolutionary traps, requiring algorithms to maintain individuals with suboptimal objective values and lower feasibility to traverse the fraudulent regions. The statistical IGD and HV results on FCP1-5 are summarized in Tables S-IX and S-X of the Supplementary Files, with Fig. S-6 visualizing the final solution distribution on FCP4. It is evident that among all compared methods, only LCMEA, which utilizes a sampling approach, managed to yield feasible solutions across all problems. Nevertheless, DSOCOL got the best results on four out of five instances. Conversely, all other competing algorithms failed to identify any feasible regions on FCP1-4. This is because these methods prioritize constraint satisfaction or objective optimization and perform knowledge transfer solely based on positional information of solutions. Consequently, they failed to fully explore the objective space, leading to their failure to handle problems with deceptive constraints. These results further verify that the proposed strategies can effectively handle CMOPs with diverse landscapes. 

## C. Ablation Studies

In this section, we set up five variants to further confirm the effectiveness and necessity of the different components. 

• DSOCOL1: The auxiliary swarm $S _ { 2 }$ employs Algorithm 4 instead of the proposed NGSS strategy; 

• DSOCOL2: The winners groups of both swarms are updated solely through polynomial mutation; 

• DSOCOL3: The proposed COL strategy is not implemented; 

• DSOCOL4: The trend learning strategy is not implemented (without Eq. (8)); 

• DSOCOL5: The orthogonal learning strategy is not implemented (without Eq. (9)); 

To fully validate the overall performance of the proposed methods, we compared DSOCOL and its five variants under both standard dimensions and 1000 dimensions across four benchmarks. Table III provides all the summary results of the ablation experiments and the detailed results of IGD and HV is presented in Tables S-XI and S-XII of the Supplementary Files. The following conclusions can be drawn from these results: (1) DSOCOL significantly outperformed DSOCOL1, which demonstrates that the proposed NGSS strategy can maintain comprehensive exploration for the whole objective space. (2) Compared to DSOCOL2, DSOCOL performed superior or competitive performance in 78% of cases, validating the effectiveness of further improvements to the winner group. (3) DSOCOL outperformed DSOCOL3, DSOCOL4, and DSOCOL5 confirming the effectiveness of the proposed COL strategy. 


TABLE III



SUMMARY RESULTS OF DSOCOL AND ITS FIVE VARIANTS ON THE FOUR BENCHMARK SUITES WITH VARYING DIMENSIONS.


<table><tr><td></td><td>IGD(+/ - / ≈)</td><td>HV(+/ - / ≈)</td></tr><tr><td>DSOCOL1 vs. DSOCOL</td><td>5/25/36</td><td>8/19/39</td></tr><tr><td>DSOCOL2 vs. DSOCOL</td><td>17/33/16</td><td>12/27/27</td></tr><tr><td>DSOCOL3 vs. DSOCOL</td><td>8/36/22</td><td>1/32/33</td></tr><tr><td>DSOCOL4 vs. DSOCOL</td><td>5/44/17</td><td>1/40/25</td></tr><tr><td>DSOCOL5 vs. DSOCOL</td><td>4/20/43</td><td>1/22/43</td></tr></table>

To clarify the impact of the proposed NGSS strategy, we plot the populations, with the median IGD among 30 runs on LIR-CMOP3, obtained by DSOCOL1 and DSOCOL are presented in Fig. 3. It was evident that although DSOCOL1 successfully converged to the feasible region, it exhibited poor performance as it only captured a limited portion of the CPF. This failure was primarily attributed to the absence of the NGSS strategy, which caused DSOCOL1 to converge rapidly in a clustered state that led the population into local feasible regions. Specifically, for problems like LIR-CMOP3 with discrete CPFs, once DSOCOL1 became trapped in local optima, it lacked the capability to re-distribute across the objective space to locate all fragmented CPF segments. In contrast, DSOCOL, equipped with the NGSS strategy, facilitated a uniform search throughout the objective space and yielded highly competitive results, which provides compelling evidence of the effectiveness and necessity of NGSS strategies in maintaining population diversity. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/3e831ea9-6727-4c0e-809d-d04660adc717/48728ca166c4ce39cd92ec8ffee8a039806e8ad490eb8f0a72b2beea09cd1235.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/3e831ea9-6727-4c0e-809d-d04660adc717/4a5da7f6267d7d02fbc7166060f981d03b93de5c003b662e16b8b60afc880c26.jpg)



Fig. 3. Populations obtained by DSOCOL1 and DSOCOL on LIR-CMOP3.


Furthermore, the effectiveness of the proposed improvements to the winner group and the COL strategy was further validated through Fig. 4. Clearly the convergence speed of DSOCOL2, which only performed polynomial mutation on the winner group, was significantly lower than DSOCOL and DSOCOL1, demonstrating that the performance of CSO operator can be effectively enhanced by further improving the winner groups through Eqs. (5) and (6). Likewise, the convergence rate of DSOCOL3 without the COL strategy deteriorated significantly and was markedly lower than those of DSOCOL, DSOCOL1, and DSOCOL2, which incorporated the COL strategy. These results confirmed that the COL strategy can successfully capture promising evolutionary directions and accelerate population convergence toward the CPF. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/3e831ea9-6727-4c0e-809d-d04660adc717/65fb1183db26d335c01fc346b74d46e341cea819c780fe1c34bc86c5e3a73515.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/3e831ea9-6727-4c0e-809d-d04660adc717/831930c121c01ce279f42cb3c4148f6f5e3dc68cd50c343655d6770974d2e4dd.jpg)



Fig. 4. Convergence results of DSOCOL and DSOCOL1-3 on DC1-DTLZ3 and LIR-CMOP11.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/3e831ea9-6727-4c0e-809d-d04660adc717/86ee2db037d29250cf500d0815c045c01b53d5cd209b7ed3fe52dce6076634f1.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/3e831ea9-6727-4c0e-809d-d04660adc717/81d12afe78fa370bf1615cebdfee1fafa4c54b99257d0abc30f1fee30394aa4a.jpg)



Fig. 5. Convergence results of DSOCOL and DSOCOL4-5 on LIR-CMOP10 and DC3-DTLZ1.


To further investigate the effectiveness and respective roles of trend learning and orthogonal learning in the COL strategy, Figs. 5 and 6 are presented for detailed analysis. As shown in Fig. 5, DSOCOL4, which excludes trend learning component, exhibited a significantly slower convergence speed than DSOCOL5 and DSOCOL, both of which incorporated trend learning. Moreover, DSOCOL converged slightly slower than DSOCOL5 on these two problems. This was because DSOCOL allocated part of the function evaluations to orthogonal learning in order to explore complementary search regions, whereas DSOCOL5 focused entirely on trend learning. These results indicated that trend learning is able to identify promising evolutionary directions, thereby accelerating the convergence toward the CPF. Fig. 6 illustrated the final populations obtained by DSOCOL5 and DSOCOL on C-DTLZ2 and LIR-CMOP3. It could be observed that both DSOCOL5 and DSOCOL, equipped with trend learning, achieved similar convergence performance. However, the final population achieved by DSOCOL, which additionally incorporated orthogonal learning, was noticeably more uniformly distributed than that of DSOCOL5. This finding suggested that orthogonal learning is capable of exploring search regions complementary to those identified by trend learning, thereby enhancing the exploration of neighborhoods around exploited areas and ultimately improving population diversity. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/3e831ea9-6727-4c0e-809d-d04660adc717/e393b7911b32a55b0beabe714e3c9b1e79a2fe7eaad9df3cb8f54532ffa8e14c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/3e831ea9-6727-4c0e-809d-d04660adc717/111e3e92a0391719768880639ef872f968da07d78eed3429b124328a5c90d693.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/3e831ea9-6727-4c0e-809d-d04660adc717/b1176e60d017a625a0b8823eda13432d0e2dede9e5a3f150050bad455d41ad4b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/3e831ea9-6727-4c0e-809d-d04660adc717/47c0f14ae4d80a6ad08aa80a10afa2b323ba092ab8fa9a352b3a17058e88a9e1.jpg)



Fig. 6. Populations obtained by DSOCOL5 and DSOCOL on C-DTLZ2 and LIR-CMOP3.


Eventually, we conducted the Friedman test on two metrics to further verify the necessity and effectiveness of the different components of DSOCOL. Table IV displays the statistical results, showing that DSOCOL always has the highest rankings. The p-values were less than 0.05, which indicates that DSOCOL is significantly better than these variants. Therefore, the experimental results further verify the necessity and effectiveness of the proposed NGSS strategy, the winner group improvement mechanism, and the COL strategy. These proposed methods ensure that the swarm can converge rapidly toward the CPF while maintaining extensive exploration across the entire objective space, thereby enhancing the algorithm’s capability for a comprehensive search along the CPF. 


TABLE IV



AVERAGE RANKINGS AND p-VALUES OF DSOCOL AND FIVE VARIANTS BY THE FRIEDMAN TEST ON FOUR BENCHMARK SUITES WITH VARYING SCALES.


<table><tr><td>Algorithm</td><td>IGD Ranking</td><td>p-value</td><td>HV Ranking</td><td>p-value</td></tr><tr><td>DSOCOL1</td><td>3.5909</td><td>0.000217</td><td>3.2955</td><td>0.040652</td></tr><tr><td>DSOCOL2</td><td>3.5682</td><td>0.000285</td><td>3.4773</td><td>0.009178</td></tr><tr><td>DSOCOL3</td><td>3.6894</td><td>0.000063</td><td>3.9773</td><td>0.000035</td></tr><tr><td>DSOCOL4</td><td>4.6667</td><td>0.000000</td><td>4.5985</td><td>0.000000</td></tr><tr><td>DSOCOL5</td><td>3.0985</td><td>0.028769</td><td>3.0227</td><td>0.226421</td></tr><tr><td>DSOCOL</td><td>2.3864</td><td></td><td>2.6288</td><td></td></tr></table>

In addition, it is worth noting that DSOCOL4 obtained a worse ranking than DSOCOL3. Although DSOCOL3 did not incorporate the COL strategy, it was still able to achieve gradual convergence by relying on the environmental selection mechanism. In contrast, the COL strategy in DSOCOL4 only included orthogonal learning without trend learning, which caused the population to consume additional function evaluations to explore complementary regions via orthogonal learning when it was unable to converge rapidly toward the CPF. Under such slow convergence conditions, further intensifying the exploration of complementary regions to enhance diversity became detrimental to overall performance, ultimately leading DSOCOL4 to achieve the worst ranking among the compared variants. This result also indirectly verified that the search directions explored by orthogonal learning are different from and complementary to the evolutionary directions identified by trend learning for rapid convergence, indicating that the COL strategy can achieve its best performance only when orthogonal learning and trend learning operate in a coordinated manner to properly balance convergence and diversity. 

## D. Parameter Analysis

DSOCOL involves two parameters: the execution frequency of the COL strategy $( T _ { C O L } )$ and the number of niches (K). Detailed statistical results for IGD and HV are provided in Table S-XIII and S-XIV of the Supplementary Files. In this study, $T _ { C O L }$ and K are set to 75 and $\lfloor \frac { N } { 1 0 } \rfloor$ , respectively. Due to space limitations, a detailed analysis regarding the characteristics and sensitivity of these two parameters is provided in Supp-Section V of the Supplementary Files. 

## E. Real-World Application

In contrast to the previous benchmark suite instances, realworld CMOPs (RWCMOPs) show distinct characteristics. To evaluate DSOCOL’s performance in practical applications, we further compared the performance of DSOCOL with the nine comparison algorithms in ten real-world applications in four different domains, which are as follows. 

• Mechanical Design Problem: Pressure Vessel Design [53], Four Bar Plane Truss [54], Front Rail Design [55]. 

• Chemical Engineering Problems: Heat Exchanger Network Design [56], Reactor Network Design [57]. 

• Process and Synthesis Problems: Process Design Problem [58], Process Synthesis Problem [59], Process Flow Sheeting Problem [60]. 

• Power Electronics Problems: Synchronous Optimal Pulse-width Modulation of 3-level Inverters [61], Synchronous Optimal Pulse-width Modulation of 7-level Inverters [62]. 

Since the true CPFs of these real-world applications are not known, we use the HV indicator to measure the performance of CMOEAs. Compared to the previous benchmark instances, these RWCMOPs are more challenging in several respects. First, the objective functions in practical problems may exhibit significantly different numerical scales, making it difficult for an algorithm to establish a credible boundary without sufficient convergence pressure. Second, these applications may involve highly nonlinear constraints and irregular Pareto front, under which the population can easily be trapped in local feasible regions or even fail to obtain feasible solutions. Third, some real-world problems are also associated with large or strongly coupled search spaces, which further increases the difficulty of locating high-quality feasible regions efficiently. Under such conditions, the advantage of DSOCOL lies in its ability to simultaneously ensure convergence reliability and distribution quality. Specifically, the COL strategy enables the main swarm to learn promising evolutionary directions and maintain sufficient convergence pressure to approach a reliable objective boundary, while the auxiliary swarm explores complementary subspaces to reduce redundant search. In addition, the NGSS strategy helps preserve uniformity in the objective space, which is particularly important when the practical Pareto front is irregular or partially disconnected. As shown in Table V, our proposed method achieves the overall highest quality results in these practical applications, with DVCEA following closely. The results validate that DSOCOL successfully maintains a balance between convergence, feasibility, and diversity, even in the context of RWCMOPs. In general, the superior or competitive performance of DSOCOL in solving real-world applications is demonstrated, compared with other methods. 


TABLE V



THE MEDIAN HV RESULTS OBTAINED BY DSOCOL AND ALL COMPARISON ALGORITHMS ON TEN RWCMOPS AMONG 30 RUNS. THE BEST RESULT IN EACH ROW IS HIGHLIGHTED


<table><tr><td>Problem</td><td>APSEA</td><td>C3M</td><td>CMOEMT</td><td>DRLOS-EMCMO</td><td>IM-C-MOEA/D</td><td>CMOCSO</td><td>DVCEA</td><td>LCMEA</td><td>POCEA</td><td>DSOCOL</td></tr><tr><td>Pressure Vessel Design</td><td>6.0454e-1 (9.59e-4) -</td><td>6.0138e-1 (2.68e-3) -</td><td>5.9416e-1 (2.07e-3) -</td><td>6.0514e-1 (9.75e-4) -</td><td>5.8511e-1 (6.09e-3) -</td><td>5.4646e-1 (4.50e-2) -</td><td>6.0601e-1 (3.80e-4) ≈</td><td>6.0205e-1 (2.11e-3) -</td><td>5.8100e-1 (7.75e-3) -</td><td>6.0603e-1 (8.24e-4)</td></tr><tr><td>Four Bar Plane Truss</td><td>4.0936e-1 (1.75e-4) -</td><td>4.0956e-1 (1.29e-4) -</td><td>4.0943e-1 (1.13e-4) -</td><td>4.0972e-1 (8.71e-5) ≈</td><td>4.0196e-1 (1.84e-3) -</td><td>4.0962e-1 (1.27e-4) -</td><td>4.0939e-1 (1.59e-4) -</td><td>4.1020e-1 (1.02e-4) +</td><td>3.3625e-1 (1.22e-2) -</td><td>4.0970e-1 (7.78e-5)</td></tr><tr><td>Front Rail Design</td><td>4.0505e-2 (5.44e-6) ≈</td><td>4.0508e-2 (4.01e-6) ≈</td><td>4.0509e-2 (4.59e-6) ≈</td><td>4.0504e-2 (6.29e-6) ≈</td><td>3.9969e-2 (2.10e-4) -</td><td>4.0508e-2 (5.64e-6) ≈</td><td>4.0506e-2 (1.52e-5) ≈</td><td>4.0492e-2 (3.38e-5) ≈</td><td>4.0323e-2 (5.99e-5) -</td><td>4.0510e-2 (4.57e-6)</td></tr><tr><td>Heat Exchanger Ntweork Design</td><td>NaN (NaN) -</td><td>NaN (NaN) -</td><td>NaN (NaN) -</td><td>NaN (NaN) -</td><td>NaN (NaN) -</td><td>NaN (NaN) -</td><td>1.0000e+0 (0.00e+0) -</td><td>8.3095e-1 (3.37e-1) -</td><td>2.0000e-1 (4.47e-1) -</td><td>5.0364e+4 (7.12e+4)</td></tr><tr><td>Reactor Network Design</td><td>NaN (NaN) -</td><td>1.8758e+0 (3.50e+0) ≈</td><td>9.9451e-1 (9.19e-3) ≈</td><td>8.3152e-1 (2.51e-1) -</td><td>NaN (NaN) -</td><td>9.9856e-1 (0.00e+0) ≈</td><td>9.3315e-1 (7.21e-2) -</td><td>7.0574e-1 (2.92e-1) -</td><td>1.0365e+0 (1.56e-1) -</td><td>3.4847e+0 (6.12e+0)</td></tr><tr><td>Process Design Problem</td><td>1.3839e-1 (2.77e-2) ≈</td><td>1.4523e-1 (6.23e-3) +</td><td>1.4802e-1 (2.67e-2) ≈</td><td>1.4537e-1 (1.03e-2) +</td><td>1.8376e-1 (2.07e-2) +</td><td>9.8472e-2 (1.35e-2) -</td><td>1.3715e-1 (1.09e-2) ≈</td><td>1.2306e-1 (2.81e-2) -</td><td>1.3076e-1 (3.21e-2) ≈</td><td>1.3543e-1 (1.63e-2)</td></tr><tr><td>Process Synthesis Problem</td><td>7.6198e-1 (1.93e-2) -</td><td>7.6952e-1 (7.32e-3) -</td><td>7.5167e-1 (2.88e-2) -</td><td>7.72851e-1 (4.96e-3) -</td><td>6.9682e-1 (1.00e-1) -</td><td>6.2336e-1 (5.38e-2) -</td><td>7.8209e-1 (2.69e-3) ≈</td><td>7.2110e-1 (7.74e-2) -</td><td>3.3347e-1 (2.35e-1) -</td><td>7.8294e-1 (2.19e-3)</td></tr><tr><td>Process Flow Sheeting Problem</td><td>7.6198e-1 (1.93e-2) -</td><td>7.6952e-1 (7.32e-3) -</td><td>7.5167e-1 (2.88e-2) -</td><td>7.7851e-1 (4.96e-3) -</td><td>6.9682e-1 (1.00e-1) -</td><td>6.2336e-1 (5.38e-2) -</td><td>7.8209e-1 (2.69e-3) -</td><td>7.2110e-1 (7.74e-2) -</td><td>3.3347e-1 (2.35e-1) -</td><td>7.8606e-1 (1.17e-3)</td></tr><tr><td>Synchronous Optimal Pulse-width Modulation of 3-level Inverters</td><td>NaN (NaN) -</td><td>NaN (NaN) -</td><td>NaN (NaN) -</td><td>NaN (NaN) -</td><td>NaN (NaN) -</td><td>NaN (NaN) -</td><td>1.8821e-1 (1.95e-1) ≈</td><td>NaN (NaN) -</td><td>NaN (NaN) -</td><td>4.8626e-1 (3.10e-1)</td></tr><tr><td rowspan="2">Synchronous Optimal Pulse-width Modulation of 7-level Inverters</td><td>7.7564e-1 (4.57e-2) +/-/-/-/-</td><td>5.9497e-1 (2.33e-1) ≈</td><td>6.6799e-1 (2.22e-1) +</td><td>7.6749e-1 (5.34e-2) +</td><td>6.0888e-1 (3.05e-1) ≈</td><td>5.6533e-1 (3.54e-1) ≈</td><td>7.6420e-1 (5.20e-2) +</td><td>5.4489e-1 (1.60e-1) ≈</td><td>2.9724e-1 (3.10e-1) -</td><td>4.9731e-1 (1.74e-1)</td></tr><tr><td>1/7/2</td><td>1/6/3</td><td>1/6/3</td><td>2/6/2</td><td>1/8/1</td><td>0/7/3</td><td>1/4/5</td><td>1/7/2</td><td>0/9/1</td><td></td></tr></table>


“NaN (NaN)” indicates that the algorithm failed to find any feasible solution across 30 independent runs. 


## V. CONCLUSION

In this paper, we proposed a dual-swarm evolutionary algorithm with collaborative orthogonal learning, named DSO-COL, to effectively balance feasibility, convergence and diversity in constrained multi-objective optimization with complex landscapes. Unlike conventional collaborative frameworks that primarily exchange solution coordinates, DSOCOL incorporates evolutionary direction learning into the collaborative paradigm. The main swarm identifies promising convergence directions and transfers them as reusable search experience to the auxiliary swarm, enabling rapid convergence toward the CPF and achieving uniform search of the CPF. The primary contribution is the geometric synergy between trend learning and orthogonal learning. Specifically, trend learning drives rapid convergence toward the CPF, whereas orthogonal learning explores the orthogonal complement of the convergence direction, thereby reducing redundant searches and broadening the solution distribution. Moreover, the proposed NGSS strategy employs a three-level subset division mechanism to enable the preferential exploration of sparse niches, ensuring uniform distribution within the objective space. 

Extensive results on 33 benchmark instances and 10 realworld problems demonstrate significant improvements in both convergence and distribution, including large-scale cases. In practical applications such as pressure vessel design and reactor network design, DSOCOL provides decision-makers with a well-converged and well-distributed set of trade-off solutions, facilitating more informed trade-off decisions in real-world cases with nonlinear constraints. Despite these advancements, extending the collaborative learning paradigm to constrained many-objective optimization problems (CMaOPs, M > 3) 

remains challenging, as traditional Pareto dominance-based selection mechanisms perform poorly in this scenario. Future research will investigate performing orthogonal searches on Cartesian products to address the exponential growth in the cardinality of non-dominated solution sets in CMaOPs [63]. By reducing the challenge from an exponential to a polynomial complexity, it is promising to expand DSOCOL into CMaOPs. 

## REFERENCES



[1] R. Li, L. Wang, H. Sang, and L. Yao, “Knowledge-guided multiview hierarchical evolutionary algorithm for flexible job shop scheduling with finite skilled workers,” IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 55, no. 10, pp. 7259–7272, 2025. 





[2] Y. Shan, X.-P. Xie, and Z. Mao, “Co-design of a switching-type control scheme for nonlinear networked systems with protocol-based communication and its application to circuits,” IEEE Transactions on Automation Science and Engineering, vol. 22, pp. 5828–5840, 2025. 





[3] T. Zhang, D. Li, Y. Li, and W. Gong, “Constrained multitasking optimization via co-evolution and domain adaptation,” Swarm and Evolutionary Computation, vol. 87, p. 101570, 2024. 





[4] F. Ming, W. Gong, B. Xue, M. Zhang, and Y. Jin, “Automated configuration of evolutionary algorithms via deep reinforcement learning for constrained multiobjective optimization,” IEEE Transactions on Cybernetics, vol. 55, no. 12, pp. 5877–5890, 2025. 





[5] J. Liu, Y. Wang, G. Sun, and T. Pang, “Constrained evolutionary bayesian optimization for expensive constrained optimization problems with inequality constraints,” IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 55, no. 3, pp. 2009–2021, 2025. 





[6] K. Yu, F. Chen, M. Yu, J. Liang, and K. Chen, “Modal detection informed classification evaluation via ensemble networks for expensive constrained multimodal optimization,” IEEE Transactions on Neural Networks and Learning Systems, pp. 1–15, 2025. 





[7] H. Wu, Q. Chen, J. Chen, Y. Jin, J. Ding, X. Zhang, and T. Chai, “A multistage expensive constrained multiobjective optimization algorithm based on ensemble infill criterion,” IEEE Transactions on Evolutionary Computation, vol. 29, no. 6, pp. 2357–2371, 2025. 





[8] K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, “A fast and elitist multiobjective genetic algorithm: NSGA-II,” IEEE Transactions on Evolutionary Computation, vol. 6, no. 2, pp. 182–197, 2002. 





[9] Q. Zhu, Q. Zhang, and Q. Lin, “A constrained multiobjective evolutionary algorithm with detect-and-escape strategy,” IEEE Transactions on Evolutionary Computation, vol. 24, no. 5, pp. 938–947, Oct. 2020. 





[10] Z. Fan, H. Li, C. Wei, W. Li, H. Huang, X. Cai, and Z. Cai, “An improved epsilon constraint handling method embedded in moea/d for constrained multi-objective optimization problems,” in 2016 IEEE Symposium Series on Computational Intelligence (SSCI), 2016, pp. 1–8. 





[11] F. Ming, W. Gong, L. Wang, and L. Gao, “Constrained multiobjective optimization via multitasking and knowledge transfer,” IEEE Transactions on Evolutionary Computation, vol. 28, no. 1, pp. 77–89, 2024. 





[12] Z. Sun, H. Ren, G. G. Yen, T. Chen, J. Wu, H. An, and J. Yang, “An evolutionary algorithm with constraint relaxation strategy for highly constrained multiobjective optimization,” IEEE Transactions on Cybernetics, vol. 53, no. 5, pp. 3190–3204, 2023. 





[13] K. Qiao, K. Yu, B. Qu, J. Liang, H. Song, and C. Yue, “An evolutionary multitasking optimization framework for constrained multiobjective optimization problems,” IEEE Transactions on Evolutionary Computation, vol. 26, no. 2, pp. 263–277, 2022. 





[14] J. Wang, G. Liang, and J. Zhang, “Cooperative differential evolution framework for constrained multiobjective optimization,” IEEE Transactions on Cybernetics, vol. 49, no. 6, pp. 2060–2072, 2019. 





[15] Z. Fan, W. Li, X. Cai, H. Li, C. Wei, Q. Zhang, K. Deb, and E. Goodman, “Push and pull search for solving constrained multi-objective optimization problems,” Swarm and Evolutionary Computation, vol. 44, pp. 665– 679, 2019. 





[16] W.-Q. Ying, W.-P. He, Y.-X. Huang, D.-T. Li, and Y. Wu, “An adaptive stochastic ranking mechanism in moea/d for constrained multi-objective optimization,” in 2016 International Conference on Information System and Artificial Intelligence (ISAI), 2016, pp. 514–518. 





[17] M. A. Jan and R. A. Khanum, “A study of two penalty-parameterless constraint handling techniques in the framework of moea/d,” Applied Soft Computing, vol. 13, no. 1, pp. 128–148, 2013. 





[18] Q. Gu, Q. Wang, N. N. Xiong, S. Jiang, and L. Chen, “Surrogate-assisted evolutionary algorithm for expensive constrained multi-objective discrete optimization problems,” Complex & Intelligent Systems, vol. 8, pp. 2699 – 2718, 2021. 





[19] Z.-Z. Liu, Y. Wang, and B.-C. Wang, “Indicator-based constrained multiobjective evolutionary algorithms,” IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 51, no. 9, pp. 5414–5426, 2021. 





[20] Z. Ma and Y. Wang, “Shift-based penalty for evolutionary constrained multiobjective optimization and its application,” IEEE Transactions on Cybernetics, vol. 53, no. 1, pp. 18–30, 2023. 





[21] L. Jiao, J. Luo, R. Shang, and F. Liu, “A modified objective function method with feasible-guiding strategy to solve constrained multiobjective optimization problems,” Applied Soft Computing, vol. 14, pp. 363–380, 2014. 





[22] K. Yu, J. Liang, B. Qu, Y. Luo, and C. Yue, “Dynamic selection preference-assisted constrained multiobjective differential evolution,” IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 52, no. 5, pp. 2954–2965, 2022. 





[23] F. Vaz, Y. Lavinas, C. Aranha, and M. Ladeira, “Exploring constraint handling techniques in real-world problems on moea/d with limited budget of evaluations,” in Evolutionary Multi-Criterion Optimization, H. Ishibuchi, Q. Zhang, R. Cheng, K. Li, H. Li, H. Wang, and A. Zhou, Eds. Cham: Springer International Publishing, 2021, pp. 555–566. 





[24] Y. Tian, Y. Zhang, Y. Su, X. Zhang, K. C. Tan, and Y. Jin, “Balancing objective optimization and constraint satisfaction in constrained evolutionary multiobjective optimization,” IEEE Transactions on Cybernetics, vol. 52, no. 9, pp. 9559–9572, 2022. 





[25] Y. Xiang, X. Yang, H. Huang, and J. Wang, “Balancing constraints and objectives by considering problem types in constrained multiobjective optimization,” IEEE Transactions on Cybernetics, vol. 53, no. 1, pp. 88–101, 2023. 





[26] B. Liu, H. Ma, X. Zhang, and Y. Zhou, “A memetic co-evolutionary differential evolution algorithm for constrained optimization,” in 2007 IEEE Congress on Evolutionary Computation, 2007, pp. 2996–3002. 





[27] Y. Yang, J. Liu, and S. Tan, “A partition-based constrained multiobjective evolutionary algorithm,” Swarm and Evolutionary Computation, vol. 66, p. 100940, 2021. 





[28] F. Ming, W. Gong, D. Li, L. Wang, and L. Gao, “A competitive and cooperative swarm optimizer for constrained multiobjective optimization problems,” IEEE Transactions on Evolutionary Computation, vol. 27, no. 5, pp. 1313–1326, 2023. 





[29] X. Yu, X. Yu, Y. Lu, G. G. Yen, and M. Cai, “Differential evolution mutation operators for constrained multi-objective optimization,” Applied Soft Computing, vol. 67, pp. 452–466, 2018. 





[30] F. Ming, W. Gong, L. Wang, and Y. Jin, “Constrained multi-objective optimization with deep reinforcement learning assisted operator selection,” IEEE/CAA Journal of Automatica Sinica, vol. 11, no. 4, pp. 919–931, 2024. 





[31] M. Miyakawa, K. Takadama, and H. Sato, “Directed mating using inverted pbi function for constrained multi-objective optimization,” in 2015 IEEE Congress on Evolutionary Computation (CEC), 2015, pp. 2929–2936. 





[32] C. He, R. Cheng, Y. Tian, X. Zhang, K. C. Tan, and Y. Jin, “Paired offspring generation for constrained large-scale multiobjective optimization,” IEEE Transactions on Evolutionary Computation, vol. 25, no. 3, pp. 448–462, 2021. 





[33] A. Gupta, Y.-S. Ong, and L. Feng, “Insights on transfer optimization: Because experience is the best teacher,” IEEE Transactions on Emerging Topics in Computational Intelligence, vol. 2, no. 1, pp. 51–64, 2018. 





[34] K. Qiao, K. Yu, B. Qu, J. Liang, H. Song, C. Yue, H. Lin, and K. C. Tan, “Dynamic auxiliary task-based evolutionary multitasking for constrained multiobjective optimization,” IEEE Transactions on Evolutionary Computation, vol. 27, no. 3, pp. 642–656, 2023. 





[35] E. Zitzler, M. Laumanns, and L. Thiele, “Spea2: Improving the strength pareto evolutionary algorithm,” TIK-Report, vol. 103, 07 2001. 





[36] R. Cheng and Y. Jin, “A competitive swarm optimizer for large scale optimization,” IEEE Transactions on Cybernetics, vol. 45, no. 2, pp. 191–204, 2015. 





[37] K. Qiao, J. Liang, K. Yu, X. Ban, C. Yue, B. Qu, and P. N. Suganthan, “Constraints separation based evolutionary multitasking for constrained multi-objective optimization problems,” IEEE/CAA Journal ofAutomatica Sinica, vol. 11, no. 8, pp. 1819–1835, 2024. 





[38] Z. Fan, W. Li, X. Cai, H. Li, C. Wei, Q. Zhang, K. Deb, and E. Goodman, “Difficulty adjustable and scalable constrained multiobjective test problem toolkit,” Evolutionary Computation, vol. 28, no. 3, pp. 339–378, 2020. 





[39] H. Jain and K. Deb, “An evolutionary many-objective optimization algorithm using reference-point based nondominated sorting approach, part ii: Handling constraints and extending to an adaptive approach,” IEEE Transactions on Evolutionary Computation, vol. 18, no. 4, pp. 602–622, 2014. 





[40] K. Li, R. Chen, G. Fu, and X. Yao, “Two-archive evolutionary algorithm for constrained multiobjective optimization,” IEEE Transactions on Evolutionary Computation, vol. 23, no. 2, pp. 303–315, 2019. 





[41] Z. Fan, W. Li, X. Cai, H. Huang, Y. Fang, Y. Yugen, J. Mo, C. Wei, and E. Goodman, “An improved epsilon constraint-handling method in moea/d for cmops with large infeasible regions,” Soft Computing, vol. 23, pp. 12 491–12 510, 2019. 





[42] J. Yuan, H. Liu, Y.-S. Ong, and Z. He, “Indicator-based evolutionary algorithm for solving constrained multiobjective optimization problems,” IEEE Transactions on Evolutionary Computation, vol. 26, no. 2, pp. 379–391, 2022. 





[43] Y. Tian, R. Cheng, X. Zhang, and Y. Jin, “PlatEMO: A MATLAB platform for evolutionary multi-objective optimization,” IEEE Computational Intelligence Magazine, vol. 12, pp. 73–87, 11 2017. 





[44] Y. Tian, R. Wang, Y. Zhang, and X. Zhang, “Adaptive population sizing for multi-population based constrained multi-objective optimization,” Neurocomputing, vol. 621, p. 129296, 2025. 





[45] R. Sun, J. Zou, Y. Liu, S. Yang, and J. Zheng, “A multistage algorithm for solving multiobjective optimization problems with multiconstraints,” IEEE Transactions on Evolutionary Computation, vol. 27, no. 5, pp. 1207–1219, 2023. 





[46] L. R. C. de Farias and A. F. R. Araújo, “An inverse modeling constrained multi-objective evolutionary algorithm based on decomposition,” 2024 IEEE International Conference on Systems, Man, and Cybernetics (SMC), pp. 3727–3732, 2024. [Online]. Available: https://api.semanticscholar.org/CorpusID:273638540 





[47] X. Ban, J. Liang, K. Qiao, K. Yu, Y. Wang, J. Peng, and B. Qu, “A decision variables classification-based evolutionary algorithm for constrained multi-objective optimization problems,” IEEE/CAA Journal of Automatica Sinica, vol. 12, no. 9, pp. 1830–1849, 2025. 





[48] L. Si, X. Zhang, Y. Zhang, S. Yang, and Y. Tian, “An efficient sampling approach to offspring generation for evolutionary large-scale constrained multi-objective optimization,” IEEE Transactions on Emerging Topics in Computational Intelligence, vol. 9, no. 3, pp. 2080–2092, 2025. 





[49] K. Deb, R. B. Agrawal et al., “Simulated binary crossover for continuous search space,” Complex systems, vol. 9, no. 2, pp. 115–148, 1995. 





[50] P. Bosman and D. Thierens, “The balance between proximity and diversity in multiobjective evolutionary algorithms,” IEEE Transactions on Evolutionary Computation, vol. 7, no. 2, pp. 174–188, 2003. 





[51] E. Zitzler and L. Thiele, “Multiobjective evolutionary algorithms: a comparative case study and the strength pareto approach,” IEEE Transactions on Evolutionary Computation, vol. 3, no. 4, pp. 257–271, 1999. 





[52] J. Alcalá-Fdez, L. Sánchez, S. García, M. J. del Jesus, S. Ventura, J. M. Garrell, J. Otero, C. Romero, J. Bacardit, V. M. Rivas, J. C. Fernández, and F. Herrera, “KEEL: A software tool to assess evolutionary algorithms for data mining problems,” Soft Comput., vol. 13, no. 3, pp. 307–318, 2009. 





[53] B. Kannan and S. N. Kramer, “An augmented lagrange multiplier based method for mixed integer discrete continuous optimization and its applications to mechanical design,” Journal of mechanical design, vol. 116, no. 2, pp. 405–411, 1994. 





[54] F. Y. Cheng and X. S. Li, “A generalized center method for multiobjective optimization,” 1997. 





[55] L. Fan, T. Yoshino, T. Xu, Y. Lin, and H. Liu, “A novel hybrid algorithm for solving multiobjective optimization problems with engineering applications,” Mathematical Problems in Engineering, vol. 2018, no. 1, p. 5316379, 2018. 





[56] G. Guillén-Gosálbez, “A novel milp-based objective reduction method for multi-objective optimization: Application to environmental problems,” Computers & Chemical Engineering, vol. 35, no. 8, pp. 1469– 1477, 2011. 





[57] H. Ryoo and N. Sahinidis, “Global optimization of nonconvex nlps and minlps with applications in process design,” Computers & Chemical Engineering, vol. 19, no. 5, pp. 551–566, 1995. 





[58] G. R. Kocis and I. E. Grossmann, “Global optimization of nonconvex mixed-integer nonlinear programming (minlp) problems in process synthesis,” Industrial & engineering chemistry research, vol. 27, no. 8, pp. 1407–1421, 1988. 





[59] G. Kocis and I. Grossmann, “A modelling and decomposition strategy for the minlp optimization of process flowsheets,” Computers & Chemical Engineering, vol. 13, no. 7, pp. 797–819, 1989. 





[60] C. A. Floudas, Nonlinear and Mixed-Integer Optimization: Fundamentals and Applications. Oxford University Press, 11 1995. 





[61] A. K. Rathore, J. Holtz, and T. Boller, “Synchronous optimal pulsewidth modulation for low-switching-frequency control of medium-voltage multilevel inverters,” IEEE Transactions on Industrial Electronics, vol. 57, no. 7, pp. 2374–2381, 2010. 





[62] A. Edpuganti and A. K. Rathore, “Fundamental switching frequency optimal pulsewidth modulation of medium-voltage cascaded seven-level inverter,” IEEE Transactions on Industry Applications, vol. 51, no. 4, pp. 3485–3492, 2015. 





[63] S. Y. Zeng, L. S. Kang, and L. X. Ding, “An orthogonal multi-objective evolutionary algorithm for multi-objective optimization problems with constraints,” Evolutionary Computation, vol. 12, no. 1, pp. 77–98, 2004. 



![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/3e831ea9-6727-4c0e-809d-d04660adc717/f4f2ce7e051052ba59f4b6a3252bbece06b503f03b82dc54e281a37430802596.jpg)



Yubo Wang received the B.Sc. degree in computer science from South-Central Minzu University, Wuhan, China, in 2020. He is currently pursuing the Ph.D. degree in computer science with the School of Computer Science, China University of Geosciences, Wuhan, China.



His current research interests include evolutionary multiobjective optimization methods, and their applications.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/3e831ea9-6727-4c0e-809d-d04660adc717/6d07abff38832ae81ede8e3a7f60a1ab91079d331e1e9eac8215ed4610c10084.jpg)



Chengyu Hu received his M.S. degree in automation and control from Wuhan University of Technology in 2003 and his Ph.D. in automation control from Huazhong University of Science and Technology in 2010. He is currently a professor and vice dean at the School of Computer Science, China University of Geosciences, Wuhan, China. His research interests include evolutionary algorithms, reinforcement learning, and cloud computing.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/3e831ea9-6727-4c0e-809d-d04660adc717/061b497f1c33b0b3d59538feccb6a1a5ab41d81b6a11415a2229c823a804f82f.jpg)


Xinyi Wu received the B.Sc. degree from Xidian University, Xi’an, China, in 2019 and M.Sc. degree from New York University. She is currently pursuing the Ph.D. degree in computer science with the School of Computer Science, China University of Geosciences, Wuhan, China. 

Her current research interests include evolutionary multimodal and multiobjective optimization methods, and their applications. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/3e831ea9-6727-4c0e-809d-d04660adc717/8d0a44d59b8cd2bcece6415c76fbf1801d5c477a29acb730d1c921db2968681f.jpg)


Tingyu Zhang received the B.Sc. degree in computer science from Wuhan Institute of Technology, Wuhan, China, in 2022. He is currently pursuing the Ph.D. degree in computer science with the School of Computer Science, China University of Geosciences, Wuhan, China. 

His current research interests include evolutionary multitask optimization, evolutionary multiobjective optimization, and their applications. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/3e831ea9-6727-4c0e-809d-d04660adc717/0a85ed9feb25fa6b52a973caf053f5adcdb4bcb5cf2199882196dbdd470df1ff.jpg)


Wenyin Gong (Senior Member, IEEE) received the B.Eng., M.Eng., and Ph.D. degrees in computer science from China University of Geosciences, Wuhan, China, in 2004, 2007, and 2010, respectively. 

ences. 

He is currently a Professor with School of Computer Science, China University of Geosciences, Wuhan, China. His research interests include evolutionary algorithms, evolutionary optimization, and their applications. He has published over 100 research papers in journals and international confer-

He served as a referee for over 30 international journals, such as IEEE TRANSACTIONS ON EVOLUTIONARY COMPUTATION, IEEE TRANSAC-TIONS ON CYBERNETICS, IEEE TRANSACTIONS ON SYSTEMS, MAN, AND CYBERNETICS: SYSTEMS, IEEE Computational Intelligence Magazine, ACM Transactions on Intelligent Systems and Technology, Information Sciences, European Journal of Operational Research, Applied Soft Computing, Journal of Power Sources, etc. Professor Gong currently serves as Associate Editor of Swarm and Evolutionary Computation, Expert Systems with Applications, Memetic Computing, etc. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/3e831ea9-6727-4c0e-809d-d04660adc717/9a4a5b41252fb37cc697d22154ba261a3f0607fbcca13de76e8b3e7b49dd8919.jpg)


Xuesong Yan received the B.Eng and M.Eng degrees in computer science from the China University of Geosciences, Wuhan, China, in 2000, and 2003, respectively, and his Ph.D. degree in computer software and theory from Wuhan University in 2006. 

He is currently a Professor in the School of Computer Science, China University of Geosciences, Wuhan, China. His research interests include intelligent optimization, artificial intelligence, and big data and its application. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/3e831ea9-6727-4c0e-809d-d04660adc717/bc5530319dfc6b37061f8445d593d29787c99070bb6e89d684effb6235ad80de.jpg)
Liang Gao (Senior Member, IEEE) received his B.Sc. degree in mechatronic engineering from Xidian University, Xi an, China, in 1996, and the Ph.D. degree in mechatronic engineering from Huazhong University of Science and Technology (HUST), Wuhan, China, in 2002. 
He is a Professor of the Department of Industrial and Manufacturing Systems Engineering (IMSE), the Deputy Director of State Key Laboratory of Digital Manufacturing Equipment and Technology and Chairman of School of Mechanical Science and 

Engineering, HUST. He was supported by the Program for New Century Excellent Talents in University in 2008 and National Science Fund for Distinguished Young Scholars of China in 2018. His research interests include intelligent optimization algorithms, big data, deep learning with theirs application in Design & Manufacturing. He published more than 450 papers indexed by SCIE, authored 13 monographs. 

Professor Gao currently serves as co-Editor-in-Chief of IET Collaborative Intelligent Manufacturing, Associate Editor of Swarm and Evolutionary Computation, Journal of Industrial and Production Engineering, etc. 