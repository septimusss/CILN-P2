# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        # si el estado siguiente es ganador devolvemos la maxima puntuacion
        if successorGameState.isWin(): return float("+inf")
        # precalculamos la distancia al fantasma
        d2ghost = util.manhattanDistance(currentGameState.getGhostPosition(1), newPos)
        # por defecto devolvemos el score del estado
        total_score = successorGameState.getScore()
        # cuanto mayor es la distancia a la comida mas cercana peor score
        closest_food = float("inf")
        for food_position in newFood.asList():
            d = util.manhattanDistance(food_position, newPos)
            closest_food = min(closest_food, d)
        total_score -= 5 * closest_food
        # si en el siguiente estado hay menos comida significa que hemos
        # comido, damos mejor score
        if (currentGameState.getNumFood() > successorGameState.getNumFood()):
            total_score += 120
        # si tenemos un fantasma muy cerca retornamos un score muy malo
        # de esta forma evitamos al fantasma cuando lo tenemos muy cerca
        if d2ghost < 3: total_score -= 500 * d2ghost
        return total_score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        # funcion que representa el paso de max-value del algoritmo
        # mira el score de cada uno de sus hijos y retorna el mayor de ellos
        def maxVal(state, depth = 0, agent = 0):
            # si el estado es ganador, perdedor o hemos llegado el limit de
            # profundidad, retornamos el score asignado a este estado
            if state.isWin() or state.isLose() or (depth == self.depth):
                return self.evaluationFunction(state)
            max = (float("-inf"), 'Stop')
            # generamos los hijos a partir de las acciones validas
            for a in state.getLegalActions(agent):
                successor = state.generateSuccessor(agent, a)
                v = (minVal(successor, depth, agent + 1), a)
                if v[0] > max[0]: max = v
            # si depth == 0 quiere decir que estamos en la llamada inicial de la
            # funcion, asi que en vez de retorna el score, retornamos la accion
            if depth == 0: return max[1]
            else: return max[0]
        # funcion que representa el paso de min-value del algoritmo
        # mira el score de cada uno de sus hijos y devuelve el menor de ellos
        def minVal(state, depth = 0, agent = 0):
            # casi igual a la funcion maxVal, con un pequeno cambio
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            min = float("inf")
            for a in state.getLegalActions(agent):
                successor = state.generateSuccessor(agent,a)
                # para soportar multiples fantasmas tenemos que ejecutar la capa
                # min para cada uno de los fantasmas, si ya estamos en el ultimo
                # fantasma volvemos a ejecutar el maxVal (pacman) con depth += 1
                if agent < state.getNumAgents() - 1:
                    v = minVal(successor, depth, agent + 1)
                else:
                    v = maxVal(successor, depth + 1, 0)
                if v < min: min = v
            return min
        # para obtener el resultado basta con ejecutar maxVal con el estado como
        # argumento, las llamadas recursivas haran el resto del trabajo
        return maxVal(gameState)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def maxVal(state, depth = 0, agent = 0, alpha = float("-inf"), beta = float("inf")):
            if state.isWin() or state.isLose() or (depth == self.depth):
                return self.evaluationFunction(state)
            max = (float("-inf"), 'Stop')
            for a in state.getLegalActions(agent):
                successor = state.generateSuccessor(agent, a)
                v = (minVal(successor, depth, agent + 1, alpha, beta), a)
                if v[0] > max[0]: max = v
                # todo es igual a minimax, lo unico que cambia es este trozo
                # ahora vamos pasando los valores de alpha y beta para hacer
                # prunning cuando se cumpla la condicion
                if v[0] > beta: return v[0]
                alpha = v[0] if v[0] > alpha else alpha
            if depth == 0: return max[1]
            else: return max[0]

        def minVal(state, depth, agent, alpha, beta):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            min = float("inf")
            for a in state.getLegalActions(agent):
                successor = state.generateSuccessor(agent,a)
                if agent < state.getNumAgents() - 1:
                    v = minVal(successor, depth, agent + 1, alpha, beta)
                else:
                    v = maxVal(successor, depth + 1, 0, alpha, beta)
                if v < min: min = v
                # similar al maxVal, en este caso miramos el valor de alpha para
                # decidir si hacemos prunning o no
                if v < alpha: return v
                beta = v if v < beta else beta
            return min
        # para obtener el resultado basta con ejecutar maxVal con el estado como
        # argumento, las llamadas recursivas haran el resto del trabajo
        return maxVal(gameState)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        def maxVal(state, depth = 0, agent = 0):
            if state.isWin() or state.isLose() or (depth == self.depth):
                return self.evaluationFunction(state)
            max = (float("-inf"), 'Stop')
            for a in state.getLegalActions(agent):
                successor = state.generateSuccessor(agent, a)
                v = (minVal(successor, depth, agent + 1), a)
                if v[0] > max[0]: max = v
            if depth == 0: return max[1]
            else: return max[0]

        def minVal(state, depth = 0, agent = 0):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            # casi igual a Q2, ahora en vez de retorna el score menor para cada
            # fantasma, retornamos una media de todas las posibles acciones
            min = set()
            for a in state.getLegalActions(agent):
                successor = state.generateSuccessor(agent,a)
                if agent < state.getNumAgents() - 1:
                    v = minVal(successor, depth, agent + 1)
                else:
                    v = maxVal(successor, depth + 1, 0)
                min.add(v)
            return sum(min)/float(len(min))

        return maxVal(gameState)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # precalculamos algunas variables
    pacmanPos = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    food = currentGameState.getFood()
    # por defect retornamos el score propio del estado
    score = currentGameState.getScore()
    # si el estado es ganador retornamos directamente un score muy bueno
    if currentGameState.isWin(): return 100 + score
    # cuanta mas comida queda por comer menos score
    if food.count() > 0: score += (1.0 / food.count()) * 20
    # si un fantasma esta asustado nos lo comemos, sino huimos
    for ghost in ghostStates:
        ghostPos = ghost.getPosition()
        if ghost.scaredTimer < 1:
            if manhattanDistance(pacmanPos, ghostPos) == 1:
                score = -100
        else:
            score += (1.0 / manhattanDistance(pacmanPos, ghostPos)) * 10
    # cuanto mas cerca estemos de la comida mas cercana mas score
    d2f = float("inf")
    for f in food.asList():
        d2f = min(d2f, manhattanDistance(pacmanPos, f))
    score -= d2f

    return score

# Abbreviation
better = betterEvaluationFunction
