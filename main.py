import argparse
from mimetypes import init
import time
from board import Board
from input import get_start_board


def cal_cost(curr_state, next_state):
    """
    calculate cost when move from curr_state to next_state

    agrs:
    curr_state: current state
    next_state: next state

    return:

    number of steps to move curr_state to next_state.
    ex: 00001111 -> 10001110 : 1 
    """

    count = 0
    for i in range(0, len(curr_state.string)):
        if curr_state.string[i] != next_state.string[i]:
            count += 1

    return count/2


def bfs(init_board):
    """BFS searching algorithm
    Args:
        init_board: initial board/state
    Returns:
        curr_state: one possible final state
        steps: total moving steps
    """
    queue = []
    visited = set()
    steps = 0
    queue.append(init_board)
    visited.add(init_board.string)

    start = time.time()
    while queue:
        size = len(queue)
        print(f"Step: {steps}, #States total: {len(queue)}")

        for _ in range(size):
            curr_state = queue.pop(0)
            if curr_state.state_checking() is True:
                print('Elapsed time: {:.4f}s'.format(time.time() - start))
                return curr_state, steps

            for state in curr_state.next_boards():
                state_str = state.string

                if state_str not in visited:
                    queue.append(state)
                    visited.add(state_str)

        steps += 1


def dfs(init_board):
    """DFS searching algorithm

    Args:
        init_board: initial board/state

    Returns:
        curr_state: one possible final state
    """
    stack = []
    stack.append(init_board)
    visited = set()

    start = time.time()
    while stack:
        curr_state = stack.pop()
       # print()
      #  print(curr_state.string)
        if curr_state.state_checking() is True:
            print('Elapsed time: {:.4f}s'.format(time.time() - start))
            return curr_state

        if curr_state.string not in visited:
            visited.add(curr_state.string)
            for next_state in curr_state.next_boards():
                stack.append(next_state)


def a_star(init_board):
    """ A* searching algorithm
        init_board: initial board/state

        heuristic function: h(state) = number of color remained
        cost function: g(state)  number of steps to move from init state to goal state


        Returns:
        curr_state: one possible final state
    """

    stack = []
    stack.append(init_board)
    cost = dict()
    visited = set()
    cost[init_board] = 0

    curr_state = init_board
    print(len(curr_state.next_boards()))

    if curr_state.state_checking() is True:
        return curr_state

    if curr_state.string not in visited:
        visited.add(curr_state.string)
        for next_state in curr_state.next_boards():
            stack.append(next_state)
            cost[next_state] = cost[curr_state] + 1
    visited.add(init_board.string)

    print(len(stack))
    while stack:

        # get state which is min g()+h() in list state
        i = 0
        while stack[i].string in visited:
            i += 1
        curr_state = stack[i]
        f_min = cost[curr_state] + curr_state.total_color_remain()
        for i in range(0, len(stack)):
            if stack[i].string not in visited and f_min <= cost[stack[i]] + stack[i].total_color_remain():
                curr_state = stack[i]
                f_min = cost[stack[i]] + stack[i].total_color_remain()

        if curr_state.state_checking() is True:
            return curr_state

        if curr_state.string not in visited:

            visited.add(curr_state.string)
            for next_state in curr_state.next_boards():
                stack.append(next_state)
                cost[next_state] = cost[curr_state] + 1


if __name__ == '__main__':
    # 1. prepare args parser
    parser = argparse.ArgumentParser('Choose searching method')
    parser.add_argument('--search', type=str, default='dfs')
    args = parser.parse_args()

    # 2. get initital game board
    start_board = Board(get_start_board(), parent=None)

    # 3. run searching algorithm
    if args.search == 'a_star':
        final_state = a_star(init_board=start_board)
    elif args.search == 'dfs':
        final_state = dfs(init_board=start_board)
    else:
        raise ValueError('Unvalid searching algorithm, please check it again.')
    print('final state string: {}'.format(final_state.string))

    # 4. get actions
    total_actions = []
    total_strings = []
    while final_state:
        total_strings.append(final_state.string)
        action = f"Move {final_state.from_which + 1} to {final_state.to_which + 1}"
        total_actions.append(action)
        final_state = final_state.parent

    for idx, data in enumerate(zip(total_strings[::-1], total_actions[::-1])):
        print(f"Step: {idx}, Action: {data[1]}")
