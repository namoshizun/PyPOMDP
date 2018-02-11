class Board:
    def __init__(self, board):
        self.board = board
        self.h, self.w = len(self.board), len(self.board[0])

    def at(self, i, j):
        if i < 0 or j < 0:
            raise IndexError
        return self.board[i][j]

    def state(self, i, j):
        if i < 0 or j < 0 or i >= self.h or j >= self.w:
            raise IndexError
        return i * self.h + j

    def for_each_cell(self, visitor_fn):
        for i in range(self.h):
            for j in range(self.w):
                visitor_fn(i, j)
    
    def adjacent_cells(self, i, j):
        cells = []
        for y, x in [(i+dy, j+dx) for dy in (-1,0,1) for dx in (-1,0,1) if dx != 0 or dy != 0]:
            if y >= 0 and x >= 0 and y < self.h and x < self.w:
                cells.append((y, x))
        return cells


class GridWorldMaker():
    def __init__(self, configs):
        self.board = Board(configs['board'])
        self.configs = configs
        self.actions = configs['actions'].split(' ')
    
    def make_meta(self, lines):
        _ = self.configs
        templates = [
            'discount: {}'.format(_['discount']),
            'values: {}'.format(_['values']),
            'actions: {}'.format(_['actions']),
            'costs: {}'.format(_['costs']),
            'states: {}'.format(_['states']),
            'observations: {}'.format(_['observations']),
            'init_state: {}\n\n'.format(_['init_state']),
        ]
        lines.append('\n'.join(templates))

    def make_R(self, lines):
        def algorithm(action):
            def wrapper(i, j):
                try:
                    next_state = self.configs['action_map'](action, i, j)
                    reward = self.board.at(next_state[0], next_state[1])
                    if action == 'halt':
                        lines.append(template.format(a=action, si=self.board.state(i, j), r=0 if reward < 0 else reward))
                    else:

                        lines.append(template.format(a=action, si=self.board.state(i, j), r=reward))
                except IndexError:
                    lines.append(template.format(a=action, si=self.board.state(i, j), r=-10))
            return wrapper
        
        template = 'R: {a} : {si} : *  : *       {r}\n'
        for action in self.actions:
            self.board.for_each_cell(algorithm(action))


    def make_T(self, lines):
        def algorithm(action):
            def wrapper(i, j):
                try:
                    next_state = self.configs['action_map'](action, i, j)
                    lines.append(template.format(a=action, si=self.board.state(i, j),
                                                 sj=self.board.state(*next_state), p=1.0))
                except IndexError:
                    lines.append(template.format(a=action, si=self.board.state(i, j),
                                                 sj=self.board.state(i, j), p=1.0))
            return wrapper

        template = 'T: {a} : {si} : {sj}         {p}\n'
        for action in self.actions:
            self.board.for_each_cell(algorithm(action))


    def make_O(self, lines):
        def algorithm(action):
            def wrapper(i, j):
                obs_prob = self.configs['observation_probability']
                try:
                    next_point = self.configs['action_map'](action, i, j)
                    next_state = self.board.state(*next_point)
                    lines.append(template.format(a=action, sj=next_state,
                                                 oj=next_state, p=obs_prob))
                    neighbours = self.board.adjacent_cells(*next_point)
                    for y, x in neighbours:
                        lines.append(template.format(a=action, sj=next_state,
                                                     oj=self.board.state(y, x), p=(1.0 - obs_prob)/len(neighbours)))

                except IndexError:
                    pass
            return wrapper

        template = 'O: {a} : {sj} : {oj}         {p}\n'
        for action in self.actions:
            self.board.for_each_cell(algorithm(action))

