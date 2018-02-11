from grid_world_maker import GridWorldMaker

if __name__ == '__main__':
    definition = {
        'discount': 0.90,
        'values': 'reward',
        'states':  ' '.join(map(str, list(range(7*7)))),
        'actions': ' '.join(['up', 'down', 'left', 'right', 'halt']),
        'costs':   ' '.join(map(str, [1, 1, 1, 1, 0.25])),
        'observations': ' '.join(map(str, list(range(7*7)))),
        'observation_probability': 0.85, # the probability of O_s = S_s, i.e., an accurate observation 
        'init_state': '24',
        'board': [ # also defines the immediate reward map
            [-1, -1, 10, 10, -1, -1, -1],
            [-1, -1, -1, 10,  1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1],
            [20, -1, -1, -1, -1, -1, -1],
            [20, -1, -1, -1, -1, -1, 40],
            [-1, -1, -1, -1, -1, 40, 40]
        ],
        'action_map': lambda action, i,j : {
            'up': (i - 1, j),
            'down': (i + 1, j),
            'left': (i, j - 1),
            'right': (i, j + 1),
            'halt': (i, j)
        }.get(action)
    }

    maker = GridWorldMaker(definition)
    
    lines = []
    maker.make_meta(lines)
    maker.make_R(lines)
    maker.make_T(lines)
    maker.make_O(lines)

    with open('./pomdp/GridWorld-5D.POMDP', 'w+') as outfile:
        outfile.writelines(lines)
