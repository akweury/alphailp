target_predicate = [
    'kp:1:image'
]

neural_predicate = [
    'in:2:group,image'
]

neural_predicate_2 = [
    'color:2:group,color',
    'shape:2:group,shape',
    'phi:2:group,phi',
    'rho:2:group,rho',

]

neural_predicate_3 = [
    'group_shape:2:group,group_shape',
]

consts = {
    'image': 'target',
    'color': 'enum',
    'shape': 'enum',
    'group_shape': 'enum',
    'group': 'amount_e',
    'phi': 'amount_4',
    'rho': 'amount_4',
}

color = ['pink', 'green', 'blue']
shape = ['sphere', 'cube']
group_shape = ['line', 'circle']
