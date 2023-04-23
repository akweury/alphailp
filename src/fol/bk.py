target_predicate = [
    'kp:1:image'
]

neural_predicate = [

]

neural_predicate_2 = [
    'in:2:group,image',
    'color:2:group,color',
    'shape:2:group,shape',
    'phi:2:group,phi',
    'rho:2:group,rho',

]

neural_predicate_3 = [
    'group_shape:2:group,group_shape',
]

const_dict = {
    'image': 'target',
    'color': 'enum',
    'shape': 'enum',
    'group': 'amount_e',
    'phi': 'amount_4',
    'rho': 'amount_4',
}

color = ['pink', 'green', 'blue']
shape = ['sphere', 'cube', 'line', 'circle']
