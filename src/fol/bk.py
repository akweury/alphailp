target_predicate = [
    'kp:1:image',
    'in:2:group,image'
]

neural_predicate = [

]

neural_predicate_2 = [
    'shape:2:group,shape',
    'color:2:group,color',
    'phi:3:group,group,phi',
    'rho:3:group,group,rho',
    'slope:2:group,slope',
]

neural_predicate_3 = [
    'group_shape:2:group,group_shape',
]

const_dict = {
    'image': 'target',
    'color': 'enum',
    'shape': 'enum',
    'group': 'amount_e',
    'phi': 'amount_8',
    'rho': 'amount_8',
    'slope': 'amount_8',
}

attr_names = ['color', 'shape', 'rho', 'phi', 'group_shape', "slope"]
color = ['pink', 'green', 'blue']
shape = ['sphere', 'cube', 'line', 'circle']
