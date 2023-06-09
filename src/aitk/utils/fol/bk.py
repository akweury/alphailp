target_predicate = [
    'kp:1:image',
    'in:2:group,image'
]

neural_predicate = []

neural_predicate_2 = {
    # 'shape_counter': 'shape_counter:2:group,number',
    # 'color_counter': 'color_counter:2:group,number',
    'shape': 'shape:2:group,shape',
    # 'color': 'color:2:group,color',
    'phi': 'phi:3:group,group,phi',
    'rho': 'rho:3:group,group,rho',
    'slope': 'slope:2:group,slope',
}

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
    'slope': 'amount_8',
    'number': 'amount_10',
}

attr_names = ['color', 'shape', 'rho', 'phi', 'group_shape', "slope", 'number']

color = ['pink', 'green', 'blue']
shape = ['sphere', 'cube', 'line', 'circle']

pred_obj_mapping = {
    'in': None,
    'shape_counter': ["sphere", "cube"],
    'color_counter': ["red", "green", "blue"],
    'shape': ['sphere', 'cube', 'line', 'circle'],
    'color:': ['red', 'green', 'blue'],
    'phi': ['x', 'y', 'z'],
    'rho': ['x', 'y', 'z'],
    'slope': ['x', 'y', 'z'],
}

"""
(explaination)
line: average distance between any two objects next to each other, group size
or 
remember the distances as a list
"""

pred_pred_mapping = {
    'shape_counter': ['shape'],
    'color_counter': ['color'],
    'in': []
}
