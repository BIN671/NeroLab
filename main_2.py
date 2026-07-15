import FC.Layers as classes
import FC.functions as functions
import FC.error_func as error_func
import random

input_layer = classes.InputLayer(2)
x1 = classes.Dense(3, functions.Relu, input_layer)
output_layer = classes.OutputLayer(1, functions.equal, x1)


model = classes.Model(input_layer, output_layer)
model.compile()

print(model.predict([1]))

print(output_layer.weights_matrix)

"""
n = 1000

for i in range(1000):
    x = random.randint(1, n*2)
    x = x/n
    y = (2*x)+50
    model.learn([x], [y], error_func.MSE, 0.01*(n**0.3))
    #print(output_layer.weights_matrix)

print(output_layer.weights_matrix)


"""

n = 70
print(0.01*(n**0.3))
for i in range(10000):
    x = random.randint(1, 2*n) / n
    y = random.randint(1, 2*n) / n
    #R = (x+y)+2*(x+y)+3*(x+6*y)+4
    R = x*4+y**2
    model.learn([x, y], R, error_func.MSE, 0.001 * (n ** 0.3))

#print(model.predict([10,3]))


#print(model.predict([100]))
#print((2*100+30))
print(output_layer.weights_matrix)
print(model.predict([10,3]))
"""
for x in range(15):
    for y in range(20):
        r = 3 + x + 2*y
        model.learn([x,y], [r], error_func.MSE)

print(model.predict([1,3]))
print(output_layer.weights_matrix)
"""