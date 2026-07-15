import FC.Layers as classes
import FC.functions as functions
import FC.error_func as error_func


input_layer = classes.InputLayer(2)
x1 = classes.Dense(3, functions.Relu, input_layer)
x2 = classes.Dense(20, functions.Relu, x1)
x3 = classes.Dense(3, functions.Relu, x2)
output_layer = classes.OutputLayer(3, functions.equal, x3)


model = classes.Model(input_layer, output_layer, error_func.MSE)
model.compile()


r = model.predict([1,3])
print(r.size)

